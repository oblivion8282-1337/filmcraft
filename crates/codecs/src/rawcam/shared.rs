//! One shared bridge process per format: every [`super::RawCamSource`] of a bridge binary whose
//! probe advertises [`Caps::multi`] gets its decode session as a clip handle on the same
//! [`MultiBridge`] — one CUDA context and one set of SDK GPU pools for the whole timeline instead
//! of one per clip (each costs ~0.4 GB of VRAM before it decodes a single frame).
//!
//! The manager holds the bridge weakly: it lives while any clip session does, so when every
//! source has retired its sessions (idle) the process exits and its VRAM is gone. A bridge that
//! dies is started again on the next request — at once after one that worked, then backing off
//! (0.5 s doubling to 16 s) while it keeps dying without serving a frame. A clip that had frames
//! outstanding in two crashes is isolated: it gets a bridge process of its own (single-clip
//! mode), so one bad clip cannot keep taking every other clip down with it.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

use super::bridge::{ClipSession, DeathReport, MultiBridge, OpenFail};

/// First pause before a shared bridge that keeps dying is started again (doubling, capped).
const RESPAWN_BACKOFF: Duration = Duration::from_millis(500);
/// Crashes with frames of one clip outstanding before that clip gets a process of its own.
const ISOLATE_AFTER: u32 = 2;
/// Largest shared-memory slot asked for (a 12K RGBA frame; the bridges cap at 1 GiB).
const MAX_SLOT_BYTES: u64 = 512 << 20;

/// Why no clip session could be had from the shared bridge.
#[derive(Debug, PartialEq, Eq)]
pub enum SharedError {
    /// The shared bridge did not start: the caller may fall back to a process of its own.
    Spawn(String),
    /// The bridge is up but refused this clip.
    Clip(String),
    /// The bridge broke, or keeps breaking (backing off).
    Bridge(String),
}

#[derive(Default)]
struct State {
    current: Weak<MultiBridge>,
    /// The death report of `current` (outlives it: read when the next request finds it gone).
    report: Option<Arc<Mutex<Option<DeathReport>>>>,
    next_handle: u64,
    /// Consecutive deaths without a frame served in between.
    failures: u32,
    retry_at: Option<Instant>,
    /// Crashes each clip was involved in, and the clips isolated for it.
    suspects: HashMap<PathBuf, u32>,
    isolated: HashSet<PathBuf>,
    /// The largest frame (bytes) any clip of this bridge was opened for: the next bridge's slots.
    slot_bytes: u64,
}

impl State {
    /// Account for a bridge that is gone: crash suspects, back-off.
    fn harvest(&mut self, now: Instant) {
        let Some(report) = self.report.take() else { return };
        let Some(report) = report.lock().unwrap_or_else(PoisonError::into_inner).take() else { return };
        if !report.crashed {
            return;
        }
        for path in report.suspects {
            let n = self.suspects.entry(path.clone()).or_insert(0);
            *n = n.saturating_add(1);
            if *n >= ISOLATE_AFTER && self.isolated.insert(path.clone()) {
                log::warn!("{}: involved in {n} camera-RAW bridge crashes; decoding it in a process of its own", path.display());
            }
        }
        self.failures = if report.served { 1 } else { self.failures.saturating_add(1) };
        self.retry_at = (self.failures >= 2).then(|| now + RESPAWN_BACKOFF * (1_u32 << (self.failures - 2).min(5)));
    }

    fn backoff_left(&self, now: Instant) -> Option<Duration> {
        self.retry_at.and_then(|t| t.checked_duration_since(now)).filter(|d| !d.is_zero())
    }
}

/// The shared bridge of one bridge binary.
pub struct Manager {
    bridge: PathBuf,
    state: Mutex<State>,
}

/// The manager of `bridge` (one per binary for the process lifetime).
pub fn manager(bridge: &Path) -> Arc<Manager> {
    static MANAGERS: OnceLock<Mutex<HashMap<PathBuf, Arc<Manager>>>> = OnceLock::new();
    let mut all = MANAGERS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(PoisonError::into_inner);
    all.entry(bridge.to_path_buf()).or_insert_with(|| Arc::new(Manager { bridge: bridge.to_path_buf(), state: Mutex::new(State::default()) })).clone()
}

impl Manager {
    /// Whether `clip` was isolated after crashes (it must use a process of its own).
    pub fn isolated(&self, clip: &Path) -> bool {
        let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if st.current.upgrade().is_none_or(|b| !b.is_alive()) {
            st.harvest(Instant::now());
        }
        st.isolated.contains(clip)
    }

    /// A session for `clip` on the shared bridge, starting it if none runs. `frame_bytes`: the
    /// clip's largest frame (sizes the shared-memory slots of a bridge started now); `shm`:
    /// whether to offer the shared-memory ring at all.
    pub fn open(&self, clip: &Path, frame_bytes: u64, shm: bool, timeout: Duration) -> Result<ClipSession, SharedError> {
        let (bridge, handle) = {
            let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            st.slot_bytes = st.slot_bytes.max(frame_bytes.min(MAX_SLOT_BYTES));
            let live = st.current.upgrade().filter(|b| b.is_alive());
            let bridge = match live {
                Some(b) => b,
                None => {
                    let now = Instant::now();
                    st.harvest(now);
                    if let Some(left) = st.backoff_left(now) {
                        return Err(SharedError::Bridge(format!("{} keeps failing; trying again in {:.1} s", self.bridge.display(), left.as_secs_f64())));
                    }
                    // Started under the lock: whoever else asks meanwhile needs this very bridge.
                    let slots = (shm && st.slot_bytes > 0).then_some(st.slot_bytes);
                    match MultiBridge::spawn(&self.bridge, slots, timeout) {
                        Ok(b) => {
                            st.current = Arc::downgrade(&b);
                            st.report = Some(b.death_report());
                            b
                        }
                        Err(e) => {
                            st.failures = st.failures.saturating_add(1);
                            if st.failures >= 2 {
                                st.retry_at = Some(now + RESPAWN_BACKOFF * (1_u32 << (st.failures - 2).min(5)));
                            }
                            return Err(SharedError::Spawn(e));
                        }
                    }
                }
            };
            let handle = st.next_handle;
            st.next_handle = st.next_handle.wrapping_add(1);
            (bridge, handle)
        };
        match ClipSession::open(bridge, handle, clip, timeout) {
            Ok(session) => Ok(session),
            Err(OpenFail::Clip(e)) => Err(SharedError::Clip(e)),
            Err(OpenFail::Bridge(e)) => Err(SharedError::Bridge(e)),
        }
    }

    /// The bridge served frames again: consecutive deaths start counting from zero.
    pub fn note_success(&self) {
        let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if st.failures > 0 {
            st.failures = 0;
            st.retry_at = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(suspects: &[&str], served: bool, crashed: bool) -> Option<Arc<Mutex<Option<DeathReport>>>> {
        Some(Arc::new(Mutex::new(Some(DeathReport { suspects: suspects.iter().map(PathBuf::from).collect(), served, crashed }))))
    }

    /// A clip with frames outstanding in two crashes is isolated; the others stay shared. A
    /// bridge that served before dying is restarted at once, one that keeps dying backs off.
    #[test]
    fn crashes_isolate_the_suspect_and_back_off() {
        let t0 = Instant::now();
        let mut st = State { report: report(&["/a.braw", "/b.braw"], true, true), ..State::default() };
        st.harvest(t0);
        assert_eq!(st.backoff_left(t0), None, "it worked before: restart at once");
        assert!(st.isolated.is_empty());
        st.report = report(&["/a.braw"], false, true);
        st.harvest(t0);
        assert!(st.isolated.contains(Path::new("/a.braw")), "two crashes with /a outstanding");
        assert!(!st.isolated.contains(Path::new("/b.braw")));
        assert_eq!(st.backoff_left(t0), Some(RESPAWN_BACKOFF), "died again without serving: back off");
        for _ in 0..50 {
            st.report = report(&[], false, true);
            st.harvest(t0);
        }
        assert_eq!(st.backoff_left(t0), Some(RESPAWN_BACKOFF * 32), "capped");
        // A bridge its last user let go is no crash.
        let mut calm = State { report: report(&["/c.braw"], false, false), ..State::default() };
        calm.harvest(t0);
        assert_eq!((calm.failures, calm.suspects.len()), (0, 0));
    }
}
