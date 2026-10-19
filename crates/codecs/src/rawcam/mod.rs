//! Camera-RAW media: Blackmagic `.braw` and RED `.R3D` clips decoded by external bridge binaries
//! (`braw-bridge` / `r3d-bridge`) that wrap the vendor SDKs and stream debayered `rgb24` frames
//! (see [`bridge`]). The SDKs are licensed and never ship with FilmCraft, so the bridges are
//! user-provided: [`path_opener`] finds them via `FILMCRAFT_BRAW_BRIDGE` / `FILMCRAFT_R3D_BRIDGE`
//! or on `PATH`, and a missing bridge yields a clear "unsupported" error naming the variable.
//!
//! Frames are full-range sRGB RGB (the SDKs' processed output); audio is extracted once per file
//! into a cache WAV (`FILMCRAFT_RAWCAM_CACHE`, default `$XDG_CACHE_HOME/filmcraft/rawcam`) and
//! served through [`filmcraft_media::wav::WavSource`]. Decoding runs on a persistent bridge
//! session ([`FrameRequest::scale`] picks the debayer tier, so thumbnails debayer at
//! quarter/eighth resolution): a protocol 2 bridge takes the tier with every command, so one
//! session serves them all; an older one gets a session per tier (at most [`TIERS`]). A
//! read-ahead thread decodes [`PREFETCH_FRAMES`] ahead of playback through the bridge's
//! pipeline, so showing a frame is normally a cache hit. Where the bridge offers it (Linux), frames come
//! through a shared-memory ring instead of its stdout pipe ([`bridge::ShmRing`]): a 6K frame
//! is ~62 MB, and the pipe alone took ~14 ms of a 25 ms seek.
//!
//! Scrubbing a paused monitor ([`filmcraft_media::cancel::with_scrub`]) is served at the
//! [`SCRUB_TIER`] while the seeks come in a burst: a full-resolution 6K debayer takes ~50 ms, a
//! quarter one ~10 ms. Those frames are flagged provisional
//! ([`filmcraft_media::pending::mark_provisional`]); the monitor asks again without the hint, and
//! that request waits (cancellably) until the playhead has rested for [`SCRUB_REST`] before it
//! decodes the exact tier — so the refinement never holds the decoder while the drag goes on.
//!
//! VRAM: each bridge process carries a CUDA context (~0.4 GB) and the SDK's GPU pools (another
//! 0.3–2 GB while it decodes). A bridge with the `multi` capability therefore serves every clip
//! of its format from one process ([`shared`]): each source holds a clip handle on it, not a
//! process. Older bridges get a process per clip, at most [`max_processes`] per bridge binary
//! (the least recently used idle one is retired to make room). Sessions idle for
//! [`IDLE_SESSIONS_AFTER`] are retired by a janitor thread, whatever kind of request used them
//! last; the shared process exits with its last session.

pub mod bridge;
mod shared;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use filmcraft_color::ColorInfo;
use filmcraft_frame::VideoFrame;
use filmcraft_media::reader::FileReader;
use filmcraft_media::{FrameRequest, MediaError, MediaInfo, MediaKind, MediaSource, SharedSource, VideoStreamInfo};
use filmcraft_time::{FrameRate, Tick};

use crate::rawcam::bridge::{DecodeSession, Session};

/// Longest we wait for a clip probe (the SDKs enumerate GPU backends on first run). Also the
/// startup wait of a persistent decode session — the same one-off SDK initialisation.
const PROBE_TIMEOUT: Duration = Duration::from_secs(120);
/// Longest audio extraction (whole-clip PCM to disk).
const AUDIO_TIMEOUT: Duration = Duration::from_secs(600);
/// Longest a single `frame` request may take (a full-quality 8K CPU debayer takes seconds; a
/// bridge silent for longer has stalled and is killed).
const FRAME_TIMEOUT: Duration = Duration::from_secs(30);
/// Longest an interactive request without a cancellation context waits for the decode slot.
/// The slot is bounded by the spawn and frame timeouts; waiting longer means it is wedged.
const MAX_SLOT_WAIT: Duration = Duration::from_secs(150);
/// First pause before a bridge that keeps failing is started again (doubling, see
/// [`VideoState::note_failure`]).
const RESPAWN_BACKOFF: Duration = Duration::from_millis(500);
/// Decoded frames kept around a seek point (scrubbing back and forth).
const CACHE_FRAMES: usize = 24;
/// Bytes of decoded frames all sources cache together (a 6K full-tier frame is ~62–75 MB: one
/// clip's [`CACHE_FRAMES`] alone came to ~1.8 GB).
const CACHE_BUDGET: usize = 1 << 30;
/// Live decode sessions kept at once for a bridge without per-command scale (one per debayer
/// tier in use), most recently used last. A protocol 2 bridge needs only one.
const TIERS: usize = 2;
/// Frames the read-ahead keeps outstanding in the bridge (its pipeline decodes up to 4 at once).
/// An interactive request waits behind at most this many replies.
const READ_AHEAD_WINDOW: usize = 4;
/// How far a contended request may reach into the cache for a frame to show while it waits.
const NEAREST_FRAME_TOLERANCE: u64 = 12;
/// Frames the read-ahead thread decodes past the playhead (≈ 250 ms at 24 fps).
const PREFETCH_FRAMES: u64 = 6;
/// A quiet spell this long retires a source's sessions (its own bridge process, or its handle on
/// the shared one — which exits with the last handle).
const IDLE_SESSIONS_AFTER: Duration = Duration::from_secs(20);
/// How often the janitor looks for idle sessions.
const JANITOR_PERIOD: Duration = Duration::from_millis(500);
/// Persistent bridge processes per bridge binary unless `FILMCRAFT_RAWCAM_MAX_PROCESSES` says
/// otherwise (single-clip bridges need one per clip; a shared multi-clip bridge is one).
const DEFAULT_MAX_PROCESSES: usize = 4;
/// Paused-monitor seeks this close together are a scrub burst; the burst ends (the playhead
/// rests) after this long without a move.
const SCRUB_REST: Duration = Duration::from_millis(150);
/// The debayer tier a scrub burst is served at (quarter resolution: ~4× cheaper than full).
const SCRUB_TIER: u32 = 4;
/// Clips at least this wide scrub at eighth resolution where the bridge has it (an 8K quarter
/// frame is still 2K).
const EIGHTH_SCRUB_WIDTH: u32 = 7680;
/// Longest a refining request waits for a scrub burst to end (another view may keep scrubbing).
const MAX_REST_WAIT: Duration = Duration::from_millis(300);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Braw,
    R3d,
}

impl Format {
    fn from_path(path: &Path) -> Option<Format> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        match ext.as_str() {
            "braw" => Some(Format::Braw),
            "r3d" => Some(Format::R3d),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Format::Braw => "Blackmagic RAW",
            Format::R3d => "RED R3D",
        }
    }

    fn bridge_name(self) -> &'static str {
        match self {
            Format::Braw => "braw-bridge",
            Format::R3d => "r3d-bridge",
        }
    }

    fn bridge_env(self) -> &'static str {
        match self {
            Format::Braw => "FILMCRAFT_BRAW_BRIDGE",
            Format::R3d => "FILMCRAFT_R3D_BRIDGE",
        }
    }

    /// The bridge's debayer tier word for a resolution tier (1 = full resolution).
    fn tier_word(self, factor: u32) -> &'static str {
        match (self, factor) {
            (Format::Braw, 1) => "full",
            (Format::Braw, 2) => "half",
            (Format::R3d, 1) => "premium",
            (Format::R3d, 2) => "half",
            (_, 4) => "quarter",
            (_, 8) => "eighth",
            _ => "quarter",
        }
    }

    /// Whether the eighth tier is used: R3D always; BRAW from protocol 2 bridges on (`per_command`
    /// — older ones are not trusted with `--debayer eighth`).
    fn has_eighth(self, per_command: bool) -> bool {
        match self {
            Format::Braw => per_command,
            Format::R3d => true,
        }
    }

    /// Resolution factor for a scale request: full/half/quarter, down to eighth where
    /// [`Self::has_eighth`]. `draft` steps down one tier: playback may debayer faster than the
    /// paused, exact view.
    fn tier_factor(self, scale: f32, draft: bool, per_command: bool) -> u32 {
        let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        let (factor, smaller) = match self {
            Format::Braw if !self.has_eighth(per_command) => {
                if scale >= 0.7 {
                    (1, 2)
                } else if scale >= 0.35 {
                    (2, 4)
                } else {
                    (4, 4)
                }
            }
            Format::Braw | Format::R3d => {
                if scale >= 0.7 {
                    (1, 2)
                } else if scale >= 0.35 {
                    (2, 4)
                } else if scale >= 0.17 {
                    (4, 8)
                } else {
                    (8, 8)
                }
            }
        };
        if draft { smaller } else { factor }
    }

    /// `steps` tiers down from `tier` (a larger factor = cheaper), capped at the format's floor.
    fn stepped_down(self, tier: u32, steps: u32, per_command: bool) -> u32 {
        let floor = if self.has_eighth(per_command) { 8 } else { 4 };
        let factor = 1_u32.checked_shl(steps.min(4)).unwrap_or(floor);
        tier.saturating_mul(factor).min(floor)
    }
}

/// How many tiers shuttling at `speed` (multiples of realtime) decodes down: two steps once it
/// outruns what full tiers decode at, one step from double speed. Zero otherwise.
fn shuttle_steps(speed: f64) -> u32 {
    if speed >= 3.2 {
        2
    } else if speed >= 1.6 {
        1
    } else {
        0
    }
}

/// Whether the shared-memory frame transport may be offered (`FILMCRAFT_RAWCAM_SHM=0` keeps
/// every frame on the pipe — comparisons, or a sandbox without `/dev/shm`).
fn shm_wanted() -> bool {
    std::env::var_os("FILMCRAFT_RAWCAM_SHM").is_none_or(|v| v != "0")
}

/// Whether clips may share one multi-clip bridge process (`FILMCRAFT_RAWCAM_MULTI=0`: a process
/// per clip, as with bridges that predate it — comparisons).
fn multi_wanted() -> bool {
    std::env::var_os("FILMCRAFT_RAWCAM_MULTI").is_none_or(|v| v != "0")
}

/// The cap on persistent processes per bridge binary (`FILMCRAFT_RAWCAM_MAX_PROCESSES`, 1..=64).
fn max_processes() -> usize {
    std::env::var("FILMCRAFT_RAWCAM_MAX_PROCESSES").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(DEFAULT_MAX_PROCESSES, |n| n.clamp(1, 64))
}

/// Whether per-request diagnostics are on (`FILMCRAFT_RAWCAM_STATS=1`): one line per
/// interactive request on stderr, so a recorded scrub answers where its time went.
fn stats_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FILMCRAFT_RAWCAM_STATS").is_some_and(|v| !v.is_empty()))
}

/// How fast interactive requests move through the clip, so shuttling (2×, 4×, …) can decode
/// at reduced resolution the way every NLE's proxy playback does. A window of recent requests;
/// the speed is only believed once the window is long enough that short bursts (a scrub flick,
/// the test loops) do not trigger it.
#[derive(Default)]
struct Shuttle {
    samples: std::collections::VecDeque<(Instant, u64)>,
}

impl Shuttle {
    fn observe(&mut self, idx: u64) {
        let now = Instant::now();
        self.samples.push_back((now, idx));
        while self.samples.len() > 12 || self.samples.front().is_some_and(|&(t, _)| now.duration_since(t) > Duration::from_secs(2)) {
            if self.samples.pop_front().is_none() {
                break;
            }
        }
    }

    /// Playback speed in multiples of realtime (1.0 = the clip's own frame rate); a short or
    /// thin window reads as 1.0.
    fn factor(&self, rate: FrameRate) -> f64 {
        let (Some(&(oldest_t, oldest_i)), Some(&(newest_t, newest_i))) = (self.samples.front(), self.samples.back()) else {
            return 1.0;
        };
        let elapsed = newest_t.duration_since(oldest_t);
        if self.samples.len() < 4 || elapsed < Duration::from_millis(250) {
            return 1.0;
        }
        let fps = rate.as_f64().max(1.0);
        newest_i.abs_diff(oldest_i) as f64 / elapsed.as_secs_f64() / fps
    }

    /// Which way the requests move: -1 backwards (scrubbing left), 0 unclear, 1 forwards.
    /// Judged over the last ~120 ms only, so a direction change flips the read-ahead at once.
    /// The read-ahead decodes in this direction, so a dragging playhead meets cached frames
    /// instead of waiting behind read-aheads it will never reach.
    fn direction(&self) -> i64 {
        let Some(&(newest_t, newest_i)) = self.samples.back() else {
            return 0;
        };
        let mut reference = self.samples.front().map(|&(_, i)| i);
        for &(t, i) in self.samples.iter().rev() {
            if newest_t.duration_since(t) >= Duration::from_millis(120) {
                reference = Some(i);
                break;
            }
        }
        match newest_i.cmp(&reference.unwrap_or(newest_i)) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }
}

/// Paused-monitor seeks ([`filmcraft_media::cancel::scrub`]): whether they currently come in a
/// burst. A single seek (a click on the ruler, a frame step) is no burst and gets its exact frame;
/// from the second seek that follows within [`SCRUB_REST`], frames are served at [`SCRUB_TIER`].
#[derive(Default)]
struct Scrub {
    /// The newest seek: when, and to which frame.
    prev: Option<(Instant, u64)>,
    /// When the newest seek that moved the playhead within [`SCRUB_REST`] of the previous one
    /// arrived (the burst's pulse).
    moved: Option<Instant>,
}

impl Scrub {
    /// Record a seek to `idx` at `now`; whether it is part of a burst.
    fn observe(&mut self, now: Instant, idx: u64) -> bool {
        let moved = self.prev.is_some_and(|(t, i)| i != idx && now.saturating_duration_since(t) < SCRUB_REST);
        if moved {
            self.moved = Some(now);
        }
        self.prev = Some((now, idx));
        self.active(now)
    }

    /// The seek to `idx` was answered at `now`: the gap to the next seek counts from here (a
    /// seek that had to wait for the decoder must not end the burst it belongs to).
    fn served(&mut self, now: Instant, idx: u64, burst: bool) {
        if self.prev.is_some_and(|(_, i)| i == idx) {
            self.prev = Some((now, idx));
            if burst {
                self.moved = Some(now);
            }
        }
    }

    /// Whether a burst is going on at `now`.
    fn active(&self, now: Instant) -> bool {
        !self.rest_left(now).is_zero()
    }

    /// How long until the playhead counts as resting (zero: it does).
    fn rest_left(&self, now: Instant) -> Duration {
        self.moved.map_or(Duration::ZERO, |t| SCRUB_REST.saturating_sub(now.saturating_duration_since(t)))
    }
}

/// The frames the read-ahead wants after `last` in `direction` (negative: backwards), nearest
/// first, inside a clip of `frame_count` frames — each at most once (no repeats at the clip's
/// start or end).
fn ahead_indices(last: u64, direction: i64, frame_count: u64) -> Vec<u64> {
    (1..=PREFETCH_FRAMES)
        .map_while(|ahead| if direction < 0 { last.checked_sub(ahead) } else { last.checked_add(ahead) })
        .take_while(|&next| next < frame_count)
        .collect()
}

fn bridge_path(f: Format) -> Option<PathBuf> {
    bridge_path_from(std::env::var_os(f.bridge_env()), std::env::var_os("PATH"), f)
}

/// [`bridge_path`] over given environment values (tests).
fn bridge_path_from(env: Option<std::ffi::OsString>, paths: Option<std::ffi::OsString>, f: Format) -> Option<PathBuf> {
    if let Some(p) = env.filter(|p| !p.is_empty()) {
        return Some(PathBuf::from(p));
    }
    paths.and_then(|paths| std::env::split_paths(&paths).map(|dir| dir.join(f.bridge_name())).find(|candidate| candidate.is_file()))
}

/// `HH:MM:SS:FF` at `fps` to a frame count (non-drop; the format these cameras write).
fn timecode_to_frames(tc: &str, rate: FrameRate) -> Option<i64> {
    let parts: Vec<&str> = tc.split(':').collect();
    if parts.len() != 4 {
        return None;
    }
    let (h, m, s, f): (i64, i64, i64, i64) = match (parts[0].parse(), parts[1].parse(), parts[2].parse(), parts[3].parse()) {
        (Ok(h), Ok(m), Ok(s), Ok(f)) => (h, m, s, f),
        _ => return None,
    };
    if h > 23 || m > 59 || s > 59 || f >= 60 {
        return None;
    }
    let seconds = h * 3600 + m * 60 + s;
    let fps = (rate.num + rate.den / 2) / rate.den;
    Some(seconds * fps + f)
}

fn cache_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("FILMCRAFT_RAWCAM_CACHE").filter(|p| !p.is_empty()) {
        return PathBuf::from(p);
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache"));
    base.join("filmcraft").join("rawcam")
}

/// Stable per-file cache key: name, size and mtime (re-extract when the file changes).
fn audio_cache_path(f: Format, input: &Path, dir: &Path) -> PathBuf {
    let len = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    let mtime =
        std::fs::metadata(input).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
    let stem: String = input
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("clip")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .take(80)
        .collect();
    dir.join(format!("{}-{}-{}-{}.wav", f.bridge_name(), stem, len, mtime))
}

/// Extract (or reuse) the clip's audio as a WAV in `dir` (the caller's cache) and open it.
fn open_audio(f: Format, bridge: &Path, input: &Path, dir: &Path) -> Option<filmcraft_media::wav::WavSource> {
    let wav = audio_cache_path(f, input, dir);
    if !wav.is_file() {
        if let Some(parent) = wav.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = wav.with_extension("part.wav");
        match bridge::extract_audio(bridge, input, &tmp, AUDIO_TIMEOUT) {
            Ok(()) => {
                if std::fs::rename(&tmp, &wav).is_err() {
                    let _ = std::fs::remove_file(&tmp);
                }
            }
            Err(e) => {
                log::warn!("{}: no audio track ({e})", input.display());
                let _ = std::fs::remove_file(&tmp);
                return None;
            }
        }
    }
    let reader = FileReader::open(&wav).ok()?;
    filmcraft_media::wav::WavSource::open(wav.file_name().and_then(|n| n.to_str()).unwrap_or("audio"), Arc::new(reader)).ok()
}

/// Pool key of a session that takes the tier with every command (protocol 2): it serves all.
const ANY_TIER: u32 = 0;

/// One live bridge run: at a single debayer tier, or at [`ANY_TIER`].
struct SessionSlot {
    tier: u32,
    session: Session,
}

/// Idle decode sessions: one serving every tier ([`ANY_TIER`]), or for older bridges at most
/// [`TIERS`] (one per debayer tier in use — playback at full resolution, thumbnails at quarter —
/// so the two never restart each other), most recently used last. While a request decodes, it owns its session *outside* this lock with [`VideoState::busy`]
/// set: concurrent requests then serve from the frame cache instead of queueing up behind the
/// decode (a scrub burst must not restart the bridge once per queued jump).
struct VideoState {
    busy: bool,
    sessions: Vec<SessionSlot>,
    /// Consecutive bridge failures (start-up or decode), reset by a decoded frame.
    failures: u32,
    /// From the second consecutive failure on: no new bridge is started before this.
    retry_at: Option<Instant>,
}

impl VideoState {
    fn new() -> Self {
        VideoState { busy: false, sessions: Vec::new(), failures: 0, retry_at: None }
    }

    /// A pooled session serving `tier` and its pool key, if its bridge still runs (a dead one is
    /// dropped: it crashed while idle, and the request must not inherit that).
    fn take_session(&mut self, tier: u32) -> Option<(u32, Session)> {
        let at = self.sessions.iter().position(|s| s.tier == ANY_TIER || s.tier == tier)?;
        let SessionSlot { tier: key, mut session } = self.sessions.remove(at);
        session.is_alive().then_some((key, session))
    }

    /// How long until a new bridge may be started (`None`: now).
    fn backoff_left(&self, now: Instant) -> Option<Duration> {
        self.retry_at.and_then(|t| t.checked_duration_since(now)).filter(|d| !d.is_zero())
    }

    /// A bridge failed. One failure is retried at once (a crash mid-session recovers on the next
    /// frame); from the second in a row, starting a new bridge waits 0.5 s, doubling up to 16 s —
    /// a broken bridge must not be respawned for every frame a monitor asks for.
    fn note_failure(&mut self, now: Instant) {
        self.failures = self.failures.saturating_add(1);
        if self.failures >= 2 {
            let doubling = (self.failures - 2).min(5);
            self.retry_at = Some(now + RESPAWN_BACKOFF * (1_u32 << doubling));
        }
    }

    fn note_success(&mut self) {
        self.failures = 0;
        self.retry_at = None;
    }

    /// Return a working session to the pool (most recently used last, at most [`TIERS`]).
    fn put_back(&mut self, tier: u32, session: Session) {
        self.sessions.push(SessionSlot { tier, session });
        if self.sessions.len() > TIERS {
            self.sessions.remove(0);
        }
    }
}

/// The decode slot, held ([`VideoState::busy`] is set): [`Lease::succeed`] / [`Lease::fail`]
/// release it with the outcome, and dropping it releases it too — a decode that panics must not
/// leave every later request waiting for a slot that never frees.
struct Lease<'a> {
    video: &'a Mutex<VideoState>,
    held: bool,
}

impl<'a> Lease<'a> {
    /// Take the slot; the caller has just set `busy` under the lock.
    fn new(video: &'a Mutex<VideoState>) -> Self {
        Lease { video, held: true }
    }

    /// `ok`: `Some(true)` the bridge works, `Some(false)` it failed, `None` no news (a per-frame
    /// error on a session that is still fine).
    fn release(mut self, outcome: Option<(u32, Session)>, ok: Option<bool>) {
        let mut state = self.video.lock().unwrap_or_else(|e| e.into_inner());
        state.busy = false;
        match ok {
            Some(true) => state.note_success(),
            Some(false) => state.note_failure(Instant::now()),
            None => {}
        }
        if let Some((tier, session)) = outcome {
            state.put_back(tier, session);
        }
        self.held = false;
    }

    /// The session decoded: back into the pool.
    fn succeed(self, key: u32, session: Session) {
        self.release(Some((key, session)), Some(true));
    }

    /// The session is in sync but the frame failed (a per-frame error): back into the pool,
    /// without counting towards the respawn back-off — no bridge has to be restarted.
    fn keep(self, key: u32, session: Session) {
        self.release(Some((key, session)), None);
    }

    /// The bridge failed (the session, if any, was dropped by the caller — outside the lock).
    fn fail(self) {
        self.release(None, Some(false));
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if self.held {
            self.video.lock().unwrap_or_else(|e| e.into_inner()).busy = false;
        }
    }
}

/// The frame index `time` falls on, inside a clip of `frame_count` frames (the last frame past
/// its end; any count, even beyond `i64`, is safe).
fn frame_index(rate: FrameRate, time: Tick, frame_count: u64) -> u64 {
    let last = i64::try_from(frame_count.saturating_sub(1)).unwrap_or(i64::MAX);
    u64::try_from(rate.frame_at(time.max(Tick::ZERO)).clamp(0, last)).unwrap_or(0)
}

pub struct RawCamSource {
    info: MediaInfo,
    path: PathBuf,
    bridge: PathBuf,
    format: Format,
    frame_count: u64,
    audio: Option<filmcraft_media::wav::WavSource>,
    /// Idle sessions + the busy flag. Never held while a bridge is read from: only around the
    /// slot bookkeeping.
    video: Mutex<VideoState>,
    /// Frames by (index, debayer tier): a thumbnail and a full frame of the same index coexist.
    /// Behind its own lock so a decoding request never blocks a cache lookup.
    cache: Mutex<BTreeMap<(u64, u32), Arc<VideoFrame>>>,
    /// The read-ahead thread's handle to us (set right after construction).
    self_ref: OnceLock<Weak<Self>>,
    /// Frame index and tier of the newest interactive request (steers the read-ahead).
    last: AtomicU64,
    last_tier: AtomicU32,
    /// The width interactive requests want their frames fitted to (steers the read-ahead).
    /// Bumped by every interactive cache miss; the read-ahead yields when it changes mid-run.
    demand: AtomicU64,
    /// Set on drop: the read-ahead thread wakes and leaves.
    stop: AtomicBool,
    prefetch_started: AtomicBool,
    /// Last decode of any kind (idle sessions are retired after [`IDLE_SESSIONS_AFTER`]).
    activity: Mutex<Instant>,
    /// Recent interactive requests, for shuttle-speed-adaptive decode tiers.
    shuttle: Mutex<Shuttle>,
    /// Recent paused-monitor seeks, for the scrub tier.
    scrub: Mutex<Scrub>,
    /// Interactive requests waiting for the decode slot: the read-ahead starts no batch while
    /// any wait, and sends no further command of a running one.
    waiting: AtomicU32,
    /// The bridge takes the debayer tier with every command (protocol 2, from the probe): one
    /// session serves every tier. Cleared if a session turns out not to (the binary changed).
    per_command: AtomicBool,
    /// Offer the bridge the shared-memory transport (its probe says it has it; off with
    /// `FILMCRAFT_RAWCAM_SHM=0`). Cleared when a bridge refuses to start with it.
    shm: AtomicBool,
    /// Decode on the format's shared multi-clip bridge (its probe says it has it; off with
    /// `FILMCRAFT_RAWCAM_MULTI=0`). Cleared when the shared bridge does not start but a process
    /// of its own does.
    multi: AtomicBool,
}

/// Counts an interactive request in [`RawCamSource::waiting`] while it lives.
struct Waiting<'a>(&'a AtomicU32);

impl<'a> Waiting<'a> {
    fn new(count: &'a AtomicU32) -> Self {
        count.fetch_add(1, Ordering::Relaxed);
        Waiting(count)
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl RawCamSource {
    fn open(path: &Path, format: Format, bridge: PathBuf, cache: Option<PathBuf>) -> Result<Arc<Self>, String> {
        let meta = bridge::probe(&bridge, path, PROBE_TIMEOUT).map_err(|e| format!("{}: {e}", path.display()))?;
        let rate = FrameRate::new(i64::from(meta.fps_num), i64::from(meta.fps_den.max(1)));
        let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        // A count past `i64` (a hostile bridge) is clamped: every index then converts safely.
        let frame_count = meta.frame_count.min(i64::MAX.unsigned_abs());
        let duration = Tick::from_rational(i64::try_from(frame_count).unwrap_or(i64::MAX), rate.den, rate.num);
        let bitrate = len.checked_mul(8).and_then(|b| b.checked_mul(u64::from(meta.fps_num))).map(|b| b / u64::from(meta.fps_den.max(1)));
        let cache = cache.unwrap_or_else(cache_dir);
        let audio = open_audio(format, &bridge, path, &cache);
        let audio_streams = audio.as_ref().map(|w| w.info().audio_streams.clone()).unwrap_or_default();
        let info = MediaInfo {
            name: path.file_name().and_then(|n| n.to_str()).unwrap_or("camera raw").to_string(),
            kind: MediaKind::Movie,
            duration,
            video: Some(VideoStreamInfo {
                width: meta.width,
                height: meta.height,
                frame_rate: rate,
                par: (1, 1),
                codec: format.label().into(),
                pixel_format: "Debayered RGB 8-bit".into(),
                color: ColorInfo::SRGB_FULL,
                has_alpha: false,
                bitrate,
                hdr: None,
            }),
            audio_streams,
            container: format.label().into(),
            start_timecode: timecode_to_frames(&meta.timecode, rate),
            file_size: Some(len),
        };
        let src = Arc::new(Self {
            info,
            path: path.to_path_buf(),
            bridge,
            format,
            frame_count,
            audio,
            video: Mutex::new(VideoState::new()),
            cache: Mutex::new(BTreeMap::new()),
            self_ref: OnceLock::new(),
            last: AtomicU64::new(0),
            last_tier: AtomicU32::new(1),
            demand: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            prefetch_started: AtomicBool::new(false),
            activity: Mutex::new(Instant::now()),
            shuttle: Mutex::new(Shuttle::default()),
            scrub: Mutex::new(Scrub::default()),
            waiting: AtomicU32::new(0),
            per_command: AtomicBool::new(meta.caps.scale),
            shm: AtomicBool::new(meta.caps.shm && shm_wanted()),
            multi: AtomicBool::new(meta.caps.multi && meta.caps.scale && multi_wanted()),
        });
        let _ = src.self_ref.set(Arc::downgrade(&src));
        register(&src);
        Ok(src)
    }

    /// Start the read-ahead thread once (after the first interactive decode).
    fn ensure_prefetch(&self) {
        if self.stop.load(Ordering::Relaxed) {
            return;
        }
        if !self.prefetch_started.swap(true, Ordering::Relaxed)
            && let Some(weak) = self.self_ref.get().cloned()
        {
            let started = bridge::spawn_guarded("filmcraft-rawcam-readahead", move || {
                let again = weak.clone();
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Self::read_ahead(again))).is_err() {
                    log::error!("camera-RAW read-ahead panicked; it restarts with the next request");
                    if let Some(this) = weak.upgrade() {
                        this.prefetch_started.store(false, Ordering::Relaxed);
                    }
                }
            });
            if !started {
                self.prefetch_started.store(false, Ordering::Relaxed);
            }
        }
    }

    /// Whether the bridge takes the tier per command (one session for every tier).
    fn per_command(&self) -> bool {
        self.per_command.load(Ordering::Relaxed)
    }

    /// `pooled`, or a fresh session serving `tier`, with its pool key (called without the lock
    /// held: starting a bridge can take seconds). A clip handle on the format's shared bridge
    /// where it serves several clips (one session for every tier), else a process of its own.
    fn open_session(&self, pooled: Option<(u32, Session)>, tier: u32) -> Result<(u32, Session), MediaError> {
        if let Some(pooled) = pooled {
            return Ok(pooled);
        }
        if self.multi.load(Ordering::Relaxed) {
            let manager = shared::manager(&self.bridge);
            if !manager.isolated(&self.path) {
                match manager.open(&self.path, self.frame_bytes(), self.shm.load(Ordering::Relaxed), PROBE_TIMEOUT) {
                    Ok(session) => return Ok((ANY_TIER, Session::Shared(session))),
                    Err(shared::SharedError::Spawn(e)) => {
                        // Replaced by a binary without the mode since the probe, or it cannot
                        // start that way: a process of its own, from now on if that works.
                        let own = self.open_own(tier);
                        if own.is_ok() {
                            log::warn!("{}: shared {} did not start ({e}); one process per clip", self.path.display(), self.format.bridge_name());
                            self.multi.store(false, Ordering::Relaxed);
                        }
                        return own;
                    }
                    Err(shared::SharedError::Clip(e) | shared::SharedError::Bridge(e)) => {
                        return Err(MediaError::Decode(format!("{}: {}: {e}", self.path.display(), self.format.bridge_name())));
                    }
                }
            }
        }
        self.open_own(tier)
    }

    /// The bytes of this clip's full-resolution RGBA frame (the shared bridge's slot size).
    fn frame_bytes(&self) -> u64 {
        self.info.video.as_ref().map_or(0, |v| u64::from(v.width).saturating_mul(u64::from(v.height)).saturating_mul(4))
    }

    /// A bridge process of this clip's own (single-clip `--persistent`), within the process cap.
    /// A protocol 2 bridge starts at full resolution and serves every tier ([`ANY_TIER`]); a
    /// session that turns out not to take per-command tiers (the binary was replaced since the
    /// probe) is restarted at `tier`.
    fn open_own(&self, tier: u32) -> Result<(u32, Session), MediaError> {
        self.make_room()?;
        let start = |word: &str| {
            let shm = self.shm.load(Ordering::Relaxed);
            let mut started = DecodeSession::spawn(&self.bridge, &self.path, word, shm, PROBE_TIMEOUT);
            if shm && started.is_err() {
                // A bridge that will not start with the shared-memory offer (replaced by an
                // older binary since the probe, or the ring rejected): the pipe, from now on.
                let plain = DecodeSession::spawn(&self.bridge, &self.path, word, false, PROBE_TIMEOUT);
                if plain.is_ok() {
                    log::warn!("{}: {} does not start with shared memory; using its pipe", self.path.display(), self.format.bridge_name());
                    self.shm.store(false, Ordering::Relaxed);
                    started = plain;
                }
            }
            started.map(Session::Own).map_err(|e| MediaError::Decode(format!("{}: cannot start {}: {e}", self.path.display(), self.format.bridge_name())))
        };
        if self.per_command() {
            let session = start(self.format.tier_word(1))?;
            if session.caps().scale {
                return Ok((ANY_TIER, session));
            }
            log::warn!("{}: {} no longer takes per-command tiers; one session per tier", self.path.display(), self.format.bridge_name());
            self.per_command.store(false, Ordering::Relaxed);
            if tier == 1 {
                return Ok((1, session));
            }
        }
        Ok((tier, start(self.format.tier_word(tier))?))
    }

    /// Below the process cap for this bridge binary, retiring the least recently used idle
    /// process of another clip if need be; an error when every one is busy.
    fn make_room(&self) -> Result<(), MediaError> {
        let cap = max_processes();
        for _ in 0..=cap {
            if bridge::live_processes(&self.bridge) < cap {
                return Ok(());
            }
            if !retire_lru_idle(&self.bridge, self) {
                break;
            }
        }
        if bridge::live_processes(&self.bridge) < cap {
            return Ok(());
        }
        Err(MediaError::Decode(format!(
            "{}: {cap} {} processes are busy already (FILMCRAFT_RAWCAM_MAX_PROCESSES)",
            self.path.display(),
            self.format.bridge_name()
        )))
    }

    /// Retire this source's idle sessions (their processes, or handles) when nothing has been
    /// decoded for `after`; whether there was anything to retire.
    fn retire_idle(&self, after: Duration, own_only: bool) -> bool {
        let idle_for = self.activity.lock().unwrap_or_else(|e| e.into_inner()).elapsed();
        if idle_for < after {
            return false;
        }
        let retired = {
            let mut state = self.video.lock().unwrap_or_else(|e| e.into_inner());
            if state.busy || (own_only && !state.sessions.iter().any(|s| !s.session.is_shared())) {
                return false;
            }
            std::mem::take(&mut state.sessions)
        };
        let any = !retired.is_empty();
        drop(retired); // kills and reaps, outside the lock
        any
    }

    /// The per-command tier word for a session under pool key `key` (none for a session that
    /// runs at one tier).
    fn scale_word(&self, key: u32, tier: u32) -> Option<&'static str> {
        (key == ANY_TIER).then(|| self.format.tier_word(tier))
    }

    /// The tier a scrub burst is served at: quarter, eighth for 8K and wider where available.
    fn scrub_tier(&self) -> u32 {
        let wide = self.info.video.as_ref().is_some_and(|v| v.width >= EIGHTH_SCRUB_WIDTH);
        if wide && self.format.has_eighth(self.per_command()) { 8 } else { SCRUB_TIER }
    }

    /// The read-ahead loop: keep [`PREFETCH_FRAMES`] frames past the newest interactive request
    /// decoded and cached, so playback reads the cache instead of waiting on a decode. It yields
    /// (yielding the slot between batches) whenever an interactive request arrives, and retires
    /// the bridge processes when nothing has been decoded for a while.
    fn read_ahead(weak: Weak<Self>) {
        loop {
            let Some(this) = weak.upgrade() else { break };
            if this.stop.load(Ordering::Relaxed) {
                break;
            }
            let last = this.last.load(Ordering::Relaxed);
            let tier = this.last_tier.load(Ordering::Relaxed);
            let start_demand = this.demand.load(Ordering::Relaxed);
            // The direction the interactive requests move: read ahead INTO the movement, so a
            // dragging playhead (forwards or backwards) meets cached frames instead of waiting
            // behind read-aheads it will never reach.
            let direction = this.shuttle.lock().unwrap_or_else(|e| e.into_inner()).direction();
            // A scrub burst owns the decoder: each of its seeks must find the bridge idle, and
            // frames around a moving playhead are wasted work.
            let scrubbing = this.scrub.lock().unwrap_or_else(|e| e.into_inner()).active(Instant::now());
            // No steering yet (tier 0 = no interactive request ever): nothing to read ahead of.
            let decoded = tier != 0 && !scrubbing && {
                // What is missing ahead of the playhead, nearest first: one batch fills a
                // pipelined bridge (reads overlap the running decode) and caches the frames as
                // they stream back, in order — a waiting request picks each up as it lands.
                let cache = this.cache.lock().unwrap_or_else(|e| e.into_inner());
                let mut missing = ahead_indices(last, direction, this.frame_count);
                missing.retain(|next| !cache.contains_key(&(*next, tier)));
                drop(cache);
                !missing.is_empty() && this.demand.load(Ordering::Relaxed) == start_demand && this.prefetch_batch(&missing, tier).is_ok()
            };
            if !decoded {
                drop(this);
                std::thread::sleep(Duration::from_millis(20));
            } else {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// Decode `indices` (in order) through one hold of the decode slot, caching each frame as
    /// it arrives. Never waits for the slot: if it is taken, the batch is simply skipped (the
    /// interactive request holding it is doing more urgent work).
    fn prefetch_batch(&self, indices: &[u64], tier: u32) -> Result<(), MediaError> {
        let rate = self.info.video.as_ref().map_or(FrameRate::FPS_25, |v| v.frame_rate);
        let mut state = self.video.lock().unwrap_or_else(|e| e.into_inner());
        if state.busy || indices.is_empty() || self.waiting.load(Ordering::Relaxed) > 0 {
            return Ok(());
        }
        let pooled = state.take_session(tier);
        if pooled.is_none() && state.backoff_left(Instant::now()).is_some() {
            return Ok(()); // a failing bridge is not restarted for read-ahead
        }
        state.busy = true;
        let lease = Lease::new(&self.video);
        drop(state);
        let (key, mut session) = match self.open_session(pooled, tier) {
            Ok(s) => s,
            Err(e) => {
                lease.fail();
                return Err(e);
            }
        };

        // A sliding window over the bridge's pipeline: it stays full (each reply read sends the
        // next command), and an interactive request arriving meanwhile waits behind at most
        // [`READ_AHEAD_WINDOW`] replies — no further command is sent once it waits.
        let start_demand = self.demand.load(Ordering::Relaxed);
        let outcome = session.request_batch_steered(
            indices,
            self.scale_word(key, tier),
            READ_AHEAD_WINDOW,
            || self.demand.load(Ordering::Relaxed) == start_demand && self.waiting.load(Ordering::Relaxed) == 0,
            FRAME_TIMEOUT,
        );
        *self.activity.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
        if outcome.is_ok() {
            lease.succeed(key, session);
        } else {
            drop(session); // kills and reaps the bridge, outside the lock
            lease.fail();
        }

        let replies = outcome.map_err(|e| {
            log::warn!("{}: read-ahead batch failed ({e}) — restarting the bridge", self.path.display());
            MediaError::Decode(format!("{}: {e}", self.path.display()))
        })?;
        for (idx, reply) in indices.iter().zip(replies) {
            match reply {
                Ok(reply) => self.cache_frame((*idx, tier), Arc::new(Self::into_frame(reply, *idx, rate))),
                // One bad frame: the interactive request for it reports the error.
                Err(e) => log::debug!("{}: read-ahead of frame {idx} failed: {e}", self.path.display()),
            }
        }
        Ok(())
    }

    /// A bridge reply as a compositor frame. The persistent bridges send RGBA/BGRA straight
    /// through — only the layouts needing work are converted (a legacy `rgb24` bridge, or
    /// R3D's channel order).
    fn into_frame(reply: bridge::FrameBytes, idx: u64, rate: FrameRate) -> VideoFrame {
        let pts = || Tick::from_rational(i64::try_from(idx).unwrap_or(i64::MAX), rate.den, rate.num);
        match reply.format {
            bridge::FrameFormat::Rgba8 => VideoFrame::rgba8(reply.width, reply.height, reply.data).with_pts(pts()),
            bridge::FrameFormat::Bgra8 => {
                let mut data = reply.data;
                for px in data.as_chunks_mut::<4>().0 {
                    px.swap(0, 2);
                }
                VideoFrame::rgba8(reply.width, reply.height, data).with_pts(pts())
            }
            bridge::FrameFormat::Rgb24 => {
                let mut rgba = Vec::with_capacity(reply.data.len() / 3 * 4);
                for px in reply.data.as_chunks::<3>().0 {
                    rgba.extend_from_slice(px);
                    rgba.push(255);
                }
                VideoFrame::rgba8(reply.width, reply.height, rgba).with_pts(pts())
            }
        }
    }

    /// The cached frame closest to `idx` (any tier) within [`NEAREST_FRAME_TOLERANCE`], if any.
    fn nearest_cached(&self, idx: u64) -> Option<Arc<VideoFrame>> {
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|((k, _), _)| k.abs_diff(idx) <= NEAREST_FRAME_TOLERANCE)
            .min_by_key(|((k, _), _)| k.abs_diff(idx))
            .map(|(_, f)| f.clone())
    }

    /// Wait (cancellably, at most [`MAX_REST_WAIT`]) until a scrub burst has ended: a request that
    /// refines a scrubbed frame must not hold the decoder while the playhead still moves.
    fn wait_for_rest(&self) -> Result<(), MediaError> {
        let started = Instant::now();
        loop {
            let left = self.scrub.lock().unwrap_or_else(|e| e.into_inner()).rest_left(Instant::now());
            if left.is_zero() || started.elapsed() >= MAX_REST_WAIT {
                return Ok(());
            }
            if filmcraft_media::cancel::cancelled() {
                return Err(MediaError::Cancelled);
            }
            std::thread::sleep(left.min(Duration::from_millis(4)));
        }
    }

    /// Store a decoded frame in the cache, trimmed to a window around the current position and
    /// to the [`CACHE_BUDGET`] all sources share.
    fn cache_frame(&self, key: (u64, u32), frame: Arc<VideoFrame>) {
        self.cache_frame_within(key, frame, &live_sources(), CACHE_BUDGET);
    }

    /// [`Self::cache_frame`] against the given sources and budget: past the budget, frames go
    /// from the source idle longest first, farthest from its playhead first; the frame just
    /// stored stays.
    fn cache_frame_within(&self, key: (u64, u32), frame: Arc<VideoFrame>, sources: &[Arc<Self>], budget: usize) {
        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.insert(key, frame);
            if cache.len() > CACHE_FRAMES {
                let farthest = cache.keys().copied().max_by_key(|k| k.0.abs_diff(key.0)).filter(|k| *k != key).unwrap_or(key);
                cache.remove(&farthest);
            }
        }
        // One cache lock at a time (never nested), so sources trimming each other cannot deadlock.
        let mut total: usize =
            sources.iter().filter(|s| !std::ptr::eq(Arc::as_ptr(s), self)).map(|s| s.cached_bytes()).fold(self.cached_bytes(), usize::saturating_add);
        if total <= budget {
            return;
        }
        let mut victims: Vec<(Instant, &Self)> =
            sources.iter().map(|s| (*s.activity.lock().unwrap_or_else(|e| e.into_inner()), &**s)).filter(|(_, s)| !std::ptr::eq(*s, self)).collect();
        victims.sort_by_key(|v| v.0);
        victims.push((Instant::now(), self)); // our own frames last
        for (_, victim) in victims {
            while total > budget {
                let Some(freed) = victim.evict_farthest(key, self) else { break };
                total = total.saturating_sub(freed);
            }
        }
    }

    /// Drop this source's cached frame farthest from its playhead (never `keep` when `owner` is
    /// this source); the bytes freed, or `None` when there is nothing to drop.
    fn evict_farthest(&self, keep: (u64, u32), owner: &Self) -> Option<usize> {
        let at = self.last.load(Ordering::Relaxed);
        let mine = std::ptr::eq(self, owner);
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let farthest = cache.keys().copied().filter(|k| !mine || *k != keep).max_by_key(|k| k.0.abs_diff(at))?;
        cache.remove(&farthest).map(|f| f.byte_size())
    }

    /// Bytes of the frames this source caches.
    fn cached_bytes(&self) -> usize {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).values().map(|f| f.byte_size()).fold(0, usize::saturating_add)
    }

    /// Frame `idx` as an sRGB RGBA frame, decoded through the tier's persistent bridge session:
    /// one `frame <index>` round trip, no SDK re-initialisation on seeks. While one request
    /// decodes, concurrent requests show the nearest cached frame instead of queueing behind it;
    /// a request only waits when the cache holds nothing to show at all, and gives up when
    /// cancelled. Interactive requests (`steer`) steer the read-ahead and wait for their exact
    /// frame (accepting a nearby one while the display waits would rock playback a frame back and
    /// forth); background work (thumbnails, read-ahead) yields to them immediately. Sustained
    /// fast *playback* (shuttling) decodes at reduced tiers — proxy-quality while fast-forwarding,
    /// exactly like every NLE; only while playing (`catch_up` is set then, and the request is no
    /// paused seek). A paused monitor's scrub burst is served at [`SCRUB_TIER`], flagged
    /// provisional so the monitor asks again for the exact frame once the playhead rests.
    fn serve(&self, idx: u64, tier: u32, want: f32, steer: bool) -> Result<Arc<VideoFrame>, MediaError> {
        let rate = self.info.video.as_ref().map_or(FrameRate::FPS_25, |v| v.frame_rate);
        let seek = steer && filmcraft_media::cancel::scrub();
        let (tier, serve_tier, burst) = if steer {
            let mut shuttle = self.shuttle.lock().unwrap_or_else(|e| e.into_inner());
            shuttle.observe(idx);
            let playing = filmcraft_media::cancel::catch_up().is_some() && !seek;
            let steps = if playing { shuttle_steps(shuttle.factor(rate)) } else { 0 };
            drop(shuttle);
            let tier = self.format.stepped_down(tier, steps, self.per_command());
            let burst = seek && self.scrub.lock().unwrap_or_else(|e| e.into_inner()).observe(Instant::now(), idx);
            (tier, if burst { tier.max(self.scrub_tier()) } else { tier }, burst)
        } else {
            (tier, tier, false)
        };
        let answered = || {
            if seek {
                self.scrub.lock().unwrap_or_else(|e| e.into_inner()).served(Instant::now(), idx, burst);
            }
        };
        // The frame served is a stand-in for the exact one (a scrub burst's cheaper tier).
        let provisional = serve_tier != tier;
        // Frames are cached by (index, tier): the tier ladder is coarse (a quarter-tier 6K
        // frame is 1512 px), which downstream conversions handle proportionally.
        let stats = steer && stats_on();
        let started = Instant::now();
        let mut waited = Duration::ZERO;
        // An interactive request that is no seek (refining a scrubbed frame, playback starting)
        // waits for a running scrub burst to end before it decodes — once.
        let mut rest_checked = !steer || seek;
        // Set while this interactive request waits for the decode slot (the read-ahead yields).
        let mut waiting = None;

        loop {
            let hit = {
                let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                // The exact frame beats a stand-in, even during a burst.
                match cache.get(&(idx, tier)) {
                    Some(f) => Some((f.clone(), tier)),
                    None if provisional => cache.get(&(idx, serve_tier)).map(|f| (f.clone(), serve_tier)),
                    None => None,
                }
            };
            if let Some((hit, hit_tier)) = hit {
                if steer {
                    self.last.store(idx, Ordering::Relaxed);
                    self.last_tier.store(hit_tier, Ordering::Relaxed);
                    self.ensure_prefetch();
                }
                if hit_tier != tier {
                    filmcraft_media::pending::mark_provisional();
                }
                if stats {
                    eprintln!("rawcam req idx={idx} tier={hit_tier} want={want:.3} seek={seek} hit total={:?}", started.elapsed());
                }
                answered();
                return Ok(hit);
            }
            if !rest_checked {
                rest_checked = true;
                let before = Instant::now();
                self.wait_for_rest()?;
                waited += before.elapsed();
                // the cache may hold it now (read-ahead, a concurrent request)
                continue;
            }
            if steer && waiting.is_none() {
                waiting = Some(Waiting::new(&self.waiting));
            }
            let mut state = self.video.lock().unwrap_or_else(|e| e.into_inner());
            if state.busy {
                // Someone is decoding. Background work takes a nearby cached frame and leaves;
                // an interactive request waits for the slot — but never past its own
                // cancellation, so scrubbing past stale frames costs nothing.
                drop(state);
                if !steer {
                    return self.nearest_cached(idx).ok_or(MediaError::Cancelled);
                }
                if filmcraft_media::cancel::cancelled() {
                    if stats {
                        eprintln!("rawcam req idx={idx} tier={tier} want={want:.3} cancelled-after-wait={waited:?}");
                    }
                    return Err(MediaError::Cancelled);
                }
                if waited >= MAX_SLOT_WAIT {
                    return Err(MediaError::Decode(format!("{}: the {} decoder is not answering", self.path.display(), self.format.bridge_name())));
                }
                let nap = Duration::from_millis(2);
                std::thread::sleep(nap);
                waited += nap;
                continue;
            }
            if filmcraft_media::cancel::cancelled() {
                return Err(MediaError::Cancelled);
            }
            let pooled = state.take_session(serve_tier);
            if pooled.is_none()
                && let Some(left) = state.backoff_left(Instant::now())
            {
                return Err(MediaError::Decode(format!(
                    "{}: {} keeps failing; trying again in {:.1} s",
                    self.path.display(),
                    self.format.bridge_name(),
                    left.as_secs_f64()
                )));
            }
            if steer {
                self.last.store(idx, Ordering::Relaxed);
                self.last_tier.store(serve_tier, Ordering::Relaxed);
                self.demand.fetch_add(1, Ordering::Relaxed);
            }
            state.busy = true;
            let lease = Lease::new(&self.video);
            drop(state);
            drop(waiting.take());

            // The session is owned locally while decoding (and started, if need be): the sessions
            // lock is free, so the bookkeeping above stays responsive. A request that breaks the
            // session desynchronises the byte stream — it is then dropped (killing its bridge),
            // and the next request starts a fresh one. A per-frame error keeps it.
            let (key, mut session) = match self.open_session(pooled, serve_tier) {
                Ok(s) => s,
                Err(e) => {
                    lease.fail();
                    return Err(e);
                }
            };

            let decode_started = Instant::now();
            let outcome = session.request_frame(idx, self.scale_word(key, serve_tier), FRAME_TIMEOUT);
            let decoded_in = decode_started.elapsed();
            let (decode_wait, transport) = session.last_timing();
            *self.activity.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
            if session.is_shared() && matches!(outcome, Ok(Ok(_))) {
                shared::manager(&self.bridge).note_success();
            }
            match &outcome {
                Ok(Ok(_)) => lease.succeed(key, session),
                Ok(Err(_)) => lease.keep(key, session),
                Err(_) => {
                    drop(session); // kills and reaps the bridge, outside the lock
                    lease.fail();
                }
            }

            if stats {
                eprintln!(
                    "rawcam req idx={idx} tier={serve_tier} want={want:.3} seek={seek} decoded wait={waited:?} decode={decoded_in:?} (bridge {decode_wait:?}, pixels {transport:?}) total={:?}",
                    started.elapsed()
                );
            }

            let reply = outcome
                .map_err(|e| {
                    log::warn!("{}: frame {idx} failed ({e}) — restarting the bridge", self.path.display());
                    MediaError::Decode(format!("{}: {e}", self.path.display()))
                })?
                .map_err(|e| MediaError::Decode(format!("{}: frame {idx}: {e}", self.path.display())))?;
            let frame = Arc::new(Self::into_frame(reply, idx, rate));
            self.cache_frame((idx, serve_tier), frame.clone());
            if provisional {
                filmcraft_media::pending::mark_provisional();
            }
            if steer {
                self.ensure_prefetch();
            }
            answered();
            return Ok(frame);
        }
    }
}

/// Every open source (weakly): the janitor's and the process cap's view of them.
fn registry() -> &'static Mutex<Vec<Weak<RawCamSource>>> {
    static SOURCES: OnceLock<Mutex<Vec<Weak<RawCamSource>>>> = OnceLock::new();
    SOURCES.get_or_init(|| Mutex::new(Vec::new()))
}

/// The live sources (dead entries are dropped on the way).
fn live_sources() -> Vec<Arc<RawCamSource>> {
    let mut all = registry().lock().unwrap_or_else(|e| e.into_inner());
    all.retain(|w| w.strong_count() > 0);
    all.iter().filter_map(Weak::upgrade).collect()
}

/// Track `src` and make sure the janitor runs.
fn register(src: &Arc<RawCamSource>) {
    static JANITOR: AtomicBool = AtomicBool::new(false);
    registry().lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(src));
    if !JANITOR.swap(true, Ordering::Relaxed) {
        let started = bridge::spawn_guarded("filmcraft-rawcam-janitor", || {
            let run = std::panic::catch_unwind(|| {
                loop {
                    std::thread::sleep(JANITOR_PERIOD);
                    for src in live_sources() {
                        src.retire_idle(IDLE_SESSIONS_AFTER, false);
                    }
                }
            });
            if run.is_err() {
                log::error!("camera-RAW janitor panicked; it restarts with the next clip opened");
            }
            JANITOR.store(false, Ordering::Relaxed);
        });
        if !started {
            JANITOR.store(false, Ordering::Relaxed);
        }
    }
}

/// Retire the idle process of the least recently used other source on `bridge`; whether one was
/// found.
fn retire_lru_idle(bridge: &Path, except: &RawCamSource) -> bool {
    let mut candidates: Vec<(Instant, Arc<RawCamSource>)> = live_sources()
        .into_iter()
        .filter(|s| !std::ptr::eq(Arc::as_ptr(s), except) && s.bridge == bridge)
        .map(|s| {
            let at = *s.activity.lock().unwrap_or_else(|e| e.into_inner());
            (at, s)
        })
        .collect();
    candidates.sort_by_key(|c| c.0);
    candidates.into_iter().any(|(_, s)| s.retire_idle(Duration::ZERO, true))
}

impl Drop for RawCamSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

impl MediaSource for RawCamSource {
    fn info(&self) -> &MediaInfo {
        &self.info
    }

    fn video_frame(&self, req: FrameRequest) -> Result<Arc<VideoFrame>, MediaError> {
        let Some(v) = self.info.video.as_ref() else { return Err(MediaError::NoStream("video")) };
        let idx = frame_index(v.frame_rate, req.time, self.frame_count);
        let tier = self.format.tier_factor(req.scale, filmcraft_media::cancel::draft(), self.per_command());
        self.serve(idx, tier, req.scale, !filmcraft_media::cancel::background())
    }

    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> Result<filmcraft_frame::AudioBuffer, MediaError> {
        match &self.audio {
            Some(w) => w.audio(start, frames, sample_rate),
            None => Err(MediaError::NoStream("audio")),
        }
    }
}

/// Open a `.braw` / `.R3D` file when its bridge is available. `None` leaves the file to the other
/// openers (also how non-camera-raw paths pass through).
pub fn path_opener(path: &Path) -> Option<Result<SharedSource, MediaError>> {
    let format = Format::from_path(path)?;
    if !path.is_file() {
        return None;
    }
    open_with(path, format, bridge_path(format), None)
}

/// [`path_opener`] with an explicit bridge lookup and audio cache dir (tests).
fn open_with(path: &Path, format: Format, bridge: Option<PathBuf>, cache: Option<PathBuf>) -> Option<Result<SharedSource, MediaError>> {
    let Some(bridge) = bridge else {
        return Some(Err(MediaError::Unsupported(format!(
            "{}: {} clips need the {} helper (set {} or put it on PATH)",
            path.display(),
            format.label(),
            format.bridge_name(),
            format.bridge_env()
        ))));
    };
    Some(RawCamSource::open(path, format, bridge, cache).map(|s| s as SharedSource).map_err(MediaError::Decode))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timecode_frames() {
        let ntsc = FrameRate::new(24000, 1001);
        assert_eq!(timecode_to_frames("00:00:00:00", ntsc), Some(0));
        assert_eq!(timecode_to_frames("00:00:01:00", ntsc), Some(24));
        assert_eq!(timecode_to_frames("01:00:00:00", ntsc), Some(24 * 3600));
        assert_eq!(timecode_to_frames("13:37:12:34", FrameRate::new(25, 1)), Some((13 * 3600 + 37 * 60 + 12) * 25 + 34));
        assert_eq!(timecode_to_frames("0:00:00:00", ntsc), Some(0));
        assert_eq!(timecode_to_frames("00:00:00:60", ntsc), None);
        assert_eq!(timecode_to_frames("junk", ntsc), None);
    }

    #[test]
    fn tier_selection_and_dims() {
        assert_eq!(Format::Braw.tier_factor(1.0, false, false), 1);
        assert_eq!(Format::Braw.tier_factor(0.7, false, false), 1);
        assert_eq!(Format::Braw.tier_factor(0.5, false, false), 2);
        assert_eq!(Format::Braw.tier_factor(0.1, false, false), 4);
        assert_eq!(Format::R3d.tier_factor(0.1, false, false), 8);
        assert_eq!(Format::Braw.tier_factor(f32::NAN, false, false), 1);
        // Draft playback steps one tier down; the floor stays the smallest tier.
        assert_eq!(Format::Braw.tier_factor(1.0, true, false), 2);
        assert_eq!(Format::R3d.tier_factor(1.0, true, false), 2);
        assert_eq!(Format::Braw.tier_factor(0.1, true, false), 4);
        // A protocol 2 BRAW bridge has the eighth tier too (an older one is not asked for it).
        assert_eq!(Format::Braw.tier_factor(0.1, false, true), 8);
        assert_eq!(Format::Braw.tier_factor(0.2, false, true), 4);
        assert_eq!(Format::Braw.tier_factor(0.2, true, true), 8);
        assert_eq!(Format::Braw.tier_word(1), "full");
        assert_eq!(Format::Braw.tier_word(8), "eighth");
        assert_eq!(Format::R3d.tier_word(1), "premium");
        assert_eq!(Format::R3d.tier_word(8), "eighth");
    }

    #[test]
    fn shuttle_tier_steps() {
        assert_eq!(shuttle_steps(1.0), 0, "realtime playback keeps its tier");
        assert_eq!(shuttle_steps(1.5), 0);
        assert_eq!(shuttle_steps(1.6), 1, "double speed decodes one tier cheaper");
        assert_eq!(shuttle_steps(3.0), 1);
        assert_eq!(shuttle_steps(3.2), 2, "high-speed shuttle two tiers");
        // Stepping is capped at the format floor.
        assert_eq!(Format::Braw.stepped_down(1, 1, false), 2);
        assert_eq!(Format::Braw.stepped_down(1, 2, false), 4);
        assert_eq!(Format::Braw.stepped_down(4, 2, false), 4, "BRAW floors at quarter");
        assert_eq!(Format::Braw.stepped_down(8, 0, true), 8, "an eighth-tier request stays eighth");
        assert_eq!(Format::Braw.stepped_down(2, 2, true), 8);
        assert_eq!(Format::R3d.stepped_down(2, 2, false), 8);
        assert_eq!(Format::R3d.stepped_down(4, 9, false), 8, "absurd step counts stay at the floor");
    }

    #[test]
    fn shuttle_window_needs_history() {
        let s = Shuttle::default();
        assert_eq!(s.factor(FrameRate::FPS_25), 1.0, "empty window reads realtime");
        // A short burst (few samples over no time) never triggers, whatever the jump sizes.
        let mut burst = Shuttle::default();
        for i in 0..3_u64 {
            burst.observe(100 * i);
        }
        assert_eq!(burst.factor(FrameRate::FPS_25), 1.0);
    }

    #[test]
    fn scrub_bursts_start_with_the_second_quick_seek_and_end_at_rest() {
        let t0 = Instant::now();
        let ms = |n: u64| t0 + Duration::from_millis(n);
        let mut s = Scrub::default();
        assert!(!s.observe(ms(0), 10), "a single seek (a click) is no burst");
        assert!(s.observe(ms(40), 12), "a quick second seek elsewhere starts one");
        assert!(s.observe(ms(80), 12), "the same frame inside the burst stays in it");
        assert!(s.active(ms(100)));
        assert_eq!(s.rest_left(ms(100)), SCRUB_REST - Duration::from_millis(60));
        assert!(!s.active(ms(40) + SCRUB_REST), "rested");
        assert_eq!(s.rest_left(ms(1000)), Duration::ZERO);
        // a slow sequence of seeks (frame stepping) never becomes a burst
        let mut slow = Scrub::default();
        for k in 0..5_u64 {
            assert!(!slow.observe(ms(k * 400), k));
        }
    }

    #[test]
    fn read_ahead_indices_never_repeat() {
        // backwards at the clip start: frame 0 once, not five times (saturating_sub did that)
        assert_eq!(ahead_indices(1, -1, 100), vec![0]);
        assert_eq!(ahead_indices(2, -1, 100), vec![1, 0]);
        assert_eq!(ahead_indices(0, -1, 100), Vec::<u64>::new());
        // forwards stops at the clip end and at u64::MAX
        assert_eq!(ahead_indices(97, 1, 100), vec![98, 99]);
        assert_eq!(ahead_indices(u64::MAX - 1, 1, u64::MAX), Vec::<u64>::new());
        assert_eq!(ahead_indices(10, 0, 100), vec![11, 12, 13, 14, 15, 16]);
    }

    #[test]
    fn non_camera_files_are_not_ours() {
        assert_eq!(Format::from_path(Path::new("/tmp/a.MP4")), None);
        assert_eq!(Format::from_path(Path::new("/tmp/a.R3D")), Some(Format::R3d));
        assert_eq!(Format::from_path(Path::new("/tmp/a.braw")), Some(Format::Braw));
    }

    #[test]
    fn bridge_lookup_prefers_the_env_variable() {
        let dir = std::env::temp_dir();
        let env = dir.join("custom-bridge");
        assert_eq!(bridge_path_from(Some(env.clone().into_os_string()), Some(dir.clone().into_os_string()), Format::Braw), Some(env));
        // Without env: the first PATH entry holding the binary wins; an empty PATH finds nothing.
        let empty = dir.join(format!("filmcraft-empty-path-{}", std::process::id()));
        assert_eq!(bridge_path_from(None, Some(empty.into_os_string()), Format::Braw), None);
    }

    #[test]
    fn missing_bridge_yields_a_named_error() {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let file = dir.join("clip.braw");
        std::fs::write(&file, b"not really braw").ok();
        let Some(Err(MediaError::Unsupported(msg))) = open_with(&file, Format::Braw, None, None) else {
            panic!("expected an unsupported error naming the env var");
        };
        assert!(msg.contains("FILMCRAFT_BRAW_BRIDGE"), "{msg}");
        // A configured bridge that does not exist fails at probe time, not silently.
        assert!(matches!(open_with(&file, Format::Braw, Some(dir.join("nonexistent-bridge")), None), Some(Err(MediaError::Decode(_)))));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn audio_cache_key_tracks_file_identity() {
        let p = audio_cache_path(Format::Braw, Path::new("/tmp/A 001.braw"), Path::new("/cache"));
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        assert!(name.starts_with("braw-bridge-A_001_braw-"), "{name}");
        assert!(name.ends_with(".wav"), "{name}");
    }

    /// A `/bin/sh` stand-in for a real bridge: an 8-frame 16×8 clip whose frame `i` is filled
    /// with byte `i+1`, honouring `--probe-only`, `--debayer` (dimension factor), the persistent
    /// `frame <index>` / `quit` stdin protocol (`--persistent`), and `--extract-audio`
    /// (a 48 kHz mono 16-bit WAV with two silent samples). The clip file's first four bytes pick
    /// a misbehaviour: "slow" takes a second per frame (the contention test), "lazy" a second to
    /// start a decode session, "fail" fails every decode session at start-up, "huge" reports
    /// `u64::MAX` frames.
    #[cfg(target_os = "linux")]
    fn fake_bridge(dir: &Path) -> PathBuf {
        fake_bridge_with(dir, 1)
    }

    /// [`fake_bridge`] speaking protocol 2 (`v2`): capabilities in `metadata`, the debayer tier
    /// per command (`frame <index> <scale>`), and per-frame errors for indices past the clip
    /// (`{"type":"error","index":N}`, the session goes on).
    #[cfg(target_os = "linux")]
    fn fake_bridge_v2(dir: &Path) -> PathBuf {
        fake_bridge_with(dir, 2)
    }

    /// [`fake_bridge_v2`] that also serves many clips from one process (`--persistent --multi`,
    /// protocol 3; no shared memory). Every multi-clip process appends its PID to
    /// `<bridge>.multi` and every command line to `<bridge>.cmds`. A clip file starting with
    /// "nope" cannot be opened, one starting with "boom" crashes the process at its first frame
    /// (in either mode), "slow" takes a second per frame. With `<bridge>.nostart` present the
    /// multi-clip mode fails at start-up.
    #[cfg(target_os = "linux")]
    fn fake_bridge_v3(dir: &Path) -> PathBuf {
        fake_bridge_with(dir, 3)
    }

    /// The protocol 3 command loop of [`fake_bridge_v3`].
    #[cfg(target_os = "linux")]
    const FAKE_MULTI: &str = r#"if [ "$multi" = 1 ]; then
    echo $$ >> "$0.multi"
    [ -f "$0.nostart" ] && { echo '{"type":"error","message":"no GPU"}' >&2; exit 1; }
    echo '{"type":"ready","protocol":3,"caps":"scale,frame-errors,multi"}' >&2
    while IFS= read -r line; do
        echo "$line" >> "$0.cmds"
        set -- $line
        case "$1" in
            open)
                h=$2; p=${line#open $h }
                m=$(head -c 4 "$p" 2>/dev/null) || m=
                if [ ! -f "$p" ] || [ "$m" = nope ]; then
                    echo "{\"type\":\"error\",\"handle\":$h,\"message\":\"cannot open clip $h\"}" >&2
                else
                    eval "clip_$h=x$m"
                    echo "{\"type\":\"metadata\",\"handle\":$h,\"timecode\":\"01:02:03:04\",\"fps_num\":24,\"fps_den\":1,\"width\":16,\"height\":8,\"frame_count\":8,\"protocol\":3,\"caps\":\"scale,frame-errors,multi\"}" >&2
                fi;;
            frame)
                h=$2 i=$3 s=$4
                eval "m=\${clip_$h:-}"
                if [ -z "$m" ]; then
                    echo "{\"type\":\"error\",\"handle\":$h,\"index\":$i,\"message\":\"no such clip handle\"}" >&2
                    continue
                fi
                [ "$m" = xboom ] && exit 3
                if [ "$i" -ge 8 ]; then
                    echo "{\"type\":\"error\",\"handle\":$h,\"index\":$i,\"message\":\"frame $i beyond clip\"}" >&2
                    continue
                fi
                case "$s" in half) w=8 hh=4;; quarter) w=4 hh=2;; eighth) w=2 hh=1;; *) w=16 hh=8;; esac
                [ "$m" = xslow ] && sleep 1
                bytes=$((w * hh * 4))
                b=$(printf "\\$(printf '%03o' $(((i + 1) % 256)))")
                a=$(printf '\377')
                out= j=0
                while [ $j -lt $bytes ]; do out="$out$b$b$b$a"; j=$((j + 4)); done
                echo "{\"type\":\"frame\",\"handle\":$h,\"index\":$i,\"width\":$w,\"height\":$hh,\"format\":\"rgba8\"}" >&2
                printf '%s' "$out"
                echo "{\"type\":\"verify\",\"handle\":$h,\"index\":$i,\"width\":$w,\"height\":$hh}" >&2;;
            close) eval "clip_$2=";;
            quit) exit 0;;
            *) :;;
        esac
    done
    exit 0
fi
"#;

    #[cfg(target_os = "linux")]
    fn fake_bridge_with(dir: &Path, level: u8) -> PathBuf {
        let script = r#"#!/bin/sh
input= probe=0 persistent=0 start=0 debayer=full count=-1 audio=
while [ $# -gt 0 ]; do
    case "$1" in
        --input) input=$2; shift 2;;
        --probe-only) probe=1; shift;;
        --persistent) persistent=1; shift;;
        --debayer) debayer=$2; shift 2;;
        --start-frame) start=$2; shift 2;;
        --count) count=$2; shift 2;;
        --extract-audio) audio=$2; shift 2;;
        *) echo "{\"type\":\"error\",\"message\":\"bad arg $1\"}" >&2; exit 1;;
    esac
done
factor=1
[ "$debayer" = half ] && factor=2
[ "$debayer" = quarter ] && factor=4
[ "$debayer" = eighth ] && factor=8
w=$((16/factor)) h=$((8/factor))
slow=0 mode=
[ -f "$input" ] && mode=$(head -c 4 "$input" 2>/dev/null)
[ "$mode" = slow ] && slow=1
frames=8
[ "$mode" = huge ] && frames=18446744073709551615
[ "$persistent" = 1 ] && [ "$mode" = lazy ] && sleep 1
if [ "$persistent" = 1 ] && [ "$mode" = fail ]; then
    echo $$ >> "$input.sessions" 2>/dev/null || :
    echo '{"type":"error","message":"no GPU"}' >&2
    exit 1
fi
echo "{\"type\":\"metadata\",\"timecode\":\"01:02:03:04\",\"fps_num\":24,\"fps_den\":1,\"width\":16,\"height\":8,\"frame_count\":$frames$CAPS}" >&2
[ "$probe" = 1 ] && { echo '{"type":"done"}' >&2; exit 0; }
if [ -n "$audio" ]; then
    printf 'RIFF$\000\000\000WAVEfmt \020\000\000\000\001\000\001\000\200\273\000\000\000\167\001\000\002\000\020\000data\004\000\000\000\000\000\000\000' > "$audio"
    echo '{"type":"done"}' >&2
    exit 0
fi
# Every decode session appends its PID: the tests assert that seeks reuse the process.
echo $$ >> "$input.sessions" 2>/dev/null || :
emit_frame() {
    i=$1
    w=$((16/factor)) h=$((8/factor))
    case "$2" in
        full) w=16 h=8;; half) w=8 h=4;; quarter) w=4 h=2;; eighth) w=2 h=1;;
    esac
    if [ -n "$CAPS" ] && { [ "$i" -ge "$frames" ] || { [ "$mode" = bad5 ] && [ "$i" = 5 ]; }; }; then
        echo "{\"type\":\"error\",\"index\":$i,\"message\":\"frame $i cannot be decoded\"}" >&2
        return
    fi
    bytes=$((w * h * 4))
    b=$(printf "\\$(printf '%03o' $(((i + 1) % 256)))")
    a=$(printf '\377')
    out= j=0
    while [ $j -lt $bytes ]; do out="$out$b$b$b$a"; j=$((j + 4)); done
    echo "{\"type\":\"frame\",\"index\":$i,\"width\":$w,\"height\":$h,\"format\":\"rgba8\"}" >&2
    printf '%s' "$out"
    echo "{\"type\":\"verify\",\"index\":$i,\"width\":$w,\"height\":$h}" >&2
    [ "$slow" = 1 ] && sleep 1
}
if [ "$persistent" = 1 ]; then
    while IFS= read -r line; do
        case "$line" in
            frame\ *) [ "$mode" = boom ] && exit 3; set -- ${line#frame }; emit_frame "$1" "$2";;
            quit) break;;
            "") : ;;
            *) echo "{\"type\":\"error\",\"message\":\"unknown command\"}" >&2;;
        esac
    done
    echo '{"type":"done"}' >&2
    exit 0
fi
i=$start end=8
if [ "$count" -ge 0 ] && [ $((start + count)) -lt $end ]; then end=$((start + count)); fi
while [ "$i" -lt "$end" ]; do
    emit_frame "$i"
    i=$((i + 1))
done
echo '{"type":"done"}' >&2
"#;
        let caps = match level {
            1 => "",
            2 => r#","protocol":2,"caps":"scale,frame-errors""#,
            _ => r#","protocol":2,"caps":"scale,frame-errors,multi""#,
        };
        let mut script = script.replacen("#!/bin/sh\n", &format!("#!/bin/sh\nCAPS='{caps}'\nmulti=0\n"), 1);
        if level >= 3 {
            script = script.replacen(
                "        --persistent) persistent=1; shift;;\n",
                "        --persistent) persistent=1; shift;;\n        --multi) multi=1; shift;;\n",
                1,
            );
            script = script.replacen("done\nfactor=1\n", &format!("done\n{FAKE_MULTI}factor=1\n"), 1);
        }
        let path = dir.join(match level {
            1 => "fake-bridge",
            2 => "fake-bridge-v2",
            _ => "fake-bridge-v3",
        });
        std::fs::write(&path, script).ok();
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755));
        path
    }

    /// Open through the fake bridge: metadata, sequential frames, seeks (served by the same
    /// persistent session), the debayer tier, and the extracted-audio track.
    #[cfg(target_os = "linux")]
    #[test]
    fn fake_bridge_end_to_end() {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let cache = dir.join("cache");
        let clip = dir.join("clip.braw");
        std::fs::write(&clip, b"placeholder").ok();
        let bridge = fake_bridge(&dir);

        let opened = RawCamSource::open(&clip, Format::Braw, bridge, Some(cache));
        let src = opened.unwrap_or_else(|e| panic!("fake bridge clip must open: {e}"));
        let info = src.info();
        let v = info.video.as_ref().unwrap_or_else(|| panic!("video stream"));
        assert_eq!((v.width, v.height), (16, 8));
        assert_eq!((v.frame_rate.num, v.frame_rate.den), (24, 1));
        assert_eq!(info.start_timecode, Some((3600 + 2 * 60 + 3) * 24 + 4));
        assert_eq!(info.audio_streams.len(), 1);

        let at = |k: i64| FrameRequest { time: Tick::from_rational(k, 1, 24), scale: 1.0 };
        let pixel = |k: i64| -> u8 {
            src.video_frame(at(k))
                .ok()
                .and_then(|f| match &f.data {
                    filmcraft_frame::PixelData::Rgba8(d) => d.first().copied(),
                    _ => None,
                })
                .unwrap_or(u8::MAX)
        };
        // Sequential playback.
        assert_eq!(pixel(0), 1);
        assert_eq!(pixel(1), 2);
        assert_eq!(pixel(2), 3);
        // Read-ahead: frames past the playhead decode in the background while playback runs,
        // so showing them is a cache hit instead of a decode on the display path.
        std::thread::sleep(Duration::from_millis(300));
        let ahead: Vec<(u64, u32)> = src.cache.lock().unwrap_or_else(|e| e.into_inner()).keys().copied().collect();
        assert!(ahead.contains(&(5, 1)), "frame 5 read ahead: {ahead:?}");
        assert_eq!(pixel(5), 6, "the read-ahead frame holds its own pixels");
        // Cached frame.
        assert_eq!(pixel(0), 1);
        // Backwards seek.
        assert_eq!(pixel(0), 1);
        assert_eq!(pixel(1), 2);
        // A far forward jump.
        assert_eq!(pixel(7), 8);
        // The last frame clamps; a frame past the end returns the last one.
        assert_eq!(pixel(8), 8);
        assert_eq!(pixel(100), 8);
        // All those full-resolution seeks ran on ONE persistent session, plus ONE for the
        // quarter tier below: the fake bridge logs a PID line per decode process.
        let sessions = std::fs::read_to_string(clip.with_extension("braw.sessions")).unwrap_or_default();
        assert_eq!(sessions.lines().count(), 1, "seeks must not restart the bridge: {sessions}");
        // A thumbnail-scale request debayers at quarter resolution.
        let small =
            src.video_frame(FrameRequest { time: Tick::ZERO, scale: 0.1 }).map_err(|e| panic!("quarter-tier frame: {e}")).ok().and_then(|f| match &f.data {
                filmcraft_frame::PixelData::Rgba8(d) => Some((f.width, f.height, d.to_vec())),
                _ => None,
            });
        let Some((w, h, d)) = small else { panic!("quarter-tier frame") };
        assert_eq!((w, h), (4, 2));
        assert_eq!(d.first(), Some(&1));
        assert_eq!(d.get(3), Some(&255), "the fake bridge's alpha passes through");
        let sessions = std::fs::read_to_string(clip.with_extension("braw.sessions")).unwrap_or_default();
        assert_eq!(sessions.lines().count(), 2, "one session per tier: {sessions}");
        // The audio track reads through the extracted WAV.
        let audio = src.audio(0, 4, 48000).ok().unwrap_or_else(|| panic!("audio"));
        assert_eq!(audio.sample_rate, 48000);
        assert_eq!(audio.channels.len(), 1);
        assert!(audio.channels.iter().all(|c| c.iter().all(|s| *s == 0.0)));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A protocol 2 bridge: one session serves every tier (the scale word goes with each
    /// command), and BRAW gets the eighth tier.
    #[cfg(target_os = "linux")]
    #[test]
    fn protocol_2_bridge_serves_every_tier_from_one_session() {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-v2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let clip = dir.join("clip.braw");
        std::fs::write(&clip, b"placeholder").ok();
        let src = RawCamSource::open(&clip, Format::Braw, fake_bridge_v2(&dir), Some(dir.join("cache"))).unwrap_or_else(|e| panic!("clip must open: {e}"));
        assert!(src.per_command(), "the probe saw the capability");
        src.prefetch_started.store(true, Ordering::Relaxed); // no read-ahead: deterministic tiers
        let dims = |scale: f32, k: i64| {
            src.video_frame(FrameRequest { time: Tick::from_rational(k, 1, 24), scale })
                .map(|f| (f.width, f.height))
                .unwrap_or_else(|e| panic!("{scale}: {e:?}"))
        };
        assert_eq!(dims(1.0, 0), (16, 8));
        assert_eq!(dims(0.5, 1), (8, 4));
        assert_eq!(dims(0.3, 2), (4, 2));
        assert_eq!(dims(0.1, 3), (2, 1), "BRAW eighth tier on a protocol 2 bridge");
        assert_eq!(dims(1.0, 4), (16, 8));
        let sessions = std::fs::read_to_string(clip.with_extension("braw.sessions")).unwrap_or_default();
        assert_eq!(sessions.lines().count(), 1, "one session for all four tiers: {sessions}");
        let keys: Vec<u32> = src.video.lock().unwrap_or_else(|e| e.into_inner()).sessions.iter().map(|s| s.tier).collect();
        assert_eq!(keys, vec![ANY_TIER]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The frames all sources cache share one byte budget: inserting past it takes frames from
    /// the source idle longest first (farthest from its playhead), never the frame just stored.
    /// Each source kept up to 24 full-tier frames on its own (~1.8 GB for a 6K clip) with
    /// nothing bounding a timeline of them.
    #[cfg(target_os = "linux")]
    #[test]
    fn cached_frames_of_all_sources_share_one_budget() {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-budget-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let bridge = fake_bridge_v2(&dir);
        let open = |name: &str| {
            let clip = dir.join(name);
            std::fs::write(&clip, b"placeholder").ok();
            let src = RawCamSource::open(&clip, Format::Braw, bridge.clone(), Some(dir.join("cache"))).unwrap_or_else(|e| panic!("{name} must open: {e}"));
            src.prefetch_started.store(true, Ordering::Relaxed);
            src
        };
        let (old, mid, new) = (open("old.braw"), open("mid.braw"), open("new.braw"));
        let all = vec![old.clone(), mid.clone(), new.clone()];
        let frame = || Arc::new(VideoFrame::rgba8(10, 25, vec![0; 1000])); // 1000 bytes
        let now = Instant::now();
        for (src, ago) in [(&old, 30), (&mid, 20), (&new, 0)] {
            *src.activity.lock().unwrap_or_else(|e| e.into_inner()) = now.checked_sub(Duration::from_secs(ago)).unwrap_or(now);
        }
        for i in 0..4 {
            old.cache_frame_within((i, 1), frame(), &all, 10_000);
            mid.cache_frame_within((i, 1), frame(), &all, 10_000);
        }
        old.last.store(0, Ordering::Relaxed);
        for i in 0..4 {
            new.cache_frame_within((i, 1), frame(), &all, 9_000);
        }
        let keys = |s: &Arc<RawCamSource>| s.cache.lock().unwrap_or_else(|e| e.into_inner()).keys().map(|k| k.0).collect::<Vec<_>>();
        let total: usize = all.iter().map(|s| s.cached_bytes()).sum();
        assert!(total <= 9_000, "over budget: {total}");
        assert_eq!(keys(&new), vec![0, 1, 2, 3], "the source in use keeps its frames");
        assert_eq!(keys(&mid), vec![0, 1, 2, 3], "the source idle for less long is untouched");
        assert_eq!(keys(&old), vec![0], "the longest idle gives up frames, farthest from its playhead first");
        // a budget smaller than one frame still keeps the frame just stored
        new.cache_frame_within((9, 1), frame(), std::slice::from_ref(&new), 10);
        assert_eq!(keys(&new), vec![9]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A frame the bridge cannot decode fails that one request; the session survives (it was
    /// killed and respawned for every such frame, and the respawn back-off then failed the
    /// healthy frames around it too).
    #[cfg(target_os = "linux")]
    #[test]
    fn a_per_frame_error_keeps_the_session() {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-bad5-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let clip = dir.join("bad5.braw");
        std::fs::write(&clip, b"bad5 placeholder").ok();
        let src = RawCamSource::open(&clip, Format::Braw, fake_bridge_v2(&dir), Some(dir.join("cache"))).unwrap_or_else(|e| panic!("clip must open: {e}"));
        let at = |k: i64| FrameRequest { time: Tick::from_rational(k, 1, 24), scale: 1.0 };
        for _ in 0..3 {
            let bad = src.video_frame(at(5));
            assert!(matches!(&bad, Err(MediaError::Decode(e)) if e.contains("frame 5")), "got {bad:?}");
        }
        for k in [4, 6, 7, 0] {
            let f = src.video_frame(at(k)).unwrap_or_else(|e| panic!("frame {k} after the bad one: {e:?}"));
            assert_eq!(f.pts, at(k).time);
        }
        let sessions = std::fs::read_to_string(clip.with_extension("braw.sessions")).unwrap_or_default();
        assert_eq!(sessions.lines().count(), 1, "the bridge was not restarted: {sessions}");
        let state = src.video.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!((state.failures, state.retry_at), (0, None), "no back-off for a bad frame");
        drop(state);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A bridge that answers its first `frame` command (metadata, `frame` event, pixels,
    /// `verify` event), then dies in the middle of the second (its `frame` event, no pixels):
    /// the decode stream breaks mid-session the way a crashed SDK process does.
    #[cfg(target_os = "linux")]
    fn dying_bridge(dir: &Path) -> PathBuf {
        let script = r#"#!/bin/sh
while [ $# -gt 0 ]; do
    case "$1" in
        --input) shift 2;;
        --probe-only) echo "{\"type\":\"metadata\",\"timecode\":\"00:00:00:00\",\"fps_num\":24,\"fps_den\":1,\"width\":16,\"height\":8,\"frame_count\":8}" >&2; echo '{"type":"done"}' >&2; exit 0;;
        --debayer|--start-frame) shift 2;;
        --count) shift 2;;
        --persistent) shift;;
        --extract-audio) shift 2;;
        *) echo "{\"type\":\"error\",\"message\":\"bad arg\"}" >&2; exit 1;;
    esac
done
echo "{\"type\":\"metadata\",\"timecode\":\"00:00:00:00\",\"fps_num\":24,\"fps_den\":1,\"width\":16,\"height\":8,\"frame_count\":8}" >&2
IFS= read -r line
case "$line" in
    frame\ *) ;;
    *) exit 0;;
esac
i=${line#frame }
echo "{\"type\":\"frame\",\"index\":$i,\"width\":16,\"height\":8,\"format\":\"rgba8\"}" >&2
b=$(printf '\001')
a=$(printf '\377')
j=0 out=
while [ $j -lt 512 ]; do out="$out$b$b$b$a"; j=$((j + 4)); done
printf '%s' "$out"
echo "{\"type\":\"verify\",\"index\":$i,\"width\":16,\"height\":8}" >&2
IFS= read -r line || exit 0
i=${line#frame }
echo "{\"type\":\"frame\",\"index\":$i,\"width\":16,\"height\":8,\"format\":\"rgba8\"}" >&2
exit 0
"#;
        let path = dir.join("dying-bridge");
        std::fs::write(&path, script).ok();
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755));
        path
    }

    /// A decode stream that dies mid-session must surface an error, and the next request must
    /// start over and succeed (a wedged session would freeze every later frame).
    #[cfg(target_os = "linux")]
    #[test]
    fn dying_bridge_session_recovers() {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-dying-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let clip = dir.join("clip.braw");
        std::fs::write(&clip, b"placeholder").ok();
        let bridge = dying_bridge(&dir);
        let raw = RawCamSource::open(&clip, Format::Braw, bridge, Some(dir.join("cache"))).unwrap_or_else(|e| panic!("clip must open: {e}"));
        let at = |k: i64| FrameRequest { time: Tick::from_rational(k, 1, 24), scale: 1.0 };
        // No read-ahead: it would race the requests below for the dying session.
        raw.prefetch_started.store(true, Ordering::Relaxed);
        // The bridge serves frame 3, then dies while answering frame 4.
        assert!(raw.video_frame(at(3)).is_ok());
        let second = raw.video_frame(at(4));
        let Err(e) = &second else { panic!("frame 4 must fail, got Ok") };
        assert!(matches!(e, MediaError::Decode(_)), "frame 4: {e:?}");
        // The failed session is gone: frame 4 restarts the bridge and decodes.
        assert!(raw.video_frame(at(4)).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Contention policy: a background request (thumbnail) takes a nearby cached frame and
    /// leaves the decode slot alone; an interactive request waits for its exact frame — accepting
    /// a neighbouring one would rock playback a frame back and forth — but never past its own
    /// cancellation.
    #[cfg(target_os = "linux")]
    #[test]
    fn contended_requests_wait_or_yield() {
        use std::sync::atomic::AtomicBool;

        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-contended-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let bridge = fake_bridge(&dir);
        let open_slow = || {
            let clip = dir.join("slow.braw");
            std::fs::write(&clip, b"slow placeholder").ok();
            RawCamSource::open(&clip, Format::Braw, bridge.clone(), Some(dir.join("cache-slow"))).ok()
        };
        let at = |k: i64| FrameRequest { time: Tick::from_rational(k, 1, 24), scale: 1.0 };

        // Frame 0 lands in the cache, then a slow decode of frame 1 holds the decode slot.
        let src = open_slow().unwrap_or_else(|| panic!("slow clip must open"));
        assert!(src.video_frame(at(0)).is_ok());
        let worker = std::thread::spawn({
            let src = Arc::clone(&src);
            move || src.video_frame(at(1))
        });
        std::thread::sleep(Duration::from_millis(150)); // the worker now holds the slot

        // A cancelled interactive request gives up instead of queueing behind the decode.
        let cancelled = Arc::new(AtomicBool::new(true));
        let gone = filmcraft_media::cancel::with_cancel(&cancelled, || src.video_frame(at(2)));
        assert!(matches!(gone, Err(MediaError::Cancelled)), "got {gone:?}");

        // A background request answers with the nearby cached frame without waiting.
        let background = filmcraft_media::cancel::with_background(true, || src.video_frame(at(2)));
        let frame = background.unwrap_or_else(|e| panic!("a cached frame must be served, got {e:?}"));
        assert_eq!(frame.pts, Tick::ZERO, "frame 0 is the nearest cached one");

        // A live interactive request waits for the slot and gets its exact frame.
        let live = Arc::new(AtomicBool::new(false));
        let exact = filmcraft_media::cancel::with_cancel(&live, || src.video_frame(at(2)));
        let frame = exact.unwrap_or_else(|e| panic!("the exact frame must arrive, got {e:?}"));
        assert_eq!(frame.pts, Tick::from_rational(2, 1, 24), "frame 2 itself, not a neighbour");
        assert!(worker.join().is_ok_and(|r| r.is_ok()));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A paused monitor's scrub burst is served at the quarter tier and flagged provisional; the
    /// refining request (no scrub hint) waits for the playhead to rest, then decodes the exact
    /// tier. A cancelled refinement gives up without decoding.
    #[cfg(target_os = "linux")]
    #[test]
    fn scrub_burst_serves_quarter_then_refines_at_rest() {
        use filmcraft_media::cancel::{with_cancel, with_scrub};
        use filmcraft_media::pending::take_provisional;
        use std::sync::atomic::AtomicBool;

        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-scrub-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let clip = dir.join("clip.braw");
        std::fs::write(&clip, b"placeholder").ok();
        let src = RawCamSource::open(&clip, Format::Braw, fake_bridge(&dir), Some(dir.join("cache"))).unwrap_or_else(|e| panic!("clip must open: {e}"));
        let at = |k: i64| FrameRequest { time: Tick::from_rational(k, 1, 24), scale: 1.0 };
        let live = Arc::new(AtomicBool::new(false));
        let dims = |f: &Arc<VideoFrame>| (f.width, f.height);
        let _ = take_provisional();

        // A click: the exact frame (16×8), nothing provisional. The last frame, so the read-ahead
        // it starts has nothing to decode that the drag (backwards) could hit.
        let first = with_cancel(&live, || with_scrub(true, || src.video_frame(at(7)))).unwrap_or_else(|e| panic!("seek: {e:?}"));
        assert_eq!(dims(&first), (16, 8));
        assert!(!take_provisional());
        // The drag goes on: quarter tier (4×2), flagged provisional.
        for k in [5, 3] {
            let f = with_cancel(&live, || with_scrub(true, || src.video_frame(at(k)))).unwrap_or_else(|e| panic!("scrub {k}: {e:?}"));
            assert_eq!(dims(&f), (4, 2), "frame {k} during the burst");
            assert!(take_provisional(), "frame {k} is a stand-in");
        }
        // A cancelled refinement (the playhead moved on) leaves without decoding.
        let gone = with_cancel(&Arc::new(AtomicBool::new(true)), || src.video_frame(at(3)));
        assert!(matches!(gone, Err(MediaError::Cancelled)), "got {gone:?}");
        // The monitor asks again without the hint: it waits for the rest, then gets 16×8.
        let t = Instant::now();
        let exact = with_cancel(&live, || src.video_frame(at(3))).unwrap_or_else(|e| panic!("refine: {e:?}"));
        assert_eq!(dims(&exact), (16, 8));
        assert!(!take_provisional());
        assert!(t.elapsed() >= Duration::from_millis(50), "the refinement waited for the rest: {:?}", t.elapsed());
        // After the rest a seek is a single click again: exact.
        std::thread::sleep(SCRUB_REST);
        let click = with_cancel(&live, || with_scrub(true, || src.video_frame(at(0)))).unwrap_or_else(|e| panic!("click: {e:?}"));
        assert_eq!(dims(&click), (16, 8));
        assert!(!take_provisional());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Real-material scrub measurement (`cargo test -- --ignored --nocapture`): a paused-monitor
    /// drag over a clip at full-resolution scale, through the app's serve path, reporting ms per
    /// seek and the tier served; then the refinement at rest. Needs `FILMCRAFT_RAWCAM_REAL`.
    #[test]
    #[ignore = "needs a real camera file (FILMCRAFT_RAWCAM_REAL) and the vendor bridge"]
    fn real_clip_scrub_timing() {
        use filmcraft_media::cancel::{with_cancel, with_scrub};
        use std::sync::atomic::AtomicBool;
        let Some(path) = std::env::var_os("FILMCRAFT_RAWCAM_REAL").map(PathBuf::from) else {
            panic!("set FILMCRAFT_RAWCAM_REAL to a .braw/.R3D file");
        };
        let format = Format::from_path(&path).unwrap_or_else(|| panic!("{} is not braw/r3d", path.display()));
        let bridge = bridge_path(format).unwrap_or_else(|| panic!("no {} on PATH", format.bridge_name()));
        let src = RawCamSource::open(&path, format, bridge, None).unwrap_or_else(|e| panic!("open failed: {e}"));
        let rate = src.info().video.as_ref().map_or(FrameRate::FPS_25, |v| v.frame_rate);
        let live = Arc::new(AtomicBool::new(false));
        let req = |idx: u64| FrameRequest { time: Tick::from_rational(idx as i64, rate.den, rate.num), scale: 1.0 };
        let last = src.frame_count.saturating_sub(1);
        // warm both sessions (SDK start-up is not what is measured)
        let _ = with_cancel(&live, || src.video_frame(req(0)));
        let _ = with_cancel(&live, || with_scrub(true, || src.video_frame(req(1))));
        let _ = with_cancel(&live, || with_scrub(true, || src.video_frame(req(2))));
        std::thread::sleep(Duration::from_millis(400));
        for (label, hint) in [("without scrub hint (before)", false), ("with scrub hint (after)", true)] {
            let mut ms = Vec::new();
            let t_all = Instant::now();
            for k in 0..30_u64 {
                let idx = (10 + k * 3).min(last);
                let t = Instant::now();
                let f = with_cancel(&live, || with_scrub(hint, || src.video_frame(req(idx)))).unwrap_or_else(|e| panic!("seek {idx}: {e:?}"));
                ms.push((t.elapsed().as_secs_f64() * 1000.0, f.width));
                std::thread::sleep(Duration::from_millis(16)); // one UI refresh between moves
            }
            let mean = ms.iter().map(|m| m.0).sum::<f64>() / ms.len() as f64;
            let worst = ms.iter().map(|m| m.0).fold(0.0, f64::max);
            eprintln!("{label}: mean {mean:.1} ms, worst {worst:.1} ms per seek, total {:?}", t_all.elapsed());
            eprintln!("  per seek (ms@width): {}", ms.iter().map(|(t, w)| format!("{t:.0}@{w}")).collect::<Vec<_>>().join(" "));
            let t = Instant::now();
            let refined = with_cancel(&live, || src.video_frame(req((10 + 29 * 3).min(last)))).unwrap_or_else(|e| panic!("refine: {e:?}"));
            eprintln!("  refine at rest: {:.1} ms ({} px wide)", t.elapsed().as_secs_f64() * 1000.0, refined.width);
            std::thread::sleep(Duration::from_millis(600));
        }
    }

    #[test]
    fn frame_index_survives_any_frame_count() {
        let rate = FrameRate::FPS_25;
        let at = |k: i64| rate.tick_of(k);
        // `last as i64` turned u64::MAX frames into -1, and clamp(0, -1) panicked
        assert_eq!(frame_index(rate, at(3), u64::MAX), 3);
        assert_eq!(frame_index(rate, Tick(i64::MAX), u64::MAX), u64::try_from(rate.frame_at(Tick(i64::MAX))).unwrap_or(0));
        assert_eq!(frame_index(rate, at(100), 8), 7, "past the end: the last frame");
        assert_eq!(frame_index(rate, Tick(-5), 8), 0, "before the start: the first frame");
        assert_eq!(frame_index(rate, at(3), 0), 0, "an empty clip has nothing but 0");
    }

    #[test]
    fn failing_bridges_back_off() {
        let t0 = Instant::now();
        let mut v = VideoState::new();
        v.note_failure(t0);
        assert_eq!(v.backoff_left(t0), None, "one failure retries at once");
        v.note_failure(t0);
        assert_eq!(v.backoff_left(t0), Some(RESPAWN_BACKOFF));
        v.note_failure(t0);
        assert_eq!(v.backoff_left(t0), Some(RESPAWN_BACKOFF * 2), "doubling");
        for _ in 0..100 {
            v.note_failure(t0);
        }
        assert_eq!(v.backoff_left(t0), Some(RESPAWN_BACKOFF * 32), "capped at 16 s");
        assert_eq!(v.backoff_left(t0 + Duration::from_secs(17)), None);
        v.note_success();
        assert_eq!((v.failures, v.backoff_left(t0)), (0, None));
    }

    /// A decode that panics while holding the slot must not leave it taken: every later request
    /// would wait for it forever.
    #[test]
    fn a_panicking_decode_releases_the_slot() {
        let video = Mutex::new(VideoState::new());
        video.lock().unwrap_or_else(|e| e.into_inner()).busy = true;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = Lease::new(&video);
            panic!("decoder bug");
        }));
        assert!(r.is_err());
        assert!(!video.lock().unwrap_or_else(|e| e.into_inner()).busy);
    }

    /// A bridge reporting `u64::MAX` frames: the duration stays positive and requests do not
    /// panic (they did: `frame_count as i64` was -1).
    #[cfg(target_os = "linux")]
    #[test]
    fn absurd_frame_counts_are_clamped() {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-huge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let clip = dir.join("huge.braw");
        std::fs::write(&clip, b"huge placeholder").ok();
        let src = RawCamSource::open(&clip, Format::Braw, fake_bridge(&dir), Some(dir.join("cache"))).unwrap_or_else(|e| panic!("clip must open: {e}"));
        assert!(src.info().duration > Tick::ZERO, "duration {:?}", src.info().duration);
        let f = src.video_frame(FrameRequest { time: Tick::from_rational(3, 1, 24), scale: 1.0 }).unwrap_or_else(|e| panic!("frame 3: {e:?}"));
        assert_eq!(f.pts, Tick::from_rational(3, 1, 24));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A bridge that fails at start-up is retried once at once, then backed off: requests in the
    /// back-off fail fast without starting another process.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_failing_bridge_is_not_respawned_per_request() {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-fail-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let clip = dir.join("fail.braw");
        std::fs::write(&clip, b"fail placeholder").ok();
        let src = RawCamSource::open(&clip, Format::Braw, fake_bridge(&dir), Some(dir.join("cache"))).unwrap_or_else(|e| panic!("clip must open: {e}"));
        let at = |k: i64| FrameRequest { time: Tick::from_rational(k, 1, 24), scale: 1.0 };
        let spawns = || std::fs::read_to_string(clip.with_extension("braw.sessions")).unwrap_or_default().lines().count();
        for k in 0..6 {
            assert!(matches!(src.video_frame(at(k)), Err(MediaError::Decode(_))), "frame {k}");
        }
        assert_eq!(spawns(), 2, "two attempts, then the back-off");
        std::thread::sleep(RESPAWN_BACKOFF + Duration::from_millis(100));
        assert!(src.video_frame(at(0)).is_err());
        assert_eq!(spawns(), 3, "tried again after the back-off");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Starting a bridge (seconds of SDK start-up) must not hold the sessions lock: a thumbnail
    /// asking meanwhile answers at once instead of blocking behind the start-up.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_starting_bridge_does_not_block_other_requests() {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-lazy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let clip = dir.join("lazy.braw");
        std::fs::write(&clip, b"lazy placeholder").ok();
        let src = RawCamSource::open(&clip, Format::Braw, fake_bridge(&dir), Some(dir.join("cache"))).unwrap_or_else(|e| panic!("clip must open: {e}"));
        let at = |k: i64| FrameRequest { time: Tick::from_rational(k, 1, 24), scale: 1.0 };
        let worker = std::thread::spawn({
            let src = Arc::clone(&src);
            move || src.video_frame(at(0))
        });
        std::thread::sleep(Duration::from_millis(200)); // the worker is starting the bridge
        let t = Instant::now();
        let thumb = filmcraft_media::cancel::with_background(true, || src.video_frame(at(0)));
        assert!(matches!(thumb, Err(MediaError::Cancelled)), "nothing cached yet: {thumb:?}");
        assert!(t.elapsed() < Duration::from_millis(400), "blocked {:?} behind the start-up", t.elapsed());
        assert!(worker.join().is_ok_and(|r| r.is_ok()));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Real-material test (`cargo test -- --ignored`): plays a clip through the same
    /// serve/prefetch machinery the app uses, against the real bridge on `PATH`. Requires
    /// `FILMCRAFT_RAWCAM_REAL` to name a `.braw`/`.R3D` file.
    #[test]
    #[ignore = "needs a real camera file (FILMCRAFT_RAWCAM_REAL) and the vendor bridge"]
    fn real_clip_plays_through_the_app_path() {
        let Some(path) = std::env::var_os("FILMCRAFT_RAWCAM_REAL").map(PathBuf::from) else {
            panic!("set FILMCRAFT_RAWCAM_REAL to a .braw/.R3D file");
        };
        let format = Format::from_path(&path).unwrap_or_else(|| panic!("{} is not braw/r3d", path.display()));
        let bridge = bridge_path(format).unwrap_or_else(|| panic!("no {} on PATH", format.bridge_name()));
        let src = RawCamSource::open(&path, format, bridge, None).unwrap_or_else(|e| panic!("open failed: {e}"));
        let rate = src.info().video.as_ref().map_or(FrameRate::FPS_25, |v| v.frame_rate);
        let frames = src.frame_count.min(40);
        let start = std::time::Instant::now();
        for idx in 0..frames {
            let req = FrameRequest { time: Tick::from_rational(idx as i64, rate.den, rate.num), scale: 0.5 };
            let t0 = std::time::Instant::now();
            let frame = src.video_frame(req).unwrap_or_else(|e| panic!("frame {idx} failed: {e:?}"));
            eprintln!("frame {idx:>3}: {:>6.1} ms  ({}×{})", t0.elapsed().as_secs_f64() * 1000.0, frame.width, frame.height);
        }
        eprintln!("{} frames in {:.2}s", frames, start.elapsed().as_secs_f64());
    }

    /// Real-material test (`cargo test -- --ignored`): a single `frame` request followed by a
    /// 6-command batch on the same session — the exact interleaving playback produces.
    #[test]
    #[ignore = "needs a real camera file (FILMCRAFT_RAWCAM_REAL) and the vendor bridge"]
    fn real_clip_single_then_batch() {
        let Some(path) = std::env::var_os("FILMCRAFT_RAWCAM_REAL").map(PathBuf::from) else {
            panic!("set FILMCRAFT_RAWCAM_REAL to a .braw/.R3D file");
        };
        let format = Format::from_path(&path).unwrap_or_else(|| panic!("{} is not braw/r3d", path.display()));
        let bridge = bridge_path(format).unwrap_or_else(|| panic!("no {} on PATH", format.bridge_name()));
        let src = RawCamSource::open(&path, format, bridge, None).unwrap_or_else(|e| panic!("open failed: {e}"));
        let pooled = src.video.lock().unwrap_or_else(|e| e.into_inner()).take_session(2);
        let (key, mut session) = src.open_session(pooled, 2).unwrap_or_else(|e| panic!("spawn: {e}"));
        let scale = src.scale_word(key, 2);
        let t0 = std::time::Instant::now();
        let one = session.request_frame(30, scale, FRAME_TIMEOUT).unwrap_or_else(|e| panic!("single request failed: {e}"));
        let one = one.unwrap_or_else(|e| panic!("frame 30: {e}"));
        eprintln!("single frame 30: {:?} ({}×{}, {:?})", t0.elapsed(), one.width, one.height, one.format);
        let t1 = std::time::Instant::now();
        let batch = session.request_batch(&[31, 32, 33, 34, 35, 36], scale, Duration::from_secs(10));
        eprintln!("batch 31..36: {:?} ({} frames)", t1.elapsed(), batch.as_ref().map_or(0, |b| b.len()));
        let _ = batch.unwrap_or_else(|e| panic!("batch after single failed: {e}"));
    }

    /// Transport benchmark on real material (`cargo test --release -p filmcraft-codecs
    /// real_clip_transport_bench -- --ignored --nocapture`): random seeks per tier through one
    /// session (ms per frame, split into decode wait / pixel transport / conversion), then
    /// sequential full-resolution throughput through a batch and through the app's serve path.
    #[test]
    #[ignore = "needs a real camera file (FILMCRAFT_RAWCAM_REAL) and the vendor bridge"]
    fn real_clip_transport_bench() {
        let Some(path) = std::env::var_os("FILMCRAFT_RAWCAM_REAL").map(PathBuf::from) else {
            panic!("set FILMCRAFT_RAWCAM_REAL to a .braw/.R3D file");
        };
        let format = Format::from_path(&path).unwrap_or_else(|| panic!("{} is not braw/r3d", path.display()));
        let bridge = bridge_path(format).unwrap_or_else(|| panic!("no {} on PATH", format.bridge_name()));
        let meta = bridge::probe(&bridge, &path, PROBE_TIMEOUT).unwrap_or_else(|e| panic!("probe: {e}"));
        let rate = FrameRate::new(i64::from(meta.fps_num), i64::from(meta.fps_den.max(1)));
        let count = meta.frame_count;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let mut session = bench_session(&bridge, &path, format);
        for tier in [8_u32, 4, 2, 1] {
            if !format.has_eighth(session.src.per_command()) && tier == 8 {
                continue;
            }
            let _ = bench_frame(&mut session, 0, tier, format, rate);
            let (mut total, mut decode, mut pixels, mut convert) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut dims = (0, 0);
            for k in 0..24_u64 {
                let idx = (k * 7919 + 13) % count.max(1);
                let (t, d, p, c, w, h) = bench_frame(&mut session, idx, tier, format, rate);
                total.push(t);
                decode.push(d);
                pixels.push(p);
                convert.push(c);
                dims = (w, h);
            }
            eprintln!(
                "random seek tier 1/{tier} ({}x{}): {:.1} ms/frame (decode wait {:.1}, pixels {:.1}, convert {:.1})",
                dims.0,
                dims.1,
                mean(&total),
                mean(&decode),
                mean(&pixels),
                mean(&convert)
            );
        }
        // Sequential full resolution: back-to-back batches through the pipelined bridge.
        let start = 100_u64.min(count.saturating_sub(1));
        let frames: Vec<u64> = (start..(start + 96).min(count)).collect();
        let t = Instant::now();
        let mut got = 0;
        for part in frames.chunks(PREFETCH_FRAMES as usize) {
            got += bench_batch(&mut session, part, format, rate);
        }
        eprintln!("sequential full (window {READ_AHEAD_WINDOW}, batches of {PREFETCH_FRAMES}): {:.1} fps", got as f64 / t.elapsed().as_secs_f64());
        drop(session);
        // Sequential full resolution through the app path (read-ahead + cache).
        let src = RawCamSource::open(&path, format, bridge, None).unwrap_or_else(|e| panic!("open failed: {e}"));
        let req = |idx: u64| FrameRequest { time: Tick::from_rational(idx as i64, rate.den, rate.num), scale: 1.0 };
        let _ = src.video_frame(req(start.saturating_sub(1)));
        let t = Instant::now();
        for &idx in &frames {
            src.video_frame(req(idx)).unwrap_or_else(|e| panic!("frame {idx}: {e:?}"));
        }
        eprintln!("sequential full (app path, read-ahead): {:.1} fps", frames.len() as f64 / t.elapsed().as_secs_f64());
    }

    /// Live bridge processes started by this test process (by `/proc` parentage and name).
    fn bridge_children() -> Vec<u32> {
        let me = std::process::id().to_string();
        let mut out = Vec::new();
        for e in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
            let Some(pid) = e.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else { continue };
            let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
            // `pid (comm) state ppid …`
            let Some((comm, rest)) = stat.split_once(") ").map(|(a, b)| (a.rsplit('(').next().unwrap_or(""), b)) else { continue };
            let ppid = rest.split_whitespace().nth(1).unwrap_or("");
            if ppid == me && comm.ends_with("-bridge") && !rest.starts_with('Z') {
                out.push(pid);
            }
        }
        out.sort_unstable();
        out
    }

    /// VRAM (MiB) of each pid per `nvidia-smi`, and the GPU's free MiB.
    fn gpu_mem(pids: &[u32]) -> (Vec<(u32, u64)>, u64) {
        let run = |args: &[&str]| {
            std::process::Command::new("nvidia-smi").args(args).output().map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default()
        };
        let apps = run(&["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]);
        let used = |pid: u32| {
            apps.lines().filter_map(|l| l.split_once(',')).find(|(p, _)| p.trim() == pid.to_string()).and_then(|(_, m)| m.trim().parse().ok()).unwrap_or(0)
        };
        let free = run(&["--query-gpu=memory.free", "--format=csv,noheader,nounits"]).trim().parse().unwrap_or(0);
        (pids.iter().map(|&p| (p, used(p))).collect(), free)
    }

    /// VRAM scenarios on real material (`cargo test --release -p filmcraft-codecs
    /// real_clips_vram_scenarios -- --ignored --nocapture`): `FILMCRAFT_RAWCAM_REAL_LIST` names
    /// clips separated by `;` — each opened as its own source (a media item; timeline clips of one
    /// item share its source), played for a moment at full resolution and scrubbed at quarter.
    /// Reports bridge processes and their VRAM, then again after the idle retirement. Refuses to
    /// start with less than 3 GB of VRAM free.
    #[test]
    #[ignore = "needs real camera files (FILMCRAFT_RAWCAM_REAL_LIST) and the vendor bridges"]
    fn real_clips_vram_scenarios() {
        let list = std::env::var("FILMCRAFT_RAWCAM_REAL_LIST").unwrap_or_else(|_| panic!("set FILMCRAFT_RAWCAM_REAL_LIST"));
        let (_, free) = gpu_mem(&[]);
        assert!(free >= 3072, "only {free} MiB VRAM free: not starting");
        let mut sources = Vec::new();
        for path in list.split(';').filter(|p| !p.is_empty()).map(PathBuf::from) {
            let format = Format::from_path(&path).unwrap_or_else(|| panic!("{} is not braw/r3d", path.display()));
            let bridge = bridge_path(format).unwrap_or_else(|| panic!("no {} on PATH", format.bridge_name()));
            sources.push(RawCamSource::open(&path, format, bridge, None).unwrap_or_else(|e| panic!("open failed: {e}")));
        }
        let t = Instant::now();
        for (k, src) in sources.iter().enumerate() {
            let rate = src.info().video.as_ref().map_or(FrameRate::FPS_25, |v| v.frame_rate);
            let req = |idx: u64, scale: f32| FrameRequest { time: Tick::from_rational(idx as i64, rate.den, rate.num), scale };
            let base = 50 + 10 * k as u64;
            for i in 0..12 {
                src.video_frame(req(base + i, 1.0)).unwrap_or_else(|e| panic!("frame: {e:?}"));
            }
            for i in 0..3 {
                src.video_frame(req(base + 40 * i, 0.2)).unwrap_or_else(|e| panic!("frame: {e:?}"));
            }
            let (_, free) = gpu_mem(&[]);
            assert!(free >= 3072, "only {free} MiB VRAM free: stopping");
        }
        std::thread::sleep(Duration::from_millis(500));
        let pids = bridge_children();
        let (mem, free) = gpu_mem(&pids);
        let total: u64 = mem.iter().map(|m| m.1).sum();
        eprintln!("{} sources, {} bridge processes, {total} MiB VRAM in all ({free} MiB free), {:?}: {mem:?}", sources.len(), pids.len(), t.elapsed());
        std::thread::sleep(Duration::from_secs(7));
        let pids = bridge_children();
        let (mem, _) = gpu_mem(&pids);
        eprintln!("after 7 s idle (bridge trim): {} bridge processes, {} MiB: {mem:?}", pids.len(), mem.iter().map(|m| m.1).sum::<u64>());
        std::thread::sleep(IDLE_SESSIONS_AFTER.saturating_sub(Duration::from_secs(4)));
        let pids = bridge_children();
        let (mem, _) = gpu_mem(&pids);
        eprintln!(
            "after {:?} idle: {} bridge processes, {} MiB: {mem:?}",
            IDLE_SESSIONS_AFTER + Duration::from_secs(3),
            pids.len(),
            mem.iter().map(|m| m.1).sum::<u64>()
        );
        drop(sources);
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(bridge_children(), Vec::<u32>::new(), "every bridge is gone with its sources");
    }

    /// Bench helpers: the source's own session pool (one session for every tier on a protocol 2
    /// bridge, one per tier otherwise).
    struct BenchSessions {
        src: Arc<RawCamSource>,
    }

    fn bench_session(bridge: &Path, path: &Path, format: Format) -> BenchSessions {
        let src = RawCamSource::open(path, format, bridge.to_path_buf(), None).unwrap_or_else(|e| panic!("open failed: {e}"));
        BenchSessions { src }
    }

    /// Run `f` on a session serving `tier` (with its scale word), then pool it again.
    fn bench_with<R>(s: &mut BenchSessions, tier: u32, f: impl FnOnce(&mut Session, Option<&'static str>) -> R) -> R {
        let pooled = s.src.video.lock().unwrap_or_else(|e| e.into_inner()).take_session(tier);
        let (key, mut session) = s.src.open_session(pooled, tier).unwrap_or_else(|e| panic!("spawn: {e}"));
        let r = f(&mut session, s.src.scale_word(key, tier));
        s.src.video.lock().unwrap_or_else(|e| e.into_inner()).put_back(key, session);
        r
    }

    fn bench_frame(s: &mut BenchSessions, idx: u64, tier: u32, format: Format, rate: FrameRate) -> (f64, f64, f64, f64, u32, u32) {
        let _ = format;
        bench_with(s, tier, |session, scale| {
            let t = Instant::now();
            let reply = session.request_frame(idx, scale, FRAME_TIMEOUT).unwrap_or_else(|e| panic!("frame {idx}: {e}"));
            let reply = reply.unwrap_or_else(|e| panic!("frame {idx}: {e}"));
            let (d, p) = session.last_timing();
            let c = Instant::now();
            let f = RawCamSource::into_frame(reply, idx, rate);
            let ms = |d: Duration| d.as_secs_f64() * 1000.0;
            (ms(t.elapsed()), ms(d), ms(p), ms(c.elapsed()), f.width, f.height)
        })
    }

    fn bench_batch(s: &mut BenchSessions, part: &[u64], format: Format, rate: FrameRate) -> usize {
        let _ = format;
        bench_with(s, 1, |session, scale| {
            let replies = session.request_batch_steered(part, scale, READ_AHEAD_WINDOW, || true, FRAME_TIMEOUT).unwrap_or_else(|e| panic!("batch: {e}"));
            replies.into_iter().zip(part).filter_map(|(r, &i)| r.ok().map(|r| RawCamSource::into_frame(r, i, rate))).count()
        })
    }

    /// `n` clip files in a fresh test dir (contents pick the fake bridge's behaviour).
    #[cfg(target_os = "linux")]
    fn clips(name: &str, contents: &[&str]) -> (PathBuf, Vec<PathBuf>) {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let paths = contents
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let p = dir.join(format!("clip{k}.braw"));
                std::fs::write(&p, c).ok();
                p
            })
            .collect();
        (dir, paths)
    }

    #[cfg(target_os = "linux")]
    fn open_quiet(clip: &Path, bridge: &Path, dir: &Path) -> Arc<RawCamSource> {
        let src = RawCamSource::open(clip, Format::Braw, bridge.to_path_buf(), Some(dir.join("cache"))).unwrap_or_else(|e| panic!("clip must open: {e}"));
        src.prefetch_started.store(true, Ordering::Relaxed); // no read-ahead: deterministic traffic
        src
    }

    #[cfg(target_os = "linux")]
    fn lines(path: &Path) -> Vec<String> {
        std::fs::read_to_string(path).unwrap_or_default().lines().map(str::to_string).collect()
    }

    /// Wait (at most 3 s) until `bridge` runs `n` persistent processes.
    #[cfg(target_os = "linux")]
    fn settle(bridge: &Path, n: usize) -> usize {
        let t = Instant::now();
        while bridge::live_processes(bridge) != n && t.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(10));
        }
        bridge::live_processes(bridge)
    }

    /// Three clips of a multi-clip bridge, decoded at once from three threads: ONE bridge process
    /// serves them all (each clip a handle on it), every frame holds its own pixels, and the
    /// process exits with the last source.
    #[cfg(target_os = "linux")]
    #[test]
    fn one_shared_process_serves_every_clip() {
        let (dir, paths) = clips("shared", &["placeholder"; 3]);
        let bridge = fake_bridge_v3(&dir);
        let sources: Vec<_> = paths.iter().map(|p| open_quiet(p, &bridge, &dir)).collect();
        let workers: Vec<_> = sources
            .iter()
            .cloned()
            .map(|src| {
                std::thread::spawn(move || {
                    (0..4_i64)
                        .map(|k| {
                            let f = src.video_frame(FrameRequest { time: Tick::from_rational(k, 1, 24), scale: if k == 3 { 0.3 } else { 1.0 } });
                            f.map(|f| {
                                (
                                    f.pts,
                                    f.width,
                                    match &f.data {
                                        filmcraft_frame::PixelData::Rgba8(d) => d.first().copied(),
                                        _ => None,
                                    },
                                )
                            })
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for w in workers {
            let got = w.join().unwrap_or_else(|_| panic!("worker"));
            for (k, f) in got.into_iter().enumerate() {
                let (pts, width, px) = f.unwrap_or_else(|e| panic!("frame {k}: {e:?}"));
                assert_eq!(pts, Tick::from_rational(k as i64, 1, 24));
                assert_eq!(px, Some(k as u8 + 1), "frame {k} holds its own pixels");
                assert_eq!(width, if k == 3 { 4 } else { 16 }, "the tier goes with each command");
            }
        }
        assert_eq!(lines(&bridge.with_extension("multi")).len(), 1, "one process for three clips");
        assert_eq!(bridge::live_processes(&bridge), 1);
        let cmds = lines(&bridge.with_extension("cmds"));
        assert_eq!(cmds.iter().filter(|l| l.starts_with("open ")).count(), 3, "{cmds:?}");
        assert!(cmds.iter().filter(|l| l.starts_with("frame ")).all(|l| l.split(' ').count() == 4), "frame <handle> <index> <scale>: {cmds:?}");
        drop(sources);
        assert_eq!(settle(&bridge, 0), 0, "the shared process exits with its last clip");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A clip the shared bridge cannot open fails alone; the clips beside it keep their process.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_clip_the_shared_bridge_refuses_fails_alone() {
        let (dir, paths) = clips("refused", &["placeholder", "nope", "placeholder"]);
        let bridge = fake_bridge_v3(&dir);
        let sources: Vec<_> = paths.iter().map(|p| open_quiet(p, &bridge, &dir)).collect();
        let at = |k: i64| FrameRequest { time: Tick::from_rational(k, 1, 24), scale: 1.0 };
        for round in 0..2 {
            assert!(sources[0].video_frame(at(round)).is_ok());
            assert!(matches!(sources[1].video_frame(at(round)), Err(MediaError::Decode(_))));
            assert!(sources[2].video_frame(at(round)).is_ok());
        }
        assert_eq!(lines(&bridge.with_extension("multi")).len(), 1, "never restarted");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A clip that crashes the shared bridge: the bridge is restarted at once and the other clip
    /// re-opens its handle on the new one; after the clip's second crash it is isolated in a
    /// process of its own, and its next crashes no longer touch the shared bridge.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_crashing_clip_is_isolated_from_the_shared_bridge() {
        let (dir, paths) = clips("crash", &["placeholder", "boom"]);
        let bridge = fake_bridge_v3(&dir);
        let good = open_quiet(&paths[0], &bridge, &dir);
        let bad = open_quiet(&paths[1], &bridge, &dir);
        let at = |k: i64| FrameRequest { time: Tick::from_rational(k, 1, 24), scale: 1.0 };
        let multis = || lines(&bridge.with_extension("multi")).len();
        assert!(good.video_frame(at(0)).is_ok());
        assert!(bad.video_frame(at(0)).is_err(), "the crash fails the clip's request");
        assert!(good.video_frame(at(1)).is_ok(), "the next request restarts the bridge and re-opens");
        assert_eq!(multis(), 2);
        std::thread::sleep(RESPAWN_BACKOFF * 2); // the bad clip's own back-off
        assert!(bad.video_frame(at(1)).is_err(), "second crash");
        assert!(good.video_frame(at(2)).is_ok());
        assert_eq!(multis(), 3);
        std::thread::sleep(RESPAWN_BACKOFF * 3);
        assert!(bad.video_frame(at(2)).is_err(), "isolated: it crashes its own process now");
        for k in 3..6 {
            assert!(good.video_frame(at(k)).is_ok(), "frame {k}");
        }
        assert_eq!(multis(), 3, "the shared bridge survived the isolated crash");
        let own = lines(&paths[1].with_extension("braw.sessions"));
        assert_eq!(own.len(), 1, "one single-clip process for the isolated clip: {own:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A shared bridge that does not start (`.nostart`): the clip gets a process of its own and
    /// stays with that mode.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_shared_bridge_that_does_not_start_falls_back_to_one_process_per_clip() {
        let (dir, paths) = clips("nostart", &["placeholder"]);
        let bridge = fake_bridge_v3(&dir);
        std::fs::write(bridge.with_extension("nostart"), "").ok();
        let src = open_quiet(&paths[0], &bridge, &dir);
        let f = src.video_frame(FrameRequest { time: Tick::from_rational(2, 1, 24), scale: 1.0 }).unwrap_or_else(|e| panic!("frame: {e:?}"));
        assert_eq!(f.pts, Tick::from_rational(2, 1, 24));
        assert!(!src.multi.load(Ordering::Relaxed));
        assert_eq!(lines(&paths[0].with_extension("braw.sessions")).len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Bridges without the multi-clip mode need a process per clip: at most the cap run at
    /// once, the least recently used idle one is retired to make room for another clip.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_process_cap_retires_the_least_recently_used_idle_process() {
        let cap = max_processes();
        let (dir, paths) = clips("cap", &vec!["placeholder"; cap + 1]);
        let bridge = fake_bridge_v2(&dir);
        let sources: Vec<_> = paths.iter().map(|p| open_quiet(p, &bridge, &dir)).collect();
        for (k, src) in sources.iter().enumerate() {
            assert!(src.video_frame(FrameRequest { time: Tick::ZERO, scale: 1.0 }).is_ok(), "clip {k}");
            assert!(bridge::live_processes(&bridge) <= cap, "clip {k}: {} processes", bridge::live_processes(&bridge));
            std::thread::sleep(Duration::from_millis(5)); // distinct activity times
        }
        assert_eq!(bridge::live_processes(&bridge), cap);
        let pooled = |s: &RawCamSource| s.video.lock().unwrap_or_else(|e| e.into_inner()).sessions.len();
        assert_eq!(pooled(&sources[0]), 0, "the oldest idle process made room");
        assert!(sources[1..].iter().all(|s| pooled(s) == 1));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Idle sessions are retired whatever used them last: a source only ever asked for
    /// thumbnails (background requests start no read-ahead) kept its bridge process forever.
    #[cfg(target_os = "linux")]
    #[test]
    fn idle_sessions_of_background_only_sources_are_retired() {
        let (dir, paths) = clips("idle", &["placeholder"]);
        let bridge = fake_bridge_v2(&dir);
        let src = open_quiet(&paths[0], &bridge, &dir);
        src.prefetch_started.store(false, Ordering::Relaxed);
        let thumb = filmcraft_media::cancel::with_background(true, || src.video_frame(FrameRequest { time: Tick::ZERO, scale: 0.1 }));
        assert!(thumb.is_ok());
        assert!(!src.prefetch_started.load(Ordering::Relaxed), "no read-ahead (it used to be the only retirer)");
        assert_eq!(bridge::live_processes(&bridge), 1);
        assert!(!src.retire_idle(IDLE_SESSIONS_AFTER, false), "not idle yet");
        assert!(live_sources().iter().any(|s| Arc::ptr_eq(s, &src)), "the janitor sees it");
        assert!(src.retire_idle(Duration::ZERO, false));
        assert_eq!(bridge::live_processes(&bridge), 0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
