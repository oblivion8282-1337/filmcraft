//! Camera-RAW media: Blackmagic `.braw` and RED `.R3D` clips decoded by external bridge binaries
//! (`braw-bridge` / `r3d-bridge`) that wrap the vendor SDKs and stream debayered `rgb24` frames
//! (see [`bridge`]). The SDKs are licensed and never ship with FilmCraft, so the bridges are
//! user-provided: [`path_opener`] finds them via `FILMCRAFT_BRAW_BRIDGE` / `FILMCRAFT_R3D_BRIDGE`
//! or on `PATH`, and a missing bridge yields a clear "unsupported" error naming the variable.
//!
//! Frames are full-range sRGB RGB (the SDKs' processed output); audio is extracted once per file
//! into a cache WAV (`FILMCRAFT_RAWCAM_CACHE`, default `$XDG_CACHE_HOME/filmcraft/rawcam`) and
//! served through [`filmcraft_media::wav::WavSource`]. Decoding runs on persistent bridge
//! sessions (one per debayer tier — [`FrameRequest::scale`] picks the tier, so thumbnails
//! debayer at quarter/eighth resolution), and a read-ahead thread decodes [`PREFETCH_FRAMES`]
//! ahead of playback so showing a frame is normally a cache hit.
//!
//! Scrubbing a paused monitor ([`filmcraft_media::cancel::with_scrub`]) is served at the
//! [`SCRUB_TIER`] while the seeks come in a burst: a full-resolution 6K debayer takes ~50 ms, a
//! quarter one ~10 ms. Those frames are flagged provisional
//! ([`filmcraft_media::pending::mark_provisional`]); the monitor asks again without the hint, and
//! that request waits (cancellably) until the playhead has rested for [`SCRUB_REST`] before it
//! decodes the exact tier — so the refinement never holds the decoder while the drag goes on.

pub mod bridge;

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

use crate::rawcam::bridge::DecodeSession;

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
/// Live decode sessions kept at once (one per debayer tier in use), most recently used last.
const TIERS: usize = 2;
/// How far a contended request may reach into the cache for a frame to show while it waits.
const NEAREST_FRAME_TOLERANCE: u64 = 12;
/// Frames the read-ahead thread decodes past the playhead (≈ 250 ms at 24 fps).
const PREFETCH_FRAMES: u64 = 6;
/// A quiet spell this long retires the bridge processes (hundreds of MB of SDK state each).
const IDLE_SESSIONS_AFTER: Duration = Duration::from_secs(20);
/// Paused-monitor seeks this close together are a scrub burst; the burst ends (the playhead
/// rests) after this long without a move.
const SCRUB_REST: Duration = Duration::from_millis(150);
/// The debayer tier a scrub burst is served at (quarter resolution: ~5× cheaper than full).
const SCRUB_TIER: u32 = 4;
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
            (Format::R3d, 8) => "eighth",
            _ => "quarter",
        }
    }

    /// Resolution factor for a scale request: full/half/quarter (BRAW), down to eighth (R3D).
    /// `draft` steps down one tier: playback may debayer faster than the paused, exact view.
    fn tier_factor(self, scale: f32, draft: bool) -> u32 {
        let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        let (factor, smaller) = match self {
            Format::Braw => {
                if scale >= 0.7 {
                    (1, 2)
                } else if scale >= 0.35 {
                    (2, 4)
                } else {
                    (4, 4)
                }
            }
            Format::R3d => {
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
    fn stepped_down(self, tier: u32, steps: u32) -> u32 {
        let floor = match self {
            Format::Braw => 4,
            Format::R3d => 8,
        };
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

/// One live bridge run, at a single debayer tier.
struct SessionSlot {
    tier: u32,
    session: DecodeSession,
}

/// Idle decode sessions, at most [`TIERS`] (one per debayer tier in use — playback at full
/// resolution, thumbnails at quarter — so the two never restart each other), most recently used
/// last. While a request decodes, it owns its session *outside* this lock with [`VideoState::busy`]
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

    /// The pooled session at `tier`, if its bridge still runs (a dead one is dropped: it crashed
    /// while idle, and the request must not inherit that).
    fn take_session(&mut self, tier: u32) -> Option<DecodeSession> {
        let at = self.sessions.iter().position(|s| s.tier == tier)?;
        let mut session = self.sessions.remove(at).session;
        session.is_alive().then_some(session)
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
    fn put_back(&mut self, tier: u32, session: DecodeSession) {
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

    fn release(mut self, outcome: Option<(u32, DecodeSession)>, ok: bool) {
        let mut state = self.video.lock().unwrap_or_else(|e| e.into_inner());
        state.busy = false;
        if ok {
            state.note_success();
        } else {
            state.note_failure(Instant::now());
        }
        if let Some((tier, session)) = outcome {
            state.put_back(tier, session);
        }
        self.held = false;
    }

    /// The session decoded: back into the pool.
    fn succeed(self, tier: u32, session: DecodeSession) {
        self.release(Some((tier, session)), true);
    }

    /// The bridge failed (the session, if any, was dropped by the caller — outside the lock).
    fn fail(self) {
        self.release(None, false);
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
    /// any wait, and stops a running one at its next chunk.
    waiting: AtomicU32,
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
        });
        let _ = src.self_ref.set(Arc::downgrade(&src));
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

    /// `pooled`, or a freshly started bridge session at `tier` (called without the lock held:
    /// the SDK start-up can take seconds).
    fn open_session(&self, pooled: Option<DecodeSession>, tier: u32) -> Result<DecodeSession, MediaError> {
        match pooled {
            Some(s) => Ok(s),
            None => DecodeSession::spawn(&self.bridge, &self.path, self.format.tier_word(tier), PROBE_TIMEOUT)
                .map_err(|e| MediaError::Decode(format!("{}: cannot start {}: {e}", self.path.display(), self.format.bridge_name()))),
        }
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
                let idle_for = this.activity.lock().unwrap_or_else(|e| e.into_inner()).elapsed();
                if idle_for > IDLE_SESSIONS_AFTER {
                    this.video.lock().unwrap_or_else(|e| e.into_inner()).sessions.clear();
                }
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
        let mut session = match self.open_session(pooled, tier) {
            Ok(s) => s,
            Err(e) => {
                lease.fail();
                return Err(e);
            }
        };

        // Small chunks: an interactive request waits at most one chunk behind this read-ahead,
        // not a whole batch (the pipeline still fills: the bridge keeps decoding while it
        // receives). Full and half tiers take one frame per chunk — a 6K full debayer alone is
        // ~50 ms; the cheap tiers two.
        let chunk = if tier <= 2 { 1 } else { 2 };
        let start_demand = self.demand.load(Ordering::Relaxed);
        let outcome = session.request_batch_steered(
            indices,
            chunk,
            || self.demand.load(Ordering::Relaxed) == start_demand && self.waiting.load(Ordering::Relaxed) == 0,
            FRAME_TIMEOUT,
        );
        *self.activity.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
        if outcome.is_ok() {
            lease.succeed(tier, session);
        } else {
            drop(session); // kills and reaps the bridge, outside the lock
            lease.fail();
        }

        let frames = outcome.map_err(|e| {
            log::warn!("{}: read-ahead batch failed ({e}) — restarting the bridge", self.path.display());
            MediaError::Decode(format!("{}: {e}", self.path.display()))
        })?;
        for (idx, reply) in indices.iter().zip(frames) {
            let frame = Arc::new(Self::into_frame(reply, *idx, rate));
            self.cache_frame((*idx, tier), frame);
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

    /// Store a decoded frame in the cache, trimmed to a window around the current position.
    fn cache_frame(&self, key: (u64, u32), frame: Arc<VideoFrame>) {
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.insert(key, frame);
        if cache.len() > CACHE_FRAMES {
            let farthest = cache.keys().copied().max_by_key(|k| k.0.abs_diff(key.0)).filter(|k| *k != key).unwrap_or(key);
            cache.remove(&farthest);
        }
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
            let tier = self.format.stepped_down(tier, steps);
            let burst = seek && self.scrub.lock().unwrap_or_else(|e| e.into_inner()).observe(Instant::now(), idx);
            (tier, if burst { tier.max(SCRUB_TIER) } else { tier }, burst)
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
            // lock is free, so the bookkeeping above stays responsive. A failed request
            // desynchronises the byte stream — the session is then dropped (killing its bridge),
            // and the next request starts a fresh one.
            let mut session = match self.open_session(pooled, serve_tier) {
                Ok(s) => s,
                Err(e) => {
                    lease.fail();
                    return Err(e);
                }
            };

            let decode_started = Instant::now();
            let outcome = session.request_frame(idx, FRAME_TIMEOUT);
            let decoded_in = decode_started.elapsed();
            *self.activity.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
            if outcome.is_ok() {
                lease.succeed(serve_tier, session);
            } else {
                drop(session); // kills and reaps the bridge, outside the lock
                lease.fail();
            }

            if stats {
                eprintln!(
                    "rawcam req idx={idx} tier={serve_tier} want={want:.3} seek={seek} decoded wait={waited:?} decode={decoded_in:?} total={:?}",
                    started.elapsed()
                );
            }

            let reply = outcome.map_err(|e| {
                log::warn!("{}: frame {idx} failed ({e}) — restarting the bridge", self.path.display());
                MediaError::Decode(format!("{}: {e}", self.path.display()))
            })?;
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
        let tier = self.format.tier_factor(req.scale, filmcraft_media::cancel::draft());
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
        assert_eq!(Format::Braw.tier_factor(1.0, false), 1);
        assert_eq!(Format::Braw.tier_factor(0.7, false), 1);
        assert_eq!(Format::Braw.tier_factor(0.5, false), 2);
        assert_eq!(Format::Braw.tier_factor(0.1, false), 4);
        assert_eq!(Format::R3d.tier_factor(0.1, false), 8);
        assert_eq!(Format::Braw.tier_factor(f32::NAN, false), 1);
        // Draft playback steps one tier down; the floor stays the smallest tier.
        assert_eq!(Format::Braw.tier_factor(1.0, true), 2);
        assert_eq!(Format::R3d.tier_factor(1.0, true), 2);
        assert_eq!(Format::Braw.tier_factor(0.1, true), 4);
        assert_eq!(Format::Braw.tier_word(1), "full");
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
        assert_eq!(Format::Braw.stepped_down(1, 1), 2);
        assert_eq!(Format::Braw.stepped_down(1, 2), 4);
        assert_eq!(Format::Braw.stepped_down(4, 2), 4, "BRAW floors at quarter");
        assert_eq!(Format::R3d.stepped_down(2, 2), 8);
        assert_eq!(Format::R3d.stepped_down(4, 9), 8, "absurd step counts stay at the floor");
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
echo "{\"type\":\"metadata\",\"timecode\":\"01:02:03:04\",\"fps_num\":24,\"fps_den\":1,\"width\":16,\"height\":8,\"frame_count\":$frames}" >&2
[ "$probe" = 1 ] && { echo '{"type":"done"}' >&2; exit 0; }
if [ -n "$audio" ]; then
    printf 'RIFF$\000\000\000WAVEfmt \020\000\000\000\001\000\001\000\200\273\000\000\000\167\001\000\002\000\020\000data\004\000\000\000\000\000\000\000' > "$audio"
    echo '{"type":"done"}' >&2
    exit 0
fi
# Every decode session appends its PID: the tests assert that seeks reuse the process.
echo $$ >> "$input.sessions" 2>/dev/null || :
bytes=$((w * h * 4))
emit_frame() {
    i=$1
    b=$(printf "\\$(printf '%03o' $((i + 1)))")
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
            frame\ *) emit_frame "${line#frame }";;
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
        let path = dir.join("fake-bridge");
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
        let mut state = src.video.lock().unwrap_or_else(|e| e.into_inner());
        let mut session = match state.sessions.iter().position(|s| s.tier == 2) {
            Some(at) => state.sessions.remove(at).session,
            None => DecodeSession::spawn(&src.bridge, &src.path, src.format.tier_word(2), PROBE_TIMEOUT).unwrap_or_else(|e| panic!("spawn: {e}")),
        };
        drop(state);
        let t0 = std::time::Instant::now();
        let one = session.request_frame(30, FRAME_TIMEOUT).unwrap_or_else(|e| panic!("single request failed: {e}"));
        eprintln!("single frame 30: {:?} ({}×{}, {:?})", t0.elapsed(), one.width, one.height, one.format);
        let t1 = std::time::Instant::now();
        let batch = session.request_batch(&[31, 32, 33, 34, 35, 36], Duration::from_secs(10));
        eprintln!("batch 31..36: {:?} ({} frames)", t1.elapsed(), batch.as_ref().map_or(0, |b| b.len()));
        let _ = batch.unwrap_or_else(|e| panic!("batch after single failed: {e}"));
    }
}
