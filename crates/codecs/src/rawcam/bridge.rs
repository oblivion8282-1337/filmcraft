//! The camera-RAW bridge subprocess protocol: `braw-bridge` / `r3d-bridge` are external helper
//! binaries that wrap the vendor SDKs (Blackmagic RAW, RED R3D) and stream debayered `rgb24`
//! frames on stdout with NDJSON events on stderr. The SDKs are licensed, so the bridges are
//! user-provided: [`crate::rawcam`] looks them up via `FILMCRAFT_BRAW_BRIDGE` /
//! `FILMCRAFT_R3D_BRIDGE` or on `PATH`.
//!
//! One-shot invocations (`--probe-only`, `--extract-audio`) answer a single question. Decoding
//! uses the bridges' `--persistent` command mode instead: the process stays alive, the SDK is
//! initialised exactly once, and each `frame <index>` line on stdin is answered by a `frame`
//! event — the dimensions about to arrive, so the byte count is never guessed — followed by the
//! frame's pixels and a `verify` event with the dimensions actually produced. A mismatch
//! desynchronises the byte stream, so [`DecodeSession::request_frame`] reports it and the caller
//! discards the session.
//!
//! Protocol 2 bridges say so in their `metadata` event (`"protocol":2,"caps":"…"`, see
//! [`Caps`]): `frame <index> <scale>` picks the debayer tier per command (one session serves
//! every tier), and a frame that cannot be produced is answered by
//! `{"type":"error","index":N,…}` in its place in the reply order — that one request fails, the
//! session goes on ([`Reply`]). Bridges without the fields are spoken to as before.
//!
//! Shared-memory transport (`shm` capability, Linux): instead of pushing ~85 MB per 6K frame
//! through the stdout pipe, the bridge writes frames into a ring of slots over a tmpfs file the
//! client created ([`ShmRing`]). The client creates the file in `/dev/shm`, unlinks it at once
//! (nothing is left behind, whatever happens later) and passes it as `--shm /proc/<pid>/fd/<n>`;
//! the bridge's `metadata` answers `"shm_slots":N,"shm_slot_bytes":S` when it took it. A frame
//! in slot k is announced with `"slot":k` (no stdout bytes), read with one positional read at
//! `k*S`, and handed back with `release k`; frames without `slot` still arrive on stdout. The
//! layout and every announced slot are validated before anything is read or allocated.
//!
//! Multi-clip transport (`multi` capability, protocol 3): one bridge process — one CUDA context,
//! one SDK instance and its GPU pools — serves every clip of a format ([`MultiBridge`]). It is
//! started without `--input` (`--persistent --multi`), announces itself with a `ready` event, and
//! clips are opened under client-chosen handles (`open <h> <path>`, answered by a `metadata` event
//! naming the handle). `frame <h> <index> <scale>` / `close <h>` address one clip; every event names
//! its handle, replies of one handle keep their command order and replies of different handles
//! interleave. A reader thread demultiplexes the event stream to the waiting requests of each
//! handle ([`ClipSession`]); one clip's per-frame errors or close never disturb another.
//!
//! GPU frames (`gpu` capability, Linux, ADR 0002): a multi-clip bridge started with
//! `--gpu-export <socket>` connects to a Unix socket FilmCraft listens on, and `frame <h> <i>
//! <scale> gpu` is answered with a frame in exportable GPU memory instead of pixels: the event
//! names the buffer and its row pitch (`"gpu":B,"pitch":P`). Each buffer's file descriptor comes
//! over the socket once (`SCM_RIGHTS`, payload: id and size), before the first frame naming it;
//! the client owns a buffer until it sends `release-gpu <B>`, and `{"type":"gpu-freed","gpu":B}`
//! says the bridge let go of one ([`GpuBuffers`]). The frames become
//! [`filmcraft_frame::PixelData::External`] pictures the GPU compositor imports.
//!
//! Everything a bridge says is untrusted: dimensions are capped ([`MAX_FRAME_DIM`],
//! [`MAX_FRAME_BYTES`]) before anything is allocated, pixel reads run on a reader thread so a
//! stalled bridge times out instead of hanging its caller, and every child process is killed
//! and reaped on every exit path.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak, mpsc};
use std::time::{Duration, Instant};

/// Largest frame side a bridge may announce (16K; the largest camera sensors are 12K).
pub const MAX_FRAME_DIM: u32 = 16384;
/// Largest frame a bridge may send, in bytes (a 12K RGBA frame is ~320 MB).
pub const MAX_FRAME_BYTES: usize = 512 << 20;
/// Most shared-memory slots a bridge may lay over the ring file.
pub const MAX_SHM_SLOTS: u64 = 16;
/// Largest shared-memory ring a bridge may claim, in bytes.
pub const MAX_SHM_BYTES: u64 = 4 << 30;
/// Most frame bytes requested from all bridges together and not yet received: read-ahead stops
/// sending once it would pass this (an interactive request always goes).
pub const MAX_INFLIGHT_BYTES: u64 = 1 << 30;
/// Longest path a multi-clip `open` command carries (the bridges cap a command line at 64 KiB).
const MAX_OPEN_PATH: usize = 16 << 10;

/// What a bridge's persistent mode supports, from the `caps` list of its `metadata` event (all
/// off for a bridge that predates the field).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caps {
    /// `frame <index> <scale>`: the debayer tier per command.
    pub scale: bool,
    /// `{"type":"error","index":N}` answers one request; the session stays usable.
    pub frame_errors: bool,
    /// `--shm <path>`: frames through a shared-memory ring instead of stdout.
    pub shm: bool,
    /// `--persistent --multi`: one process serves many clips under handles (protocol 3).
    pub multi: bool,
    /// `--gpu-export <socket>` is understood (probe `caps`): offer the GPU frame socket.
    pub gpu_export: bool,
    /// Frames in exportable GPU memory are on (the `ready` event of a bridge given the socket).
    pub gpu: bool,
    /// GPU frames can come as half floats (`frame … gpu f16`), the decode's full precision.
    pub f16: bool,
    /// Host frames can come in 16 bits (`frame … rgba16`).
    pub rgba16: bool,
}

impl Caps {
    fn parse(list: Option<&str>) -> Self {
        let mut caps = Caps::default();
        for word in list.unwrap_or_default().split(',').map(str::trim) {
            match word {
                "scale" => caps.scale = true,
                "frame-errors" => caps.frame_errors = true,
                "shm" => caps.shm = true,
                "multi" => caps.multi = true,
                "gpu" => caps.gpu = true,
                "gpu-export" => caps.gpu_export = true,
                "f16" => caps.f16 = true,
                "rgba16" => caps.rgba16 = true,
                _ => {}
            }
        }
        caps
    }
}

/// Clip facts from a bridge's `metadata` event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metadata {
    pub timecode: String,
    pub fps_num: u32,
    pub fps_den: u32,
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
    /// Protocol version (1 when absent).
    pub protocol: u32,
    pub caps: Caps,
    /// The shared-memory ring the bridge laid out (`shm_slots`, `shm_slot_bytes`): only in a
    /// persistent session started with `--shm` that the bridge could use. Unvalidated.
    pub shm: Option<(u64, u64)>,
    /// The colour space every frame comes in (`color`: a `filmcraft_color::ColorSpace` id, e.g.
    /// `bmdfilm5-bmdwg`); `None` when the bridge does not say (older bridges: display RGB).
    pub color: Option<String>,
}

/// The pixel layout of the bytes a bridge sends after a `frame` event (`format` field; absent
/// means the older `rgb24`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameFormat {
    /// 3 bytes per pixel, R G B.
    Rgb24,
    /// 4 bytes per pixel, R G B A — used as-is for [`filmcraft_frame::PixelData::Rgba8`].
    Rgba8,
    /// 4 bytes per pixel, B G R A — one in-place channel swap away from RGBA.
    Bgra8,
    /// 8 bytes per pixel: half floats R G B A, encoded values, straight alpha (GPU frames only).
    RgbaF16,
    /// 8 bytes per pixel: little-endian u16 R G B A, encoded values, straight alpha.
    Rgba16,
}

impl FrameFormat {
    fn parse(s: Option<&str>) -> Option<Self> {
        match s {
            None => Some(Self::Rgb24),
            Some("rgb24") => Some(Self::Rgb24),
            Some("rgba8") => Some(Self::Rgba8),
            Some("bgra8") => Some(Self::Bgra8),
            Some("rgbaf16") => Some(Self::RgbaF16),
            Some("rgba16") => Some(Self::Rgba16),
            _ => None,
        }
    }

    /// Bytes one pixel takes in this layout.
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb24 => 3,
            Self::Rgba8 | Self::Bgra8 => 4,
            Self::RgbaF16 | Self::Rgba16 => 8,
        }
    }
}

/// The byte count of a `width`×`height` frame in `format`, or why a bridge announcing it is not
/// believed (empty, a side over [`MAX_FRAME_DIM`], over [`MAX_FRAME_BYTES`]).
pub fn frame_bytes(width: u32, height: u32, format: FrameFormat) -> Result<usize, String> {
    if width == 0 || height == 0 {
        return Err(format!("bridge announced an empty {width}×{height} frame"));
    }
    if width > MAX_FRAME_DIM || height > MAX_FRAME_DIM {
        return Err(format!("bridge announced a {width}×{height} frame (larger than {MAX_FRAME_DIM} px)"));
    }
    let bytes = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|px| px.checked_mul(format.bytes_per_pixel()))
        .filter(|&b| b <= MAX_FRAME_BYTES);
    bytes.ok_or_else(|| format!("bridge announced a {width}×{height} frame (more than {} MB)", MAX_FRAME_BYTES >> 20))
}

/// One reply's pixels as the bridge sent them.
#[derive(Clone, Debug)]
pub struct FrameBytes {
    /// Empty for a frame in GPU memory ([`FrameBytes::gpu`]).
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: FrameFormat,
    /// The frame lives in exported GPU memory instead of `data`.
    pub gpu: Option<GpuFrame>,
}

/// Where a `frame` event says a GPU frame is: the exported buffer's id and the row pitch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuRef {
    pub buffer: u64,
    pub pitch: u32,
}

/// A frame in exported GPU memory: rows of `pitch` bytes from offset 0 of `memory`. The bridge
/// keeps its hands off the buffer until the last clone of `release` is dropped.
#[derive(Clone, Debug)]
pub struct GpuFrame {
    pub memory: Arc<filmcraft_frame::ExternalMemory>,
    pub pitch: u32,
    pub release: Arc<GpuRelease>,
}

/// Hands a GPU buffer back to the bridge (`release-gpu <B>`) when dropped.
pub struct GpuRelease {
    shared: Weak<MultiShared>,
    buffer: u64,
}

impl std::fmt::Debug for GpuRelease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuRelease({})", self.buffer)
    }
}

impl Drop for GpuRelease {
    fn drop(&mut self) {
        if let Some(shared) = self.shared.upgrade() {
            let _ = shared.send_line(&format!("release-gpu {}\n", self.buffer));
        }
    }
}

/// Whether a `width`×`height` frame of `bpp`-byte pixels with rows `pitch` apart fits `size`
/// bytes from offset 0, with the rows aligned as a GPU copy needs (256 bytes).
pub fn gpu_layout_fits(width: u32, height: u32, bpp: u32, pitch: u32, size: u64) -> bool {
    let row = u64::from(width) * u64::from(bpp);
    width > 0 && height > 0 && pitch.is_multiple_of(256) && u64::from(pitch) >= row && u64::from(pitch) * u64::from(height - 1) + row <= size
}

/// Largest GPU buffer a bridge may export (the bridges round frames up to their allocation
/// granularity; a 12K RGBA frame is ~320 MB).
pub const MAX_GPU_BUFFER: u64 = 1 << 30;

/// `"0123…"` (32 hex digits) as a 16-byte UUID.
fn parse_uuid(hex: &str) -> Option<[u8; 16]> {
    let hex = hex.as_bytes();
    if hex.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, b) in out.iter_mut().enumerate() {
        let pair = std::str::from_utf8(hex.get(i * 2..i * 2 + 2)?).ok()?;
        *b = u8::from_str_radix(pair, 16).ok()?;
    }
    Some(out)
}

/// One NDJSON event from a bridge's stderr.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Metadata(Metadata),
    /// A multi-clip bridge is up (`--multi`): its protocol, capabilities and shared-memory ring.
    Ready {
        protocol: u32,
        caps: Caps,
        shm: Option<(u64, u64)>,
        /// The GPU the bridge decodes on (`gpu_uuid`, with the `gpu` capability).
        gpu_uuid: Option<[u8; 16]>,
    },
    /// Before a frame's pixels: the frame the following bytes belong to, its dimensions and
    /// pixel layout.
    Frame {
        index: u64,
        width: u32,
        height: u32,
        format: FrameFormat,
        /// The shared-memory slot holding the pixels (none: they follow on stdout).
        slot: Option<u32>,
        /// The GPU buffer holding the frame (no pixels follow).
        gpu: Option<GpuRef>,
    },
    /// The bridge freed a GPU buffer it had exported (it will not name it again).
    GpuFreed {
        buffer: u64,
    },
    /// One requested frame could not be produced (protocol 2): it takes that request's place in
    /// the reply order, no pixels follow, the session goes on.
    FrameError {
        index: u64,
        message: String,
    },
    /// After a frame's pixels: the dimensions actually produced.
    Verify {
        index: u64,
        width: u32,
        height: u32,
    },
    Progress {
        frame: u64,
        total: u64,
    },
    Done,
    Error(String),
}

/// The line after a `"key":` separator, or nothing when the key is absent.
fn after_key<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let at = line.find(&format!("\"{key}\":"))? + key.len() + 3;
    line.get(at..)
}

/// `"key":"value"` (with `\"`/`\\`/`\n` escapes) or nothing when absent/unterminated.
fn json_str(line: &str, key: &str) -> Option<String> {
    let rest = after_key(line, key)?.strip_prefix('"')?;
    let mut out = String::new();
    let mut esc = false;
    for c in rest.chars() {
        if esc {
            match c {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                other => out.push(other),
            }
            esc = false;
        } else if c == '\\' {
            esc = true;
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

/// `"key":123` (unsigned; leading `-` reads as none).
fn json_u64(line: &str, key: &str) -> Option<u64> {
    let digits: String = after_key(line, key)?.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() { None } else { digits.parse().ok() }
}

/// `"key":123` that fits a `u32` (a larger value reads as none, never truncated).
fn json_u32(line: &str, key: &str) -> Option<u32> {
    u32::try_from(json_u64(line, key)?).ok()
}

fn json_dims(line: &str, kind: &str) -> Option<Event> {
    let index = json_u64(line, "index")?;
    let width = json_u32(line, "width")?;
    let height = json_u32(line, "height")?;
    match kind {
        "frame" => Some(Event::Frame {
            index,
            width,
            height,
            format: FrameFormat::parse(json_str(line, "format").as_deref())?,
            slot: match after_key(line, "slot") {
                // present but not a u32: refused, never read as "on stdout"
                Some(_) => Some(json_u32(line, "slot")?),
                None => None,
            },
            gpu: match after_key(line, "gpu") {
                Some(_) => Some(GpuRef { buffer: json_u64(line, "gpu")?, pitch: json_u32(line, "pitch")? }),
                None => None,
            },
        }),
        "verify" => Some(Event::Verify { index, width, height }),
        _ => None,
    }
}

/// Parse one bridge NDJSON line.
pub fn parse_event(line: &str) -> Option<Event> {
    match json_str(line, "type").as_deref() {
        Some("metadata") => Some(Event::Metadata(Metadata {
            timecode: json_str(line, "timecode").unwrap_or_default(),
            fps_num: json_u32(line, "fps_num")?,
            fps_den: match json_u64(line, "fps_den") {
                Some(d) => u32::try_from(d).ok()?,
                None => 1,
            },
            width: json_u32(line, "width")?,
            height: json_u32(line, "height")?,
            frame_count: json_u64(line, "frame_count")?,
            protocol: json_u32(line, "protocol").unwrap_or(1),
            caps: Caps::parse(json_str(line, "caps").as_deref()),
            shm: json_u64(line, "shm_slots").zip(json_u64(line, "shm_slot_bytes")),
            color: json_str(line, "color"),
        })),
        Some("ready") => Some(Event::Ready {
            protocol: json_u32(line, "protocol").unwrap_or(1),
            caps: Caps::parse(json_str(line, "caps").as_deref()),
            shm: json_u64(line, "shm_slots").zip(json_u64(line, "shm_slot_bytes")),
            gpu_uuid: json_str(line, "gpu_uuid").as_deref().and_then(parse_uuid),
        }),
        Some("gpu-freed") => Some(Event::GpuFreed { buffer: json_u64(line, "gpu")? }),
        Some(kind @ ("frame" | "verify")) => json_dims(line, kind),
        Some("progress") => Some(Event::Progress { frame: json_u64(line, "frame")?, total: json_u64(line, "total")? }),
        Some("done") => Some(Event::Done),
        Some("error") => {
            let message = json_str(line, "message").unwrap_or_else(|| "unknown bridge error".into());
            Some(match json_u64(line, "index") {
                Some(index) => Event::FrameError { index, message },
                None => Event::Error(message),
            })
        }
        _ => None,
    }
}

/// Parse one multi-clip bridge line: the clip handle it names (if any) and the event.
pub fn parse_tagged(line: &str) -> Option<(Option<u64>, Event)> {
    let event = parse_event(line)?;
    let handle = match after_key(line, "handle") {
        // present but not a number: refused, never read as "no handle"
        Some(_) => Some(json_u64(line, "handle")?),
        None => None,
    };
    Some((handle, event))
}

/// Live persistent bridge processes per bridge binary ([`live_processes`]).
fn live_table() -> &'static Mutex<HashMap<PathBuf, usize>> {
    static LIVE: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Persistent decode processes of `bridge` running right now (single-clip sessions and shared
/// multi-clip bridges; the one-shot probes and audio extractions are not counted).
pub fn live_processes(bridge: &Path) -> usize {
    live_table().lock().unwrap_or_else(PoisonError::into_inner).get(bridge).copied().unwrap_or(0)
}

/// Counts one persistent process in [`live_processes`] while it lives.
struct ProcessToken(PathBuf);

impl ProcessToken {
    fn new(bridge: &Path) -> Self {
        let mut t = live_table().lock().unwrap_or_else(PoisonError::into_inner);
        let n = t.entry(bridge.to_path_buf()).or_insert(0);
        *n = n.saturating_add(1);
        ProcessToken(bridge.to_path_buf())
    }
}

impl Drop for ProcessToken {
    fn drop(&mut self) {
        let mut t = live_table().lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(n) = t.get_mut(&self.0) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                t.remove(&self.0);
            }
        }
    }
}

/// Frame bytes requested from any bridge and not yet received ([`MAX_INFLIGHT_BYTES`]).
static INFLIGHT: AtomicU64 = AtomicU64::new(0);

/// Frame bytes requested and not yet received, across every bridge.
pub fn inflight_bytes() -> u64 {
    INFLIGHT.load(Ordering::Relaxed)
}

/// A share of the in-flight frame budget, returned on drop.
pub struct Inflight(u64);

impl Inflight {
    /// Reserve `bytes`: always when `force` (an interactive request), otherwise only while the
    /// total stays within [`MAX_INFLIGHT_BYTES`].
    pub fn reserve(bytes: u64, force: bool) -> Option<Self> {
        let mut now = INFLIGHT.load(Ordering::Relaxed);
        loop {
            let next = now.saturating_add(bytes);
            if !force && next > MAX_INFLIGHT_BYTES {
                return None;
            }
            match INFLIGHT.compare_exchange_weak(now, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return Some(Inflight(bytes)),
                Err(seen) => now = seen,
            }
        }
    }
}

impl Drop for Inflight {
    fn drop(&mut self) {
        let _ = INFLIGHT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(self.0)));
    }
}

/// The budget share of one requested frame of a `width`×`height` clip (RGBA at full size: the
/// most any tier delivers).
fn frame_budget(width: u32, height: u32) -> u64 {
    u64::from(width).saturating_mul(u64::from(height)).saturating_mul(4)
}

fn base_command(bridge: &Path) -> Command {
    let mut cmd = Command::new(bridge);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd
}

/// Start `cmd`. A bridge binary being replaced (or a script just written while another thread
/// forks) fails `exec` with "text file busy" for a moment: retried briefly instead of failing.
fn spawn_child(cmd: &mut Command, bridge: &Path) -> Result<Child, String> {
    let mut tries = 0;
    loop {
        match cmd.spawn() {
            Ok(child) => return Ok(child),
            Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy && tries < 20 => {
                tries += 1;
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => return Err(format!("cannot run {}: {e}", bridge.display())),
        }
    }
}

/// Kill `child` (if still running) and reap it, so no zombie is left behind.
fn reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Run `f` on a named thread under `catch_unwind` (a panic is logged, never propagated: a dead
/// helper thread must not take anything else down). `false` when the thread could not start.
pub(crate) fn spawn_guarded(name: &str, f: impl FnOnce() + Send + 'static) -> bool {
    let label = name.to_string();
    std::thread::Builder::new()
        .name(label.clone())
        .spawn(move || {
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err() {
                log::error!("{label} thread panicked");
            }
        })
        .is_ok()
}

/// Drain a bridge's stderr into `tx` until it exits. The event channel closing early (receiver
/// dropped) just stops the forwarding; the pipe drain keeps the child from blocking on stderr.
fn forward_events(stderr: std::process::ChildStderr, tx: mpsc::Sender<Event>) {
    for line in BufReader::new(stderr).lines() {
        let Ok(line) = line else { break };
        if let Some(e) = parse_event(&line)
            && tx.send(e).is_err()
        {
            break;
        }
    }
}

/// Forward `child`'s stderr events on a guarded thread; the receiving end, or an error (the
/// child is then reaped).
fn events_of(child: &mut Child) -> Result<mpsc::Receiver<Event>, String> {
    let Some(stderr) = child.stderr.take() else {
        reap(child);
        return Err("no stderr pipe".into());
    };
    let (tx, rx) = mpsc::channel();
    if !spawn_guarded("filmcraft-rawcam-events", move || forward_events(stderr, tx)) {
        reap(child);
        return Err("cannot start the bridge event reader".into());
    }
    Ok(rx)
}

/// Wait for `child` to exit, at most `timeout`. Returns `None` on timeout (child still running).
fn wait_with_timeout(child: &mut Child, timeout: Duration) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {}
            Err(_) => return None,
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// `--probe-only`: clip metadata, or the bridge's error message.
pub fn probe(bridge: &Path, input: &Path, timeout: Duration) -> Result<Metadata, String> {
    let mut cmd = base_command(bridge);
    cmd.arg("--input").arg(input).arg("--probe-only").arg("--debayer").arg("full");
    let mut child = spawn_child(&mut cmd, bridge)?;
    let rx = events_of(&mut child)?;
    let mut result = Err(format!("{}: no metadata from bridge", input.display()));
    let deadline = Instant::now() + timeout;
    while let Ok(event) = rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        match event {
            Event::Metadata(m) => {
                if m.width == 0 || m.height == 0 || m.frame_count == 0 || m.fps_num == 0 {
                    result = Err(format!("{}: bridge reported empty clip", input.display()));
                } else {
                    result = Ok(m);
                }
                break;
            }
            Event::Error(e) => {
                result = Err(e);
                break;
            }
            _ => {}
        }
    }
    let _ = wait_with_timeout(&mut child, Duration::from_secs(2));
    reap(&mut child);
    result
}

/// `--extract-audio`: decode the clip's audio into `out` (WAV).
pub fn extract_audio(bridge: &Path, input: &Path, out: &Path, timeout: Duration) -> Result<(), String> {
    let mut cmd = base_command(bridge);
    cmd.arg("--input").arg(input).arg("--extract-audio").arg(out);
    let mut child = spawn_child(&mut cmd, bridge)?;
    let rx = events_of(&mut child)?;
    let status = match wait_with_timeout(&mut child, timeout) {
        Some(s) => s,
        None => {
            reap(&mut child);
            return Err(format!("{}: audio extraction timed out", input.display()));
        }
    };
    let mut error = None;
    for event in rx.try_iter() {
        if let Event::Error(e) = event {
            error = Some(e);
        }
    }
    if status.success() && out.is_file() { Ok(()) } else { Err(error.unwrap_or_else(|| format!("{}: audio extraction failed", input.display()))) }
}

/// Read exactly `n` bytes (`n` already capped by [`frame_bytes`]); the buffer is reserved
/// fallibly and never zero-filled first.
fn read_pixels(from: &mut impl Read, n: usize) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    buf.try_reserve_exact(n).map_err(|_| format!("no memory for a {} MB frame", n >> 20))?;
    let limit = u64::try_from(n).map_err(|_| "frame too large".to_string())?;
    from.take(limit).read_to_end(&mut buf).map_err(|e| format!("bridge frame stream broke: {e}"))?;
    if buf.len() != n {
        return Err(format!("bridge frame stream ended after {} of {n} bytes", buf.len()));
    }
    Ok(buf)
}

/// The shared-memory ring a bridge announced, validated: `slots` slots of `slot_bytes` bytes,
/// all inside a file of `file_len` bytes, within [`MAX_SHM_SLOTS`] / [`MAX_SHM_BYTES`].
pub fn shm_layout(slots: u64, slot_bytes: u64, file_len: u64) -> Result<(u32, u64), String> {
    if slots == 0 || slots > MAX_SHM_SLOTS {
        return Err(format!("bridge announced {slots} shared-memory slots (1..={MAX_SHM_SLOTS})"));
    }
    if slot_bytes == 0 || slot_bytes > MAX_SHM_BYTES {
        return Err(format!("bridge announced {slot_bytes}-byte shared-memory slots"));
    }
    let total = slots
        .checked_mul(slot_bytes)
        .filter(|&t| t <= MAX_SHM_BYTES)
        .ok_or_else(|| format!("bridge announced a {slots}×{slot_bytes}-byte ring (over {} GB)", MAX_SHM_BYTES >> 30))?;
    if total > file_len {
        return Err(format!("bridge announced a {total}-byte ring in a {file_len}-byte file"));
    }
    let slots = u32::try_from(slots).map_err(|_| "too many shared-memory slots".to_string())?;
    Ok((slots, slot_bytes))
}

/// Where a `bytes`-long frame announced in `slot` starts, or why it is not believed (no such
/// slot, more bytes than a slot holds).
pub fn shm_offset(layout: (u32, u64), slot: u32, bytes: usize) -> Result<u64, String> {
    let (slots, slot_bytes) = layout;
    if slot >= slots {
        return Err(format!("bridge announced shared-memory slot {slot} of {slots}"));
    }
    let len = u64::try_from(bytes).map_err(|_| "frame too large".to_string())?;
    if len > slot_bytes {
        return Err(format!("bridge announced a {len}-byte frame in a {slot_bytes}-byte slot"));
    }
    u64::from(slot).checked_mul(slot_bytes).ok_or_else(|| "shared-memory offset overflows".to_string())
}

/// The client side of the shared-memory transport: the (already unlinked) ring file and its
/// validated layout. Reads are positional `read`s into a fresh buffer — one kernel copy per
/// frame, no mapping (and no `unsafe`) needed.
pub struct ShmRing {
    file: File,
    layout: (u32, u64),
}

impl ShmRing {
    /// A fresh, empty ring file for a bridge to lay out, and the path the bridge opens it by.
    /// It lives on tmpfs (`/dev/shm`, else the temp dir) and is unlinked before this returns:
    /// both processes hold it open, and it disappears with the last of them — even after a
    /// crash. Linux only (the path is `/proc/<pid>/fd/<n>`).
    #[cfg(target_os = "linux")]
    pub fn create() -> Result<(File, std::path::PathBuf), String> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::OpenOptionsExt;
        use std::sync::atomic::{AtomicU64, Ordering};
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let shm = Path::new("/dev/shm");
        let dir = if shm.is_dir() { shm.to_path_buf() } else { std::env::temp_dir() };
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos());
        let name = format!("filmcraft-rawcam-{}-{nanos}-{}", std::process::id(), SERIAL.fetch_add(1, Ordering::Relaxed));
        let path = dir.join(name);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        std::fs::remove_file(&path).map_err(|e| format!("cannot unlink {}: {e}", path.display()))?;
        let via = std::path::PathBuf::from(format!("/proc/{}/fd/{}", std::process::id(), file.as_raw_fd()));
        Ok((file, via))
    }

    /// The ring a bridge announced over `file`, once its layout checks out.
    pub fn open(file: File, slots: u64, slot_bytes: u64) -> Result<Self, String> {
        let len = file.metadata().map_err(|e| format!("shared-memory file: {e}"))?.len();
        Ok(ShmRing { file, layout: shm_layout(slots, slot_bytes, len)? })
    }

    /// The `bytes` of a frame announced in `slot` (`bytes` already capped by [`frame_bytes`]);
    /// a file the bridge cut short is an error, never a short frame.
    pub fn read(&mut self, slot: u32, bytes: usize) -> Result<Vec<u8>, String> {
        let at = shm_offset(self.layout, slot, bytes)?;
        self.file.seek(SeekFrom::Start(at)).map_err(|e| format!("shared-memory seek: {e}"))?;
        read_pixels(&mut self.file, bytes).map_err(|e| format!("shared memory: {e}"))
    }
}

/// The pixel side of a session: a guarded thread owns the bridge's stdout and reads one frame
/// per request, so the caller waits with a deadline instead of blocking in `read` on a bridge
/// that stalled mid-frame.
struct PixelReader {
    want: mpsc::Sender<usize>,
    got: mpsc::Receiver<Result<Vec<u8>, String>>,
}

impl PixelReader {
    fn start(mut stdout: ChildStdout) -> Option<Self> {
        let (want, want_rx) = mpsc::channel::<usize>();
        let (got_tx, got) = mpsc::channel();
        let started = spawn_guarded("filmcraft-rawcam-pixels", move || {
            for n in want_rx {
                let read = read_pixels(&mut stdout, n);
                let failed = read.is_err();
                if got_tx.send(read).is_err() || failed {
                    break;
                }
            }
        });
        started.then_some(PixelReader { want, got })
    }
}

/// One request's outcome on a session that is still in sync: the frame, or why the bridge could
/// not produce it (a protocol 2 per-frame error — the session stays usable). Failures that break
/// the session are the outer `Err` of the request functions.
pub type Reply = Result<FrameBytes, String>;

/// A persistent decode session: the bridge process stays alive (`--persistent`), the SDK is
/// initialised once, and frames are requested one `frame <index>` command at a time (or in
/// batches — [`DecodeSession::request_batch`]). Events read past the reply being handled are
/// parked in [`DecodeSession::pending`]: a pipelined bridge emits the next reply's `frame`
/// announcement while this one is still being read, and losing it would stall the stream.
pub struct DecodeSession {
    child: Child,
    stdin: ChildStdin,
    pixels: PixelReader,
    events: mpsc::Receiver<Event>,
    pending: std::collections::VecDeque<Event>,
    /// The last frame read whose `verify` event has not been seen yet: it is checked when that
    /// event arrives (usually while the next reply is awaited — the stderr thread delivers it a
    /// moment after the pixels), so a desynchronised stream fails the next request at the latest.
    unverified: Option<(u64, u32, u32)>,
    /// The start-up `metadata` event: this bridge's capabilities.
    meta: Metadata,
    /// The shared-memory transport, when the bridge took it.
    shm: Option<ShmRing>,
    /// Where the last reply's time went: waiting for its announcement (decode), then reading
    /// its pixels (transport). Diagnostics only (`FILMCRAFT_RAWCAM_STATS`, the benches).
    pub last_timing: (Duration, Duration),
    _token: ProcessToken,
}

impl DecodeSession {
    /// Spawn a persistent session at debayer tier `debayer` (the bridge's tier word:
    /// `full`/`half`/`quarter`/`eighth`, `premium` for R3D full quality; with [`Caps::scale`]
    /// only the default for commands without a scale word). Waits for the startup `metadata`
    /// event so a broken bridge fails here, not at the first frame. On every failure the bridge
    /// process is killed and reaped. `shm` offers the shared-memory transport (only to a bridge
    /// with [`Caps::shm`] — an older one refuses the argument); the session uses it when the
    /// bridge accepts, stdout otherwise.
    pub fn spawn(bridge: &Path, input: &Path, debayer: &str, shm: bool, timeout: Duration) -> Result<Self, String> {
        let mut cmd = Command::new(bridge);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd.arg("--input").arg(input).arg("--debayer").arg(debayer).arg("--persistent");
        let mut ring_file = None;
        #[cfg(target_os = "linux")]
        if shm {
            match ShmRing::create() {
                Ok((file, via)) => {
                    cmd.arg("--shm").arg(via);
                    ring_file = Some(file);
                }
                Err(e) => log::warn!("no shared-memory transport ({e}); frames come over the pipe"),
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = shm;
        let mut child = spawn_child(&mut cmd, bridge)?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            reap(&mut child);
            return Err("no stdin/stdout pipe".into());
        };
        let events = events_of(&mut child)?;
        let Some(pixels) = PixelReader::start(stdout) else {
            reap(&mut child);
            return Err("cannot start the bridge pixel reader".into());
        };
        let deadline = Instant::now() + timeout;
        loop {
            match events.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(Event::Metadata(meta)) => {
                    let shm = match (ring_file, meta.shm) {
                        (Some(file), Some((slots, slot_bytes))) => match ShmRing::open(file, slots, slot_bytes) {
                            Ok(ring) => Some(ring),
                            Err(e) => {
                                reap(&mut child);
                                return Err(e);
                            }
                        },
                        _ => None,
                    };
                    return Ok(Self {
                        child,
                        stdin,
                        pixels,
                        events,
                        pending: std::collections::VecDeque::new(),
                        unverified: None,
                        meta,
                        shm,
                        last_timing: (Duration::ZERO, Duration::ZERO),
                        _token: ProcessToken::new(bridge),
                    });
                }
                Ok(Event::Error(e)) => {
                    reap(&mut child);
                    return Err(e);
                }
                Ok(_) => {}
                Err(_) => {
                    reap(&mut child);
                    return Err(format!("{}: bridge did not start up (no metadata event)", input.display()));
                }
            }
        }
    }

    /// What this bridge supports (from its start-up `metadata` event).
    pub fn caps(&self) -> Caps {
        self.meta.caps
    }

    /// Whether frames come through shared memory (the bridge accepted `--shm`).
    pub fn uses_shm(&self) -> bool {
        self.shm.is_some()
    }

    /// Whether the bridge process is still running (a pooled session whose bridge crashed while
    /// idle is replaced instead of failing the next request).
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// The next event: parked ones first, then the channel (`None` on timeout).
    fn next_event(&mut self, deadline: Instant) -> Option<std::result::Result<Event, mpsc::RecvTimeoutError>> {
        if let Some(event) = self.pending.pop_front() {
            return Some(Ok(event));
        }
        let timeout = deadline.saturating_duration_since(Instant::now());
        if timeout.is_zero() {
            return match self.events.try_recv() {
                Ok(event) => Some(Ok(event)),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => Some(Err(mpsc::RecvTimeoutError::Disconnected)),
            };
        }
        match self.events.recv_timeout(timeout) {
            Ok(event) => Some(Ok(event)),
            Err(e) => Some(Err(e)),
        }
    }

    /// The command line for frame `index`: with the per-command `scale` word when the bridge
    /// takes one (otherwise the session's own tier applies and the word is left out).
    fn command(&self, index: u64, scale: Option<&str>) -> String {
        match scale.filter(|_| self.meta.caps.scale) {
            Some(word) => format!("frame {index} {word}\n"),
            None => format!("frame {index}\n"),
        }
    }

    /// Request one frame: sends `frame <index> [scale]`, waits for the `frame` event (its
    /// dimensions size the read), reads the pixels and checks the trailing `verify` event. The
    /// outer error means the byte stream can no longer be trusted — the caller must discard the
    /// session; the inner one is a per-frame error (the session stays usable).
    pub fn request_frame(&mut self, index: u64, scale: Option<&str>, timeout: Duration) -> Result<Reply, String> {
        let cmd = self.command(index, scale);
        self.stdin.write_all(cmd.as_bytes()).map_err(|e| format!("bridge command failed: {e}"))?;
        self.stdin.flush().map_err(|e| format!("bridge command failed: {e}"))?;
        let deadline = Instant::now() + timeout;
        self.receive_frame(index, deadline)
    }

    /// Request several frames in one go: the bridge pipelines them (reads overlap the running
    /// decodes). Replies arrive (and are returned) in command order. On an outer error the
    /// stream is desynchronised — the caller must discard the session.
    pub fn request_batch(&mut self, indices: &[u64], scale: Option<&str>, timeout: Duration) -> Result<Vec<Reply>, String> {
        self.request_batch_steered(indices, scale, indices.len().max(1), || true, timeout)
    }

    /// [`Self::request_batch`] through a sliding window: at most `window` commands are
    /// outstanding, and each reply read makes room for the next command — the bridge's pipeline
    /// stays full instead of draining between chunks. `steer_more` is consulted before every
    /// further command: the frames already requested are committed (their replies must be
    /// read), but no more are sent once the caller no longer wants them, so an interactive
    /// request waits at most `window` replies behind the read-ahead. `timeout` bounds each reply.
    pub fn request_batch_steered(
        &mut self,
        indices: &[u64],
        scale: Option<&str>,
        window: usize,
        mut steer_more: impl FnMut() -> bool,
        timeout: Duration,
    ) -> Result<Vec<Reply>, String> {
        let window = window.max(1);
        let mut out = Vec::with_capacity(indices.len());
        let mut sent = 0;
        let mut sending = true;
        let budget = frame_budget(self.meta.width, self.meta.height);
        let mut held = VecDeque::new();
        while out.len() < indices.len() {
            let mut cmd = String::new();
            while sending && sent < indices.len() && sent - out.len() < window {
                if sent > 0 && !steer_more() {
                    sending = false;
                    break;
                }
                // The first frame always goes; more only while the global budget allows.
                let Some(share) = Inflight::reserve(budget, sent == 0) else {
                    sending = false;
                    break;
                };
                held.push_back(share);
                if let Some(&index) = indices.get(sent) {
                    cmd.push_str(&self.command(index, scale));
                }
                sent += 1;
            }
            if !cmd.is_empty() {
                self.stdin.write_all(cmd.as_bytes()).map_err(|e| format!("bridge command failed: {e}"))?;
                self.stdin.flush().map_err(|e| format!("bridge command failed: {e}"))?;
            }
            let Some(&index) = indices.get(out.len()).filter(|_| out.len() < sent) else { break };
            out.push(self.receive_frame(index, Instant::now() + timeout)?);
            held.pop_front();
        }
        Ok(out)
    }

    /// Check a `verify` event against the frame it trails.
    fn check_verify(&mut self, index: u64, width: u32, height: u32) -> Result<(), String> {
        let Some((i, w, h)) = self.unverified.take() else { return Ok(()) };
        if index != i {
            return Err(format!("bridge out of sync: verified frame {index} for {i}"));
        }
        if (width, height) != (w, h) {
            return Err(format!("bridge dimensions disagree: {width}×{height} after {w}×{h}"));
        }
        Ok(())
    }

    /// Wait for the reply to `index` — its `frame` event (dimensions and layout size the read)
    /// or its per-frame error — read the pixels (on the reader thread, until `deadline`), check
    /// the `verify` events. Events of later replies are parked, not consumed.
    fn receive_frame(&mut self, index: u64, deadline: Instant) -> Result<Reply, String> {
        let waiting_since = Instant::now();
        let (width, height, format, slot) = loop {
            match self.next_event(deadline) {
                Some(Ok(Event::Frame { index: i, width, height, format, slot, .. })) => {
                    if i != index {
                        return Err(format!("bridge out of sync: answered frame {i} for {index}"));
                    }
                    break (width, height, format, slot);
                }
                Some(Ok(Event::FrameError { index: i, message })) => {
                    if i != index {
                        return Err(format!("bridge out of sync: error for frame {i} while waiting for {index}"));
                    }
                    // The previous frame's `verify` came before this reply (or never will).
                    self.unverified = None;
                    return Ok(Err(message));
                }
                Some(Ok(Event::Verify { index: i, width: w, height: h })) => self.check_verify(i, w, h)?,
                Some(Ok(Event::Error(e))) => return Err(e),
                Some(Ok(_)) => {}
                Some(Err(mpsc::RecvTimeoutError::Disconnected)) => return Err("bridge exited".into()),
                Some(Err(mpsc::RecvTimeoutError::Timeout)) | None => return Err("bridge frame timed out".into()),
            }
        };
        // The previous frame's `verify` would have come before this announcement: a bridge that
        // sends none (an older one) is not held to it.
        self.unverified = None;
        let announced = Instant::now();
        let bytes = frame_bytes(width, height, format)?;
        let data = match slot {
            // Shared memory: one positional read (tmpfs never blocks on the bridge), then the
            // slot goes back to the bridge.
            Some(slot) => {
                let Some(ring) = self.shm.as_mut() else {
                    return Err(format!("bridge announced shared-memory slot {slot} without a ring"));
                };
                let data = ring.read(slot, bytes)?;
                self.stdin.write_all(format!("release {slot}\n").as_bytes()).map_err(|e| format!("bridge command failed: {e}"))?;
                self.stdin.flush().map_err(|e| format!("bridge command failed: {e}"))?;
                data
            }
            None => {
                self.pixels.want.send(bytes).map_err(|_| "bridge frame stream closed".to_string())?;
                match self.pixels.got.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                    Ok(read) => read?,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        // Stalled mid-frame: kill it (the reader thread then sees the pipe close).
                        reap(&mut self.child);
                        return Err("bridge frame timed out".into());
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => return Err("bridge frame stream closed".into()),
                }
            }
        };
        self.unverified = Some((index, width, height));
        self.last_timing = (announced.duration_since(waiting_since), announced.elapsed());
        // The `verify` event trails its pixels; when it is already here, check it now and park
        // anything else — it belongs to a later reply, and draining past it would swallow that
        // reply's `frame` announcement and stall the whole batch. Otherwise the next reply checks it.
        if let Ok(event) = self.events.try_recv() {
            match event {
                Event::Verify { index: i, width: w, height: h } => self.check_verify(i, w, h)?,
                Event::Error(e) => return Err(e),
                other => self.pending.push_back(other),
            }
        }
        Ok(Ok(FrameBytes { data, width, height, format, gpu: None }))
    }
}

impl Drop for DecodeSession {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        reap(&mut self.child);
    }
}

/// A decode session as the sources pool it: a bridge process of its own (single-clip
/// `--persistent`), or one clip handle on the format's shared multi-clip bridge.
pub enum Session {
    Own(DecodeSession),
    Shared(ClipSession),
}

impl Session {
    pub fn caps(&self) -> Caps {
        match self {
            Session::Own(s) => s.caps(),
            Session::Shared(s) => s.caps(),
        }
    }

    pub fn uses_shm(&self) -> bool {
        match self {
            Session::Own(s) => s.uses_shm(),
            Session::Shared(s) => s.uses_shm(),
        }
    }

    pub fn is_alive(&mut self) -> bool {
        match self {
            Session::Own(s) => s.is_alive(),
            Session::Shared(s) => s.is_alive(),
        }
    }

    /// Whether this is a handle on a shared bridge (dropping it closes the handle only).
    pub fn is_shared(&self) -> bool {
        matches!(self, Session::Shared(_))
    }

    /// The GPU the session can export frames from (a shared bridge with the `gpu` capability).
    pub fn gpu_device(&self) -> Option<[u8; 16]> {
        match self {
            Session::Own(_) => None,
            Session::Shared(s) => s.bridge().gpu_device(),
        }
    }

    /// Ask for the next frames in GPU memory, where the session can export them.
    pub fn set_gpu(&mut self, on: bool) {
        if let Session::Shared(s) = self {
            s.set_gpu(on);
        }
    }

    pub fn last_timing(&self) -> (Duration, Duration) {
        match self {
            Session::Own(s) => s.last_timing,
            Session::Shared(s) => s.last_timing,
        }
    }

    pub fn request_frame(&mut self, index: u64, scale: Option<&str>, timeout: Duration) -> Result<Reply, String> {
        match self {
            Session::Own(s) => s.request_frame(index, scale, timeout),
            Session::Shared(s) => s.request_frame(index, scale, timeout),
        }
    }

    pub fn request_batch(&mut self, indices: &[u64], scale: Option<&str>, timeout: Duration) -> Result<Vec<Reply>, String> {
        self.request_batch_steered(indices, scale, indices.len().max(1), || true, timeout)
    }

    pub fn request_batch_steered(
        &mut self,
        indices: &[u64],
        scale: Option<&str>,
        window: usize,
        steer_more: impl FnMut() -> bool,
        timeout: Duration,
    ) -> Result<Vec<Reply>, String> {
        match self {
            Session::Own(s) => s.request_batch_steered(indices, scale, window, steer_more, timeout),
            Session::Shared(s) => s.request_batch_steered(indices, scale, window, steer_more, timeout),
        }
    }
}

/// How a multi-clip `open` failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenFail {
    /// The bridge refused this clip (it stays up for the others).
    Clip(String),
    /// The bridge itself broke (exited, stalled, desynchronised).
    Bridge(String),
}

/// What a request on a shared bridge gets back from the event reader.
enum Arrived {
    Pixels(FrameBytes),
    /// The pixels wait in a shared-memory slot; the lease hands the slot back when dropped.
    Slot {
        lease: SlotLease,
        bytes: usize,
        width: u32,
        height: u32,
        format: FrameFormat,
    },
    /// The frame is in an exported GPU buffer; the release hands it back if nobody keeps it.
    Gpu {
        release: GpuRelease,
        gpu: GpuRef,
        width: u32,
        height: u32,
        format: FrameFormat,
    },
    /// The bridge could not produce this frame (the handle goes on).
    Failed(String),
}

/// Outer `Err`: the bridge broke.
type Delivery = Result<Arrived, String>;

struct Pending {
    index: u64,
    tx: mpsc::Sender<Delivery>,
}

/// The reader thread's routing table.
#[derive(Default)]
struct Demux {
    /// Unanswered frame requests per handle, in command order.
    queues: HashMap<u64, VecDeque<Pending>>,
    /// `open` commands waiting for their `metadata` (or error).
    opens: HashMap<u64, mpsc::Sender<Result<Metadata, String>>>,
    /// The clip behind each open handle (crash attribution).
    paths: HashMap<u64, PathBuf>,
    /// The last frame read whose `verify` has not arrived yet: (handle, index, width, height).
    unverified: Option<(u64, u64, u32, u32)>,
    /// Why the bridge is unusable, once it is.
    dead: Option<String>,
}

/// What a dead shared bridge leaves behind for its manager: the clips that had requests
/// outstanding when it died (crash suspects), and whether it ever delivered a frame.
#[derive(Clone, Debug, Default)]
pub struct DeathReport {
    pub suspects: Vec<PathBuf>,
    pub served: bool,
    /// It died on its own (crash, desync, wedge), not because its last user let it go.
    pub crashed: bool,
}

/// The shared ring file: the client's end of it (kept open so `/proc/self/fd/<n>` stays valid)
/// and its validated layout. Each [`ClipSession`] reads through a description of its own.
struct RingFile {
    _file: File,
    via: PathBuf,
    layout: (u32, u64),
}

struct MultiShared {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    demux: Mutex<Demux>,
    ring: OnceLock<RingFile>,
    caps: OnceLock<Caps>,
    /// The GPU buffers the bridge exported (with the `gpu` capability and a connected socket).
    gpu: OnceLock<Arc<GpuBuffers>>,
    served: AtomicBool,
    /// Its last user let it go: the exit that follows is no crash.
    retired: AtomicBool,
    report: Arc<Mutex<Option<DeathReport>>>,
}

impl MultiShared {
    /// Write one command line (the bridge's stdin is shared by every handle).
    fn send_line(&self, line: &str) -> Result<(), String> {
        let mut stdin = self.stdin.lock().unwrap_or_else(PoisonError::into_inner);
        stdin.write_all(line.as_bytes()).and_then(|()| stdin.flush()).map_err(|e| format!("bridge command failed: {e}"))
    }

    /// The bridge is unusable: every waiting request and `open` fails with `why`, the crash
    /// suspects are recorded, and the process is killed (the reader thread then sees EOF).
    fn fail(&self, why: &str) {
        let (queues, opens) = {
            let mut d = self.demux.lock().unwrap_or_else(PoisonError::into_inner);
            if d.dead.is_some() {
                (HashMap::new(), HashMap::new())
            } else {
                d.dead = Some(why.to_string());
                let suspects: Vec<PathBuf> = d.queues.iter().filter(|(_, q)| !q.is_empty()).filter_map(|(h, _)| d.paths.get(h).cloned()).collect();
                *self.report.lock().unwrap_or_else(PoisonError::into_inner) =
                    Some(DeathReport { suspects, served: self.served.load(Ordering::Relaxed), crashed: !self.retired.load(Ordering::Relaxed) });
                (std::mem::take(&mut d.queues), std::mem::take(&mut d.opens))
            }
        };
        // Outside the routing lock: a dropped delivery may write a `release`.
        for p in queues.into_values().flatten() {
            let _ = p.tx.send(Err(why.to_string()));
        }
        for tx in opens.into_values() {
            let _ = tx.send(Err(why.to_string()));
        }
        let _ = self.child.lock().unwrap_or_else(PoisonError::into_inner).kill();
    }
}

/// Hands a shared-memory slot back to the bridge when dropped (read, or never collected).
struct SlotLease {
    shared: Weak<MultiShared>,
    slot: u32,
}

impl Drop for SlotLease {
    fn drop(&mut self) {
        if let Some(shared) = self.shared.upgrade() {
            let _ = shared.send_line(&format!("release {}\n", self.slot));
        }
    }
}

/// The GPU buffers a bridge exported, by its buffer id: filled by a reader thread from the
/// socket (`SCM_RIGHTS`), emptied by `gpu-freed` events. Frames naming a buffer wait here for it.
pub struct GpuBuffers {
    uuid: [u8; 16],
    table: Mutex<GpuTable>,
    arrived: std::sync::Condvar,
}

#[derive(Default)]
struct GpuTable {
    by_id: HashMap<u64, Arc<filmcraft_frame::ExternalMemory>>,
    /// The socket is gone (bridge exited, or it sent something unusable): no buffer comes any more.
    closed: bool,
}

impl GpuBuffers {
    fn new(uuid: [u8; 16]) -> Self {
        GpuBuffers { uuid, table: Mutex::new(GpuTable::default()), arrived: std::sync::Condvar::new() }
    }

    /// Whether buffers can still arrive.
    pub fn usable(&self) -> bool {
        !self.table.lock().unwrap_or_else(PoisonError::into_inner).closed
    }

    /// A buffer the bridge exported: `size` bytes behind `handle` (a file descriptor).
    fn insert(&self, id: u64, size: u64, handle: Box<dyn std::any::Any + Send + Sync>) {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let memory = Arc::new(filmcraft_frame::ExternalMemory::new(NEXT.fetch_add(1, Ordering::Relaxed), size, self.uuid, handle));
        self.table.lock().unwrap_or_else(PoisonError::into_inner).by_id.insert(id, memory);
        self.arrived.notify_all();
    }

    /// The bridge let go of buffer `id`: frames still holding it keep the memory alive.
    fn forget(&self, id: u64) {
        self.table.lock().unwrap_or_else(PoisonError::into_inner).by_id.remove(&id);
    }

    fn close(&self) {
        self.table.lock().unwrap_or_else(PoisonError::into_inner).closed = true;
        self.arrived.notify_all();
    }

    /// Buffer `id`, waiting until `deadline` for its descriptor to arrive.
    fn wait_for(&self, id: u64, deadline: Instant) -> Option<Arc<filmcraft_frame::ExternalMemory>> {
        let mut t = self.table.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Some(m) = t.by_id.get(&id) {
                return Some(m.clone());
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if t.closed || left.is_zero() {
                return None;
            }
            t = self.arrived.wait_timeout(t, left).unwrap_or_else(PoisonError::into_inner).0;
        }
    }
}

/// The socket a bridge connects to for `--gpu-export`: a listener in a private directory, both
/// removed once the bridge connected (or gave up).
#[cfg(target_os = "linux")]
struct GpuSocket {
    dir: PathBuf,
    listener: std::os::unix::net::UnixListener,
}

#[cfg(target_os = "linux")]
impl GpuSocket {
    fn create() -> Result<Self, String> {
        use std::os::unix::fs::DirBuilderExt;
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let base = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).filter(|d| d.is_dir()).unwrap_or_else(std::env::temp_dir);
        let dir = base.join(format!("filmcraft-gpu-{}-{}", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed)));
        std::fs::DirBuilder::new().mode(0o700).create(&dir).map_err(|e| format!("GPU socket dir: {e}"))?;
        match std::os::unix::net::UnixListener::bind(dir.join("s")) {
            Ok(listener) => Ok(GpuSocket { dir, listener }),
            Err(e) => {
                let _ = std::fs::remove_dir_all(&dir);
                Err(format!("GPU socket: {e}"))
            }
        }
    }

    fn path(&self) -> PathBuf {
        self.dir.join("s")
    }

    /// The bridge's connection, if it made one (it connects before its `ready` event).
    fn accept(&self) -> Option<std::os::unix::net::UnixStream> {
        self.listener.set_nonblocking(true).ok()?;
        let (stream, _) = self.listener.accept().ok()?;
        stream.set_nonblocking(false).ok()?;
        Some(stream)
    }
}

#[cfg(target_os = "linux")]
impl Drop for GpuSocket {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Read `(id, size, fd)` messages off the bridge's socket into `buffers` until it closes or sends
/// something unusable.
#[cfg(target_os = "linux")]
fn gpu_socket_loop(stream: std::os::unix::net::UnixStream, buffers: &GpuBuffers) {
    use rustix::net::{RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, recvmsg};
    use std::mem::MaybeUninit;
    loop {
        let mut payload = [0u8; 16];
        let mut got = 0;
        let mut fd = None;
        while got < payload.len() {
            let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(4))];
            let mut control = RecvAncillaryBuffer::new(&mut space);
            let Some(rest) = payload.get_mut(got..) else { break };
            let n = match recvmsg(&stream, &mut [std::io::IoSliceMut::new(rest)], &mut control, RecvFlags::CMSG_CLOEXEC) {
                Ok(m) if m.bytes > 0 => m.bytes,
                _ => return buffers.close(),
            };
            for msg in control.drain() {
                if let RecvAncillaryMessage::ScmRights(fds) = msg {
                    for f in fds {
                        // one descriptor per message; extras are closed as they drop
                        if fd.is_none() {
                            fd = Some(f);
                        }
                    }
                }
            }
            got += n;
        }
        let (Some(id), Some(size)) = (
            payload.get(..8).and_then(|b| b.try_into().ok()).map(u64::from_le_bytes),
            payload.get(8..).and_then(|b| b.try_into().ok()).map(u64::from_le_bytes),
        ) else {
            return buffers.close();
        };
        match fd {
            Some(fd) if size > 0 && size <= MAX_GPU_BUFFER => buffers.insert(id, size, Box::new(fd)),
            _ => {
                log::warn!("camera-RAW bridge sent an unusable GPU buffer ({id}, {size} bytes); no more GPU frames from it");
                return buffers.close();
            }
        }
    }
}

/// One multi-clip bridge process (`--persistent --multi`) shared by every clip of its format:
/// one CUDA context, one SDK instance, one set of GPU pools. A guarded reader thread routes the
/// bridge's events to the requests waiting on each handle; dropping the last reference kills and
/// reaps the process.
pub struct MultiBridge {
    shared: Arc<MultiShared>,
    _token: ProcessToken,
}

impl MultiBridge {
    /// Start a multi-clip bridge and wait for its `ready` event. `shm_slot_bytes`: offer a
    /// shared-memory ring with slots of this size (the largest frame expected; a bridge without
    /// [`Caps::shm`] must not be offered one). On every failure the process is killed and reaped.
    ///
    /// `gpu_export`: offer the GPU frame socket (a bridge without [`Caps::gpu`] must not be
    /// offered one: it refuses the argument).
    pub fn spawn(bridge: &Path, shm_slot_bytes: Option<u64>, gpu_export: bool, timeout: Duration) -> Result<Arc<Self>, String> {
        let mut cmd = Command::new(bridge);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd.arg("--persistent").arg("--multi");
        #[cfg(target_os = "linux")]
        let gpu_socket = if gpu_export {
            match GpuSocket::create() {
                Ok(sock) => {
                    cmd.arg("--gpu-export").arg(sock.path());
                    Some(sock)
                }
                Err(e) => {
                    log::warn!("no GPU frames ({e}); they come through host memory");
                    None
                }
            }
        } else {
            None
        };
        #[cfg(not(target_os = "linux"))]
        let _ = gpu_export;
        let mut ring_file = None;
        #[cfg(target_os = "linux")]
        if let Some(bytes) = shm_slot_bytes {
            match ShmRing::create() {
                Ok((file, via)) => {
                    cmd.arg("--shm").arg(&via).arg("--shm-slot-bytes").arg(bytes.to_string());
                    ring_file = Some((file, via));
                }
                Err(e) => log::warn!("no shared-memory transport ({e}); frames come over the pipe"),
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = shm_slot_bytes;
        let mut child = spawn_child(&mut cmd, bridge)?;
        let (Some(stdin), Some(stdout), Some(stderr)) = (child.stdin.take(), child.stdout.take(), child.stderr.take()) else {
            reap(&mut child);
            return Err("no stdin/stdout/stderr pipe".into());
        };
        let shared = Arc::new(MultiShared {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            demux: Mutex::new(Demux::default()),
            ring: OnceLock::new(),
            caps: OnceLock::new(),
            gpu: OnceLock::new(),
            served: AtomicBool::new(false),
            retired: AtomicBool::new(false),
            report: Arc::new(Mutex::new(None)),
        });
        let this = Arc::new(MultiBridge { shared: shared.clone(), _token: ProcessToken::new(bridge) });
        let (ready_tx, ready_rx) = mpsc::channel();
        let reader = shared.clone();
        let started = spawn_guarded("filmcraft-rawcam-demux", move || {
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| demux_loop(&reader, stderr, stdout, ready_tx)));
            if run.is_err() {
                // Nobody would route replies any more: every waiting clip must learn it now.
                reader.fail("the bridge event reader failed");
            }
        });
        if !started {
            return Err("cannot start the bridge event reader".into());
        }
        let ready = match ready_rx.recv_timeout(timeout) {
            Ok(r) => r,
            Err(_) => Err(format!("{}: bridge did not start up (no ready event)", bridge.display())),
        };
        let (caps, shm, gpu_uuid) = ready?; // dropping `this` kills and reaps the bridge
        if !caps.multi {
            return Err(format!("{}: bridge does not serve multiple clips", bridge.display()));
        }
        #[cfg(target_os = "linux")]
        if let (Some(sock), Some(uuid)) = (gpu_socket, gpu_uuid.filter(|_| caps.gpu)) {
            match sock.accept() {
                Some(stream) => {
                    let buffers = Arc::new(GpuBuffers::new(uuid));
                    let reader = buffers.clone();
                    let started = spawn_guarded("filmcraft-rawcam-gpu", move || {
                        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu_socket_loop(stream, &reader))).is_err() {
                            reader.close();
                        }
                    });
                    if started {
                        let _ = shared.gpu.set(buffers);
                    }
                }
                None => log::warn!("{}: announced GPU frames but did not connect", bridge.display()),
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = gpu_uuid;
        if let (Some((file, via)), Some((slots, slot_bytes))) = (ring_file, shm) {
            let len = file.metadata().map_err(|e| format!("shared-memory file: {e}"))?.len();
            let layout = shm_layout(slots, slot_bytes, len)?;
            let _ = shared.ring.set(RingFile { _file: file, via, layout });
        }
        let _ = shared.caps.set(caps);
        Ok(this)
    }

    /// The capabilities from the `ready` event.
    pub fn caps(&self) -> Caps {
        self.shared.caps.get().copied().unwrap_or_default()
    }

    /// The GPU this bridge exports frames from, while it can (`gpu` capability, socket up).
    pub fn gpu_device(&self) -> Option<[u8; 16]> {
        self.shared.gpu.get().filter(|g| g.usable()).map(|g| g.uuid)
    }

    /// Whether the bridge still serves (not exited, not desynchronised).
    pub fn is_alive(&self) -> bool {
        self.shared.demux.lock().unwrap_or_else(PoisonError::into_inner).dead.is_none()
    }

    /// Filled in once the bridge died: what its manager needs to decide about respawning.
    pub fn death_report(&self) -> Arc<Mutex<Option<DeathReport>>> {
        self.shared.report.clone()
    }

    /// Mark the bridge unusable and kill it (a request timed out: it is wedged).
    pub fn kill(&self, why: &str) {
        self.shared.fail(why);
    }

    /// Open `path` under `handle` (fresh for this bridge, never reused) and wait for its metadata.
    pub fn open(&self, handle: u64, path: &Path, timeout: Duration) -> Result<Metadata, OpenFail> {
        let line = open_line(handle, path).map_err(OpenFail::Clip)?;
        let (tx, rx) = mpsc::channel();
        {
            let _stdin = self.shared.stdin.lock().unwrap_or_else(PoisonError::into_inner);
            let mut d = self.shared.demux.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(why) = &d.dead {
                return Err(OpenFail::Bridge(why.clone()));
            }
            if d.paths.contains_key(&handle) || d.opens.contains_key(&handle) {
                return Err(OpenFail::Clip(format!("clip handle {handle} is in use")));
            }
            d.opens.insert(handle, tx);
            d.paths.insert(handle, path.to_path_buf());
        }
        if let Err(e) = self.shared.send_line(&line) {
            self.shared.fail(&e);
            return Err(OpenFail::Bridge(e));
        }
        let forget = || {
            let mut d = self.shared.demux.lock().unwrap_or_else(PoisonError::into_inner);
            d.opens.remove(&handle);
            d.paths.remove(&handle);
        };
        match rx.recv_timeout(timeout) {
            Ok(Ok(meta)) if meta.frame_count > 0 && meta.width > 0 && meta.height > 0 => Ok(meta),
            Ok(Ok(_)) => {
                forget();
                let _ = self.shared.send_line(&format!("close {handle}\n"));
                Err(OpenFail::Clip(format!("{}: bridge reported an empty clip", path.display())))
            }
            Ok(Err(e)) => {
                forget();
                if self.is_alive() { Err(OpenFail::Clip(e)) } else { Err(OpenFail::Bridge(e)) }
            }
            Err(_) => {
                let why = format!("{}: the bridge did not answer `open`", path.display());
                self.shared.fail(&why);
                Err(OpenFail::Bridge(why))
            }
        }
    }

    /// Close `handle`: its unanswered requests are dropped (no replies follow).
    fn close(&self, handle: u64) {
        let dropped = {
            let mut d = self.shared.demux.lock().unwrap_or_else(PoisonError::into_inner);
            d.paths.remove(&handle);
            d.opens.remove(&handle);
            let q = d.queues.remove(&handle);
            if d.unverified.is_some_and(|u| u.0 == handle) {
                d.unverified = None;
            }
            (q, d.dead.is_none())
        };
        if dropped.1 {
            let _ = self.shared.send_line(&format!("close {handle}\n"));
        }
        drop(dropped.0);
    }

    /// Queue `frame <handle> <index> [scale] [gpu]`; the reply arrives on the receiver. `gpu`
    /// asks for the frame in GPU memory (only with a scale word, and only while exports work).
    fn send_frame(&self, handle: u64, index: u64, scale: Option<&str>, gpu: bool) -> Result<mpsc::Receiver<Delivery>, String> {
        let line = match scale {
            Some(word) if gpu && self.gpu_device().is_some() && self.caps().f16 && super::f16_frames_enabled() => {
                format!("frame {handle} {index} {word} gpu f16\n")
            }
            Some(word) if gpu && self.gpu_device().is_some() => format!("frame {handle} {index} {word} gpu\n"),
            // host frames in 16 bits where the bridge has them
            Some(word) if self.caps().rgba16 && super::f16_frames_enabled() => format!("frame {handle} {index} {word} rgba16\n"),
            Some(word) => format!("frame {handle} {index} {word}\n"),
            None => format!("frame {handle} {index}\n"),
        };
        let (tx, rx) = mpsc::channel();
        // stdin before the routing table (the order every writer takes them in): the queue order
        // is the command order.
        let mut stdin = self.shared.stdin.lock().unwrap_or_else(PoisonError::into_inner);
        {
            let mut d = self.shared.demux.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(why) = &d.dead {
                return Err(why.clone());
            }
            d.queues.entry(handle).or_default().push_back(Pending { index, tx });
        }
        let written = stdin.write_all(line.as_bytes()).and_then(|()| stdin.flush());
        drop(stdin);
        if let Err(e) = written {
            let why = format!("bridge command failed: {e}");
            self.shared.fail(&why);
            return Err(why);
        }
        Ok(rx)
    }
}

impl Drop for MultiBridge {
    fn drop(&mut self) {
        self.shared.retired.store(true, Ordering::Relaxed);
        let _ = self.shared.send_line("quit\n");
        let mut child = self.shared.child.lock().unwrap_or_else(PoisonError::into_inner);
        // A moment for an orderly exit (it has nothing left to do), then it is killed and reaped.
        let _ = wait_with_timeout(&mut child, Duration::from_millis(100));
        reap(&mut child);
    }
}

/// The `open` command line for `path`, or why it cannot be sent (a newline would split it).
fn open_line(handle: u64, path: &Path) -> Result<String, String> {
    let Some(text) = path.to_str() else { return Err(format!("{}: path is not UTF-8", path.display())) };
    if text.is_empty() || text.contains(['\n', '\r']) || text.len() > MAX_OPEN_PATH {
        return Err(format!("{}: path cannot be sent to the bridge", path.display()));
    }
    Ok(format!("open {handle} {text}\n"))
}

/// The multi-clip event reader: startup (`ready` or an error) goes to `ready`, then every event
/// is routed to its handle. Pixels on stdout are read right here, in announcement order (they
/// must be, whoever waits for them); shared-memory frames are handed over as a slot lease.
fn demux_loop(
    shared: &Arc<MultiShared>,
    stderr: std::process::ChildStderr,
    mut stdout: ChildStdout,
    ready: mpsc::Sender<Result<(Caps, Option<(u64, u64)>, Option<[u8; 16]>), String>>,
) {
    let mut ready = Some(ready);
    let mut lines = BufReader::new(stderr).lines();
    let why = loop {
        let Some(Ok(line)) = lines.next() else { break "bridge exited".to_string() };
        let Some((handle, event)) = parse_tagged(&line) else { continue };
        if let Some(tx) = ready.take() {
            match event {
                Event::Ready { caps, shm, gpu_uuid, .. } => {
                    let _ = tx.send(Ok((caps, shm, gpu_uuid)));
                }
                Event::Error(e) => {
                    let _ = tx.send(Err(e));
                    break "bridge failed at start-up".to_string();
                }
                _ => ready = Some(tx),
            }
            continue;
        }
        if let Err(why) = route(shared, &mut stdout, handle, event) {
            break why;
        }
    };
    if let Some(tx) = ready {
        let _ = tx.send(Err(why.clone()));
    }
    shared.fail(&why);
}

/// The request queue of `h` while it is open (an open handle without requests has an empty
/// one: a reply for it is unsolicited); `None` once it was closed (late replies are dropped).
fn open_queue(d: &mut Demux, h: u64) -> Option<&mut VecDeque<Pending>> {
    if d.paths.contains_key(&h) { Some(d.queues.entry(h).or_default()) } else { None }
}

/// Route one event; `Err` means the stream can no longer be trusted (the bridge is failed).
fn route(shared: &Arc<MultiShared>, stdout: &mut ChildStdout, handle: Option<u64>, event: Event) -> Result<(), String> {
    match (handle, event) {
        (Some(h), Event::Frame { index, width, height, format, slot, gpu }) => {
            let bytes = frame_bytes(width, height, format)?;
            let arrived = match (gpu, slot) {
                (Some(gpu), _) => {
                    let release = GpuRelease { shared: Arc::downgrade(shared), buffer: gpu.buffer };
                    if matches!(format, FrameFormat::Rgb24 | FrameFormat::Rgba16) {
                        drop(release);
                        Arrived::Failed("bridge sent a GPU frame as rgb24 / rgba16 (GPU frames are rgba8, bgra8 or rgbaf16)".into())
                    } else {
                        Arrived::Gpu { release, gpu, width, height, format }
                    }
                }
                (None, None) => Arrived::Pixels(FrameBytes { data: read_pixels(stdout, bytes)?, width, height, format, gpu: None }),
                (None, Some(slot)) => {
                    let Some(ring) = shared.ring.get() else {
                        return Err(format!("bridge announced shared-memory slot {slot} without a ring"));
                    };
                    shm_offset(ring.layout, slot, bytes)?;
                    Arrived::Slot { lease: SlotLease { shared: Arc::downgrade(shared), slot }, bytes, width, height, format }
                }
            };
            let pending = {
                let mut d = shared.demux.lock().unwrap_or_else(PoisonError::into_inner);
                d.unverified = Some((h, index, width, height));
                match open_queue(&mut d, h) {
                    // a closed handle: the frame (and its slot) is dropped below
                    None => None,
                    Some(q) => match q.pop_front() {
                        Some(p) if p.index == index => Some(p),
                        Some(p) => return Err(format!("bridge out of sync: clip {h} answered frame {index} for {}", p.index)),
                        None => return Err(format!("bridge out of sync: clip {h} sent frame {index} nobody asked for")),
                    },
                }
            };
            shared.served.store(true, Ordering::Relaxed);
            if let Some(p) = pending {
                let _ = p.tx.send(Ok(arrived));
            }
        }
        (Some(h), Event::Verify { index, width, height }) => {
            let mut d = shared.demux.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some((uh, ui, uw, uhh)) = d.unverified.take()
                && (uh, ui) == (h, index)
                && (uw, uhh) != (width, height)
            {
                return Err(format!("bridge dimensions disagree: {width}×{height} after {uw}×{uhh}"));
            }
        }
        (Some(h), Event::FrameError { index, message }) => {
            let pending = {
                let mut d = shared.demux.lock().unwrap_or_else(PoisonError::into_inner);
                match open_queue(&mut d, h) {
                    None => None,
                    Some(q) => match q.pop_front() {
                        Some(p) if p.index == index => Some(p),
                        Some(p) => return Err(format!("bridge out of sync: error for clip {h} frame {index} while waiting for {}", p.index)),
                        None => return Err(format!("bridge out of sync: clip {h} reported frame {index} nobody asked for")),
                    },
                }
            };
            if let Some(p) = pending {
                let _ = p.tx.send(Ok(Arrived::Failed(message)));
            }
        }
        (Some(h), Event::Metadata(meta)) => {
            let tx = shared.demux.lock().unwrap_or_else(PoisonError::into_inner).opens.remove(&h);
            if let Some(tx) = tx {
                let _ = tx.send(Ok(meta));
            }
        }
        (Some(h), Event::Error(message)) => {
            let tx = shared.demux.lock().unwrap_or_else(PoisonError::into_inner).opens.remove(&h);
            match tx {
                Some(tx) => {
                    let _ = tx.send(Err(message));
                }
                None => log::warn!("camera-RAW bridge, clip {h}: {message}"),
            }
        }
        (None, Event::Error(message)) => log::warn!("camera-RAW bridge: {message}"),
        (_, Event::GpuFreed { buffer }) => {
            if let Some(g) = shared.gpu.get() {
                g.forget(buffer);
            }
        }
        // A frame or verify without a handle cannot be routed: the pixels after it would desync.
        (None, Event::Frame { .. }) => return Err("bridge sent a frame without a clip handle".into()),
        _ => {}
    }
    Ok(())
}

/// One clip on a shared [`MultiBridge`]: what a single-clip [`DecodeSession`] is for its own
/// process. Dropping it closes the handle; the bridge serves on for the other clips.
pub struct ClipSession {
    bridge: Arc<MultiBridge>,
    handle: u64,
    meta: Metadata,
    /// Ask for frames in GPU memory ([`ClipSession::set_gpu`]).
    gpu: bool,
    /// This session's own description of the ring file (positional reads need no shared offset).
    ring: Option<File>,
    pub last_timing: (Duration, Duration),
}

impl ClipSession {
    /// Open `path` on `bridge` under `handle`.
    pub fn open(bridge: Arc<MultiBridge>, handle: u64, path: &Path, timeout: Duration) -> Result<Self, OpenFail> {
        let mut meta = bridge.open(handle, path, timeout)?;
        // The bridge-wide capabilities (the metadata repeats them; a missing field is no reason
        // to lose the per-command scale).
        meta.caps = bridge.caps();
        Ok(ClipSession { bridge, handle, meta, gpu: false, ring: None, last_timing: (Duration::ZERO, Duration::ZERO) })
    }

    pub fn caps(&self) -> Caps {
        self.meta.caps
    }

    /// The clip facts the bridge answered `open` with.
    pub fn metadata(&self) -> &Metadata {
        &self.meta
    }

    pub fn uses_shm(&self) -> bool {
        self.bridge.shared.ring.get().is_some()
    }

    pub fn is_alive(&mut self) -> bool {
        self.bridge.is_alive()
    }

    /// The shared bridge this clip is served by.
    pub fn bridge(&self) -> &Arc<MultiBridge> {
        &self.bridge
    }

    /// Ask for the next frames in GPU memory (when the bridge exports them) or in host memory.
    pub fn set_gpu(&mut self, on: bool) {
        self.gpu = on;
    }

    /// One frame: the outer error means this session (or the whole bridge) broke.
    pub fn request_frame(&mut self, index: u64, scale: Option<&str>, timeout: Duration) -> Result<Reply, String> {
        let rx = self.bridge.send_frame(self.handle, index, scale, self.gpu)?;
        self.receive(&rx, Instant::now() + timeout)
    }

    /// [`DecodeSession::request_batch_steered`] on this handle: at most `window` requests
    /// outstanding, no further one once `steer_more` says no or the in-flight budget is spent.
    pub fn request_batch_steered(
        &mut self,
        indices: &[u64],
        scale: Option<&str>,
        window: usize,
        mut steer_more: impl FnMut() -> bool,
        timeout: Duration,
    ) -> Result<Vec<Reply>, String> {
        let window = window.max(1);
        let budget = frame_budget(self.meta.width, self.meta.height);
        let mut out = Vec::with_capacity(indices.len());
        let mut waiting: VecDeque<(mpsc::Receiver<Delivery>, Inflight)> = VecDeque::new();
        let mut sending = true;
        let mut sent = 0;
        while out.len() < indices.len() {
            while sending && sent < indices.len() && waiting.len() < window {
                if sent > 0 && !steer_more() {
                    sending = false;
                    break;
                }
                let Some(share) = Inflight::reserve(budget, sent == 0) else {
                    sending = false;
                    break;
                };
                let Some(&index) = indices.get(sent) else { break };
                waiting.push_back((self.bridge.send_frame(self.handle, index, scale, self.gpu)?, share));
                sent += 1;
            }
            let Some((rx, _share)) = waiting.pop_front() else { break };
            out.push(self.receive(&rx, Instant::now() + timeout)?);
        }
        Ok(out)
    }

    /// Wait for one reply until `deadline` and collect its pixels.
    fn receive(&mut self, rx: &mpsc::Receiver<Delivery>, deadline: Instant) -> Result<Reply, String> {
        let waiting_since = Instant::now();
        let delivery = match rx.recv_timeout(deadline.saturating_duration_since(waiting_since)) {
            Ok(d) => d,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Silent for the whole frame timeout: wedged, for every clip it serves.
                self.bridge.kill("bridge frame timed out");
                return Err("bridge frame timed out".into());
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err("bridge exited".into()),
        };
        let announced = Instant::now();
        let reply = match delivery? {
            Arrived::Failed(message) => Ok(Err(message)),
            Arrived::Pixels(f) => Ok(Ok(f)),
            Arrived::Slot { lease, bytes, width, height, format } => {
                let data = self.read_slot(lease.slot, bytes)?;
                drop(lease); // `release`: the bridge may reuse the slot
                Ok(Ok(FrameBytes { data, width, height, format, gpu: None }))
            }
            Arrived::Gpu { release, gpu, width, height, format } => self.gpu_frame(release, gpu, width, height, format, deadline),
        };
        self.last_timing = (announced.duration_since(waiting_since), announced.elapsed());
        reply
    }

    /// The `bytes` of a frame in shared-memory `slot` (already validated against the layout).
    fn read_slot(&mut self, slot: u32, bytes: usize) -> Result<Vec<u8>, String> {
        let Some(ring) = self.bridge.shared.ring.get() else { return Err("no shared-memory ring".into()) };
        let at = shm_offset(ring.layout, slot, bytes)?;
        if self.ring.is_none() {
            self.ring = Some(File::open(&ring.via).map_err(|e| format!("shared memory: {e}"))?);
        }
        let Some(file) = self.ring.as_mut() else { return Err("no shared-memory ring".into()) };
        file.seek(SeekFrom::Start(at)).map_err(|e| format!("shared-memory seek: {e}"))?;
        read_pixels(file, bytes).map_err(|e| format!("shared memory: {e}"))
    }

    /// A reply in GPU buffer `gpu`: waits (until `deadline`) for the buffer's descriptor if it
    /// has not come yet. A buffer that never arrives or does not hold the frame fails this one
    /// frame (the buffer goes back); the session goes on.
    fn gpu_frame(&self, release: GpuRelease, gpu: GpuRef, width: u32, height: u32, format: FrameFormat, deadline: Instant) -> Result<Reply, String> {
        let memory = self.bridge.shared.gpu.get().and_then(|g| g.wait_for(gpu.buffer, deadline));
        let Some(memory) = memory else {
            if let Some(g) = self.bridge.shared.gpu.get() {
                // it was announced but never came: no more GPU frames from this bridge
                g.close();
            }
            return Ok(Err(format!("GPU buffer {} never arrived", gpu.buffer)));
        };
        if !gpu_layout_fits(width, height, format.bytes_per_pixel() as u32, gpu.pitch, memory.size()) {
            return Ok(Err(format!("GPU frame {width}×{height} (pitch {}) does not fit buffer {} ({} bytes)", gpu.pitch, gpu.buffer, memory.size())));
        }
        Ok(Ok(FrameBytes { data: Vec::new(), width, height, format, gpu: Some(GpuFrame { memory, pitch: gpu.pitch, release: Arc::new(release) }) }))
    }
}

impl Drop for ClipSession {
    fn drop(&mut self) {
        self.bridge.close(self.handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bridge_events() {
        let meta = r#"{"type":"metadata","timecode":"13:37:12:34","fps_num":24000,"fps_den":1001,"width":8192,"height":4320,"frame_count":1200}"#;
        assert_eq!(
            parse_event(meta),
            Some(Event::Metadata(Metadata {
                timecode: "13:37:12:34".into(),
                fps_num: 24000,
                fps_den: 1001,
                width: 8192,
                height: 4320,
                frame_count: 1200,
                protocol: 1,
                caps: Caps::default(),
                shm: None,
                color: None
            }))
        );
        // Protocol 2: version and capabilities (unknown words are ignored).
        let v2 =
            r#"{"type":"metadata","timecode":"00:00:00:00","fps_num":25,"width":16,"height":8,"frame_count":3,"protocol":2,"caps":"scale,frame-errors,warp"}"#;
        let Some(Event::Metadata(m)) = parse_event(v2) else { panic!("metadata v2") };
        assert_eq!(
            (m.protocol, m.caps),
            (2, Caps { scale: true, frame_errors: true, shm: false, multi: false, gpu_export: false, gpu: false, f16: false, rgba16: false })
        );
        assert_eq!(
            parse_event(r#"{"type":"error","index":12,"message":"frame index 12 beyond clip"}"#),
            Some(Event::FrameError { index: 12, message: "frame index 12 beyond clip".into() })
        );
        assert_eq!(
            parse_event(r#"{"type":"frame","index":7,"width":4096,"height":2160}"#),
            Some(Event::Frame { index: 7, width: 4096, height: 2160, format: FrameFormat::Rgb24, slot: None, gpu: None })
        );
        assert_eq!(
            parse_event(r#"{"type":"frame","index":7,"width":4096,"height":2160,"format":"rgba8"}"#),
            Some(Event::Frame { index: 7, width: 4096, height: 2160, format: FrameFormat::Rgba8, slot: None, gpu: None })
        );
        assert_eq!(
            parse_event(r#"{"type":"frame","index":7,"width":4096,"height":2160,"format":"bgra8"}"#),
            Some(Event::Frame { index: 7, width: 4096, height: 2160, format: FrameFormat::Bgra8, slot: None, gpu: None })
        );
        assert_eq!(parse_event(r#"{"type":"frame","index":7,"width":4096,"height":2160,"format":"ycbcr"}"#), None);
        assert_eq!(parse_event(r#"{"type":"verify","index":7,"width":4096,"height":2160}"#), Some(Event::Verify { index: 7, width: 4096, height: 2160 }));
        assert_eq!(parse_event(r#"{"type":"progress","frame":3,"total":1200}"#), Some(Event::Progress { frame: 3, total: 1200 }));
        assert_eq!(parse_event(r#"{"type":"done"}"#), Some(Event::Done));
        assert_eq!(parse_event(r#"{"type":"error","message":"No audio in R3D clip"}"#), Some(Event::Error("No audio in R3D clip".into())));
        assert_eq!(parse_event(r#"{"type":"error","message":"quote \" and backslash \\ out"}"#), Some(Event::Error("quote \" and backslash \\ out".into())));
        assert_eq!(parse_event("not json"), None);
        assert_eq!(parse_event(r#"{"type":"metadata"}"#), None);
        assert_eq!(parse_event(r#"{"type":"frame","index":1}"#), None);
        // Values past u32 are refused, not truncated (`as u32` made 2^32+16 read as 16).
        assert_eq!(parse_event(r#"{"type":"frame","index":7,"width":4294967312,"height":8}"#), None);
        assert_eq!(parse_event(r#"{"type":"verify","index":7,"width":16,"height":4294967296}"#), None);
        assert_eq!(parse_event(r#"{"type":"metadata","fps_num":4294967320,"width":16,"height":8,"frame_count":8}"#), None);
        assert_eq!(parse_event(r#"{"type":"metadata","fps_num":24,"fps_den":4294967296,"width":16,"height":8,"frame_count":8}"#), None);
    }

    #[test]
    fn frame_sizes_are_capped_before_allocating() {
        assert_eq!(frame_bytes(16, 8, FrameFormat::Rgba8), Ok(512));
        assert_eq!(frame_bytes(6144, 3456, FrameFormat::Bgra8), Ok(6144 * 3456 * 4));
        assert_eq!(frame_bytes(12288, 6480, FrameFormat::Rgba8), Ok(12288 * 6480 * 4), "12K is a real camera");
        // `width * height * bpp` overflowed here (u32::MAX² × 4), and `vec![0; n]` aborted on n
        assert!(frame_bytes(u32::MAX, u32::MAX, FrameFormat::Rgba8).is_err());
        assert!(frame_bytes(MAX_FRAME_DIM + 1, 8, FrameFormat::Rgb24).is_err());
        assert!(frame_bytes(MAX_FRAME_DIM, MAX_FRAME_DIM, FrameFormat::Rgba8).is_err(), "1 GiB is over the byte cap");
        assert!(frame_bytes(0, 8, FrameFormat::Rgba8).is_err());
    }

    /// Write `script` as an executable bridge in a fresh test directory; (dir, bridge, clip).
    #[cfg(target_os = "linux")]
    fn script_bridge(name: &str, script: &str) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let bridge = dir.join("bridge");
        std::fs::write(&bridge, script).ok();
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&bridge, std::fs::Permissions::from_mode(0o755));
        }
        let clip = dir.join("clip.braw");
        std::fs::write(&clip, b"placeholder").ok();
        (dir, bridge, clip)
    }

    /// A bridge that announces an absurd frame: the request fails instead of overflowing or
    /// allocating exabytes.
    #[cfg(target_os = "linux")]
    #[test]
    fn absurd_frame_announcements_fail_cleanly() {
        let (dir, bridge, clip) = script_bridge(
            "bigframe",
            r#"#!/bin/sh
echo '{"type":"metadata","timecode":"00:00:00:00","fps_num":24,"fps_den":1,"width":16,"height":8,"frame_count":8}' >&2
while IFS= read -r line; do
    echo '{"type":"frame","index":0,"width":4294967295,"height":4294967295,"format":"rgba8"}' >&2
done
"#,
        );
        let mut session = DecodeSession::spawn(&bridge, &clip, "full", false, Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
        let r = session.request_frame(0, None, Duration::from_secs(5));
        assert!(r.as_ref().is_err_and(|e| e.contains("4294967295")), "got {:?}", r.map(|f| f.map(|f| f.data.len())));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A bridge that announces a frame and then stalls: the request times out (it blocked in
    /// `read_exact` forever) and the bridge is killed.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_stalled_bridge_times_out() {
        let (dir, bridge, clip) = script_bridge(
            "stall",
            r#"#!/bin/sh
echo '{"type":"metadata","timecode":"00:00:00:00","fps_num":24,"fps_den":1,"width":16,"height":8,"frame_count":8}' >&2
IFS= read -r line
echo '{"type":"frame","index":0,"width":16,"height":8,"format":"rgba8"}' >&2
printf 'abc'
sleep 30
"#,
        );
        let mut session = DecodeSession::spawn(&bridge, &clip, "full", false, Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
        let t = Instant::now();
        let r = session.request_frame(0, None, Duration::from_millis(300));
        assert!(r.as_ref().is_err_and(|e| e.contains("timed out")), "got {:?}", r.map(|f| f.map(|f| f.data.len())));
        assert!(t.elapsed() < Duration::from_secs(5), "took {:?}", t.elapsed());
        assert!(!session.is_alive(), "the stalled bridge is killed");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A bridge that reports an error at start-up and lingers: it is killed and reaped, not left
    /// running (and later a zombie).
    #[cfg(target_os = "linux")]
    #[test]
    fn a_bridge_failing_at_start_up_is_reaped() {
        let (dir, bridge, clip) = script_bridge(
            "startfail",
            r#"#!/bin/sh
echo $$ > "$2.pid"
echo '{"type":"error","message":"no GPU"}' >&2
sleep 30
"#,
        );
        let r = DecodeSession::spawn(&bridge, &clip, "full", false, Duration::from_secs(10));
        assert!(matches!(&r, Err(e) if e == "no GPU"), "got {:?}", r.as_ref().err());
        let pid = std::fs::read_to_string(clip.with_extension("braw.pid")).unwrap_or_default();
        let pid = pid.trim();
        assert!(!pid.is_empty(), "the bridge wrote its pid");
        assert!(!Path::new(&format!("/proc/{pid}")).exists(), "bridge {pid} still exists (running or a zombie)");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A `verify` event that disagrees with the frame it trails is caught — at the latest by the
    /// next request (it arrives a moment after the pixels, and `try_recv` right after them
    /// almost always missed it, so the check never fired).
    #[cfg(target_os = "linux")]
    #[test]
    fn a_disagreeing_verify_fails_the_session() {
        let (dir, bridge, clip) = script_bridge(
            "badverify",
            r#"#!/bin/sh
echo '{"type":"metadata","timecode":"00:00:00:00","fps_num":24,"fps_den":1,"width":4,"height":2,"frame_count":8}' >&2
b=$(printf '\001')
out= j=0
while [ $j -lt 32 ]; do out="$out$b"; j=$((j + 1)); done
while IFS= read -r line; do
    i=${line#frame }
    echo "{\"type\":\"frame\",\"index\":$i,\"width\":4,\"height\":2,\"format\":\"rgba8\"}" >&2
    printf '%s' "$out"
    sleep 0.05
    echo "{\"type\":\"verify\",\"index\":$i,\"width\":8,\"height\":2}" >&2
done
"#,
        );
        let mut session = DecodeSession::spawn(&bridge, &clip, "full", false, Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
        let first = session.request_frame(0, None, Duration::from_secs(5));
        let second = session.request_frame(1, None, Duration::from_secs(5));
        let caught = [&first, &second].iter().any(|r| r.as_ref().is_err_and(|e| e.contains("disagree")));
        assert!(caught, "mismatch not detected: {:?} / {:?}", first.map(|f| f.map(|f| f.width)), second.map(|f| f.map(|f| f.width)));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A pipelined bridge answers a batch with replies back-to-back: while the parent still
    /// reads frame N's pixels, the `frame` announcement of N+1 is already on the event channel.
    /// Losing that announcement stalls the stream at one frozen frame — replies must be parked,
    /// never drained past.
    #[cfg(target_os = "linux")]
    #[test]
    fn batch_replies_arriving_together() {
        let dir = std::env::temp_dir().join(format!("filmcraft-rawcam-batch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let script = r#"#!/bin/sh
echo '{"type":"metadata","timecode":"00:00:00:00","fps_num":24,"fps_den":1,"width":4,"height":2,"frame_count":8}' >&2
IFS= read -r line
b=$(printf '\001')
out= j=0
while [ $j -lt 24 ]; do out="$out$b"; j=$((j + 1)); done
echo '{"type":"frame","index":3,"width":4,"height":2}' >&2
printf '%s' "$out"
echo '{"type":"verify","index":3,"width":4,"height":2}' >&2
echo '{"type":"frame","index":4,"width":4,"height":2}' >&2
printf '%s' "$out"
echo '{"type":"verify","index":4,"width":4,"height":2}' >&2
while IFS= read -r l; do : ; done
"#;
        let bridge = dir.join("burst-bridge");
        std::fs::write(&bridge, script).ok();
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&bridge, std::fs::Permissions::from_mode(0o755));
        }
        let input = dir.join("clip.braw");
        std::fs::write(&input, b"placeholder").ok();
        let mut session = DecodeSession::spawn(&bridge, &input, "full", false, Duration::from_secs(10)).ok().unwrap_or_else(|| panic!("session must start"));
        let frames =
            session.request_batch(&[3, 4], None, Duration::from_secs(5)).unwrap_or_else(|e| panic!("a burst of replies must not stall the batch: {e}"));
        assert_eq!(frames.len(), 2);
        let frames: Vec<FrameBytes> = frames.into_iter().map(|r| r.unwrap_or_else(|e| panic!("frame: {e}"))).collect();
        assert_eq!(frames[0].width, 4, "width comes from the frame event");
        assert_eq!(frames[0].height, 2);
        assert_eq!(frames[0].format, FrameFormat::Rgb24);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A protocol 2 bridge answering odd frames with per-frame errors: a batch gets each reply
    /// in its place, and the session stays usable afterwards. An error naming another frame is a
    /// desynchronised stream.
    #[cfg(target_os = "linux")]
    #[test]
    fn per_frame_errors_take_their_place_in_a_batch() {
        let (dir, bridge, clip) = script_bridge(
            "frameerr",
            r#"#!/bin/sh
echo '{"type":"metadata","timecode":"00:00:00:00","fps_num":24,"fps_den":1,"width":2,"height":1,"frame_count":8,"protocol":2,"caps":"scale,frame-errors"}' >&2
while IFS= read -r line; do
    set -- ${line#frame }
    i=$1
    echo "$line" >> "$0.log"
    if [ "$i" = 9 ]; then
        echo '{"type":"error","index":3,"message":"wrong one"}' >&2
    elif [ $((i % 2)) = 1 ]; then
        echo "{\"type\":\"error\",\"index\":$i,\"message\":\"frame $i is corrupt\"}" >&2
    else
        echo "{\"type\":\"frame\",\"index\":$i,\"width\":2,\"height\":1,\"format\":\"rgba8\"}" >&2
        printf 'abcdefgh'
        echo "{\"type\":\"verify\",\"index\":$i,\"width\":2,\"height\":1}" >&2
    fi
done
"#,
        );
        let mut session = DecodeSession::spawn(&bridge, &clip, "full", false, Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
        assert!(session.caps().scale && session.caps().frame_errors);
        let replies =
            session.request_batch(&[0, 1, 2, 3, 4], Some("quarter"), Duration::from_secs(5)).unwrap_or_else(|e| panic!("the session must survive: {e}"));
        let shape: Vec<bool> = replies.iter().map(|r| r.is_ok()).collect();
        assert_eq!(shape, [true, false, true, false, true]);
        assert!(replies.get(1).is_some_and(|r| r.as_ref().is_err_and(|e| e.contains("corrupt"))));
        let log = std::fs::read_to_string(bridge.with_extension("log")).unwrap_or_default();
        assert!(log.lines().all(|l| l.ends_with(" quarter")), "the scale word goes with every command: {log}");
        // Still in sync: the next request works.
        assert!(session.request_frame(6, None, Duration::from_secs(5)).is_ok_and(|r| r.is_ok()));
        // An error for a frame nobody asked for breaks the session.
        assert!(session.request_frame(9, None, Duration::from_secs(5)).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The read-ahead window: no further command once the caller stops wanting more, but the
    /// commands already sent are all answered (at most `window` of them).
    #[cfg(target_os = "linux")]
    #[test]
    fn a_steered_batch_stops_sending_but_reads_what_it_sent() {
        let (dir, bridge, clip) = script_bridge(
            "window",
            r#"#!/bin/sh
echo '{"type":"metadata","timecode":"00:00:00:00","fps_num":24,"fps_den":1,"width":2,"height":1,"frame_count":8}' >&2
while IFS= read -r line; do
    i=${line#frame }
    echo "$i" >> "$0.log"
    echo "{\"type\":\"frame\",\"index\":$i,\"width\":2,\"height\":1,\"format\":\"rgba8\"}" >&2
    printf 'abcdefgh'
done
"#,
        );
        let mut session = DecodeSession::spawn(&bridge, &clip, "full", false, Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
        let mut asked = 0;
        let replies = session
            .request_batch_steered(
                &[0, 1, 2, 3, 4, 5],
                Some("half"),
                3,
                || {
                    asked += 1;
                    asked < 3
                },
                Duration::from_secs(5),
            )
            .unwrap_or_else(|e| panic!("batch: {e}"));
        assert_eq!(replies.len(), 3, "the first command plus two allowed by the steering");
        let log = std::fs::read_to_string(bridge.with_extension("log")).unwrap_or_default();
        assert_eq!(log.lines().collect::<Vec<_>>(), ["0", "1", "2"], "a bridge without the scale capability gets no scale word");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shm_layouts_are_validated_before_use() {
        assert_eq!(shm_layout(6, 85_983_232, 6 * 85_983_232), Ok((6, 85_983_232)));
        assert_eq!(shm_layout(1, 32, 4096), Ok((1, 32)), "a file larger than the ring is fine");
        assert!(shm_layout(0, 32, 64).is_err(), "no slots");
        assert!(shm_layout(MAX_SHM_SLOTS + 1, 32, u64::MAX).is_err(), "too many slots");
        assert!(shm_layout(2, 0, 64).is_err(), "empty slots");
        assert!(shm_layout(4, 32, 64).is_err(), "a ring past the end of its file");
        // slots × slot_bytes overflowed u64 here, and then passed the file-size check
        assert!(shm_layout(16, u64::MAX / 8, u64::MAX).is_err());
        assert!(shm_layout(2, MAX_SHM_BYTES, u64::MAX).is_err(), "over the ring cap");
        let layout = (2, 32);
        assert_eq!(shm_offset(layout, 0, 32), Ok(0));
        assert_eq!(shm_offset(layout, 1, 8), Ok(32));
        assert!(shm_offset(layout, 2, 8).is_err(), "no such slot");
        assert!(shm_offset(layout, u32::MAX, 8).is_err());
        assert!(shm_offset(layout, 1, 33).is_err(), "more than a slot holds");
        assert!(shm_offset((u32::MAX, u64::MAX), u32::MAX - 1, 1).is_err(), "offset overflow");
    }

    #[test]
    fn parses_shm_announcements() {
        let meta = r#"{"type":"metadata","timecode":"00:00:00:00","fps_num":25,"width":16,"height":8,"frame_count":3,"protocol":2,"caps":"scale,frame-errors,shm","shm_slots":6,"shm_slot_bytes":2097152}"#;
        let Some(Event::Metadata(m)) = parse_event(meta) else { panic!("metadata") };
        assert!(m.caps.shm);
        assert_eq!(m.shm, Some((6, 2_097_152)));
        assert_eq!(
            parse_event(r#"{"type":"frame","index":7,"width":4,"height":2,"format":"rgba8","slot":3}"#),
            Some(Event::Frame { index: 7, width: 4, height: 2, format: FrameFormat::Rgba8, slot: Some(3), gpu: None })
        );
        // A slot that is no u32 is refused, not taken for "on stdout" (that would desync).
        assert_eq!(parse_event(r#"{"type":"frame","index":7,"width":4,"height":2,"format":"rgba8","slot":-1}"#), None);
        assert_eq!(parse_event(r#"{"type":"frame","index":7,"width":4,"height":2,"format":"rgba8","slot":4294967296}"#), None);
    }

    /// A shell bridge with the shared-memory transport over the client's `/proc/<pid>/fd/<n>`
    /// path: a two-slot ring of 32 bytes, frame `i` (4×2 RGBA, every byte `i+1`) in slot
    /// `i % 2`; `release` lines are logged. The clip's first word picks a misbehaviour.
    #[cfg(target_os = "linux")]
    const SHM_BRIDGE: &str = r#"#!/bin/sh
shm= input=
while [ $# -gt 0 ]; do
    case "$1" in
        --shm) shm=$2; shift 2;;
        --input) input=$2; shift 2;;
        --debayer) shift 2;;
        *) shift;;
    esac
done
mode=$(head -c 5 "$input")
slots=2
[ "$mode" = short ] && slots=4
[ -n "$shm" ] && truncate -s 64 "$shm"
ring=
[ -n "$shm" ] && ring=",\"shm_slots\":$slots,\"shm_slot_bytes\":32"
echo "{\"type\":\"metadata\",\"timecode\":\"00:00:00:00\",\"fps_num\":24,\"fps_den\":1,\"width\":4,\"height\":2,\"frame_count\":8,\"protocol\":2,\"caps\":\"scale,frame-errors,shm\"$ring}" >&2
while IFS= read -r line; do
    case "$line" in
        release\ *) echo "$line" >> "$input.log"; continue;;
        quit) exit 0;;
    esac
    i=${line#frame }
    i=${i%% *}
    k=$((i % 2))
    [ "$mode" = wild5 ] && k=7
    b=$(printf "\\$(printf '%03o' $((i + 1)))")
    out= j=0
    while [ $j -lt 32 ]; do out="$out$b"; j=$((j + 1)); done
    if [ -n "$shm" ]; then
        printf '%s' "$out" | dd of="$shm" bs=32 seek=$k conv=notrunc status=none
        [ "$mode" = trunc ] && truncate -s 40 "$shm"
        echo "{\"type\":\"frame\",\"index\":$i,\"width\":4,\"height\":2,\"format\":\"rgba8\",\"slot\":$k}" >&2
    else
        echo "{\"type\":\"frame\",\"index\":$i,\"width\":4,\"height\":2,\"format\":\"rgba8\"}" >&2
        printf '%s' "$out"
    fi
    echo "{\"type\":\"verify\",\"index\":$i,\"width\":4,\"height\":2}" >&2
done
"#;

    /// Frames through shared memory: the right bytes from the right slot, every slot released,
    /// and the ring file gone from `/dev/shm` the moment it was handed over.
    #[cfg(target_os = "linux")]
    #[test]
    fn frames_arrive_through_shared_memory() {
        let (dir, bridge, clip) = script_bridge("shm", SHM_BRIDGE);
        std::fs::write(&clip, b"plain").ok();
        let ours = || {
            std::fs::read_dir("/dev/shm")
                .map(|d| {
                    d.filter_map(|e| e.ok())
                        .filter(|e| e.file_name().to_string_lossy().starts_with(&format!("filmcraft-rawcam-{}-", std::process::id())))
                        .count()
                })
                .unwrap_or(0)
        };
        let mut session = DecodeSession::spawn(&bridge, &clip, "full", true, Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
        assert!(session.uses_shm(), "the bridge took the ring");
        assert_eq!(ours(), 0, "the ring file is unlinked at once");
        let replies = session.request_batch(&[0, 1, 2, 5], Some("full"), Duration::from_secs(5)).unwrap_or_else(|e| panic!("batch: {e}"));
        for (reply, i) in replies.into_iter().zip([0_u8, 1, 2, 5]) {
            let f = reply.unwrap_or_else(|e| panic!("frame {i}: {e}"));
            assert_eq!((f.width, f.height, f.data.len()), (4, 2, 32));
            assert!(f.data.iter().all(|&b| b == i + 1), "frame {i} holds its own pixels: {:?}", f.data);
        }
        // The bridge reads the releases after the commands queued before them.
        let log_of = || std::fs::read_to_string(clip.with_extension("braw.log")).unwrap_or_default();
        let t = Instant::now();
        while log_of().lines().count() < 4 && t.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(session);
        let log = log_of();
        assert_eq!(log.lines().collect::<Vec<_>>(), ["release 0", "release 1", "release 0", "release 1"]);
        // Without the offer the same bridge streams over the pipe.
        let mut plain = DecodeSession::spawn(&bridge, &clip, "full", false, Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
        assert!(!plain.uses_shm());
        let f = plain.request_frame(3, None, Duration::from_secs(5)).unwrap_or_else(|e| panic!("{e}")).unwrap_or_else(|e| panic!("{e}"));
        assert!(f.data.iter().all(|&b| b == 4));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Hostile shared-memory bridges fail cleanly: a ring larger than its file at start-up, a
    /// slot outside the ring, a file cut short under an announced frame.
    #[cfg(target_os = "linux")]
    #[test]
    fn hostile_shared_memory_fails_cleanly() {
        let (dir, bridge, clip) = script_bridge("shm-hostile", SHM_BRIDGE);
        std::fs::write(&clip, b"short").ok();
        let r = DecodeSession::spawn(&bridge, &clip, "full", true, Duration::from_secs(10));
        assert!(r.as_ref().is_err_and(|e| e.contains("byte file")), "a 4-slot ring in a 2-slot file: {:?}", r.as_ref().err());
        for (mode, expect) in [("wild5", "slot 7"), ("trunc", "ended after")] {
            std::fs::write(&clip, mode).ok();
            let mut session =
                DecodeSession::spawn(&bridge, &clip, "full", true, Duration::from_secs(10)).unwrap_or_else(|e| panic!("{mode}: session must start: {e}"));
            let r = session.request_frame(1, None, Duration::from_secs(5));
            assert!(r.as_ref().is_err_and(|e| e.contains(expect)), "{mode}: got {:?}", r.map(|f| f.map(|f| f.data.len())));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parses_multi_clip_events() {
        let ready = r#"{"type":"ready","protocol":3,"caps":"scale,frame-errors,shm,multi","shm_slots":5,"shm_slot_bytes":2097152}"#;
        let Some((None, Event::Ready { protocol, caps, shm, .. })) = parse_tagged(ready) else { panic!("ready") };
        assert_eq!((protocol, caps.multi, caps.shm, shm), (3, true, true, Some((5, 2_097_152))));
        assert_eq!(
            parse_tagged(r#"{"type":"frame","handle":7,"index":3,"width":4,"height":2,"format":"rgba8","slot":1}"#),
            Some((Some(7), Event::Frame { index: 3, width: 4, height: 2, format: FrameFormat::Rgba8, slot: Some(1), gpu: None }))
        );
        assert_eq!(
            parse_tagged(r#"{"type":"error","handle":2,"index":9,"message":"bad"}"#),
            Some((Some(2), Event::FrameError { index: 9, message: "bad".into() }))
        );
        assert_eq!(parse_tagged(r#"{"type":"error","handle":2,"message":"cannot open"}"#), Some((Some(2), Event::Error("cannot open".into()))));
        let Some((Some(4), Event::Metadata(m))) = parse_tagged(
            r#"{"type":"metadata","handle":4,"timecode":"00:00:00:00","fps_num":25,"width":16,"height":8,"frame_count":3,"protocol":3,"caps":"scale,multi"}"#,
        ) else {
            panic!("metadata")
        };
        assert!(m.caps.multi && m.caps.scale);
        // A handle that is no number is refused, never read as "no handle" (that would misroute).
        assert_eq!(parse_tagged(r#"{"type":"frame","handle":-1,"index":3,"width":4,"height":2}"#), None);
        assert_eq!(parse_tagged(r#"{"type":"frame","handle":"x","index":3,"width":4,"height":2}"#), None);
        // A key spelled inside an escaped message is no key.
        assert_eq!(parse_tagged(r#"{"type":"error","message":"see \"handle\":5"}"#), Some((None, Event::Error("see \"handle\":5".into()))));
    }

    #[test]
    fn open_lines_cannot_be_split() {
        assert_eq!(open_line(3, Path::new("/m/A 001.braw")), Ok("open 3 /m/A 001.braw\n".into()));
        assert!(open_line(3, Path::new("/m/evil\nframe 0 1.braw")).is_err(), "a newline would inject a command");
        assert!(open_line(3, Path::new("/m/cr\r.braw")).is_err());
        assert!(open_line(3, Path::new("")).is_err());
        assert!(open_line(3, &PathBuf::from("/".repeat(MAX_OPEN_PATH + 1))).is_err());
    }

    #[test]
    fn the_inflight_budget_bounds_background_requests() {
        // Other tests reserve too: measure relative to what is held now.
        let base = inflight_bytes();
        let big = Inflight::reserve(MAX_INFLIGHT_BYTES, true);
        assert!(big.is_some(), "an interactive request always goes");
        assert!(Inflight::reserve(1, false).is_none(), "nothing more for read-ahead while the budget is spent");
        let forced = Inflight::reserve(64, true);
        assert!(forced.is_some());
        drop((big, forced));
        assert!(inflight_bytes() <= base.saturating_add(MAX_INFLIGHT_BYTES), "returned on drop");
        assert_eq!(frame_budget(u32::MAX, u32::MAX), u64::MAX, "saturates");
    }

    /// A multi-clip bridge script: `ready`, then `open` answered with metadata, and the replies
    /// for a pair of frame commands sent in the opposite clip order (handle 1 before handle 0).
    /// The first clip byte selects a misbehaviour.
    #[cfg(target_os = "linux")]
    const MULTI_BRIDGE: &str = r#"#!/bin/sh
shm= sb=0
while [ $# -gt 0 ]; do
    case "$1" in
        --shm) shm=$2; shift 2;;
        --shm-slot-bytes) sb=$2; shift 2;;
        *) shift;;
    esac
done
ring=
if [ -n "$shm" ]; then truncate -s $((2 * sb)) "$shm"; ring=",\"shm_slots\":2,\"shm_slot_bytes\":$sb"; fi
[ -f "$0.noready" ] && { echo '{"type":"metadata","timecode":"00:00:00:00","fps_num":24,"width":4,"height":2,"frame_count":8}' >&2; sleep 30; }
echo "{\"type\":\"ready\",\"protocol\":3,\"caps\":\"scale,frame-errors,shm,multi\"$ring}" >&2
mode=
px() { b=$(printf "\\$(printf '%03o' $(($1 + 1)))"); out= j=0; while [ $j -lt 32 ]; do out="$out$b"; j=$((j + 1)); done; }
reply() { # handle index
    px $2
    if [ -n "$shm" ]; then
        k=$(($2 % 2))
        [ "$mode" = wild ] && k=5
        printf '%s' "$out" | dd of="$shm" bs=$sb seek=$k conv=notrunc status=none
        echo "{\"type\":\"frame\",\"handle\":$1,\"index\":$2,\"width\":4,\"height\":2,\"format\":\"rgba8\",\"slot\":$k}" >&2
    else
        echo "{\"type\":\"frame\",\"handle\":$1,\"index\":$2,\"width\":4,\"height\":2,\"format\":\"rgba8\"}" >&2
        printf '%s' "$out"
    fi
    w=4; [ "$mode" = badv ] && w=8
    echo "{\"type\":\"verify\",\"handle\":$1,\"index\":$2,\"width\":$w,\"height\":2}" >&2
}
pending=
while IFS= read -r line; do
    echo "$line" >> "$0.log"
    set -- $line
    case "$1" in
        open)
            p=${line#open $2 }
            m=$(head -c 4 "$p")
            if [ "$m" = nope ]; then
                echo "{\"type\":\"error\",\"handle\":$2,\"message\":\"cannot open\"}" >&2
            else
                [ "$2" = 0 ] && mode=$m
                echo "{\"type\":\"metadata\",\"handle\":$2,\"timecode\":\"00:00:00:00\",\"fps_num\":24,\"fps_den\":1,\"width\":4,\"height\":2,\"frame_count\":8,\"protocol\":3,\"caps\":\"scale,frame-errors,shm,multi\"}" >&2
            fi;;
        frame)
            case "$mode" in
                stry) echo '{"type":"frame","handle":0,"index":6,"width":4,"height":2,"format":"rgba8"}' >&2; printf '%32s' x; continue;;
                nohd) echo '{"type":"frame","index":6,"width":4,"height":2,"format":"rgba8"}' >&2; continue;;
                stal) sleep 30; continue;;
            esac
            [ "$3" -ge 8 ] && { echo "{\"type\":\"error\",\"handle\":$2,\"index\":$3,\"message\":\"beyond clip\"}" >&2; continue; }
            if [ -z "$pending" ]; then
                pending="$2 $3"
            elif [ "${pending%% *}" = "$2" ]; then
                reply $pending; reply $2 $3; pending=
            else
                reply $2 $3; reply $pending; pending=
            fi;;
        release|close) :;;
        quit) exit 0;;
    esac
done
"#;

    /// Open clip `name` (its first bytes pick the script's mode) under `handle`.
    #[cfg(target_os = "linux")]
    fn clip_session(bridge: &Arc<MultiBridge>, dir: &Path, handle: u64, name: &str, mode: &str) -> ClipSession {
        let clip = dir.join(name);
        std::fs::write(&clip, mode).ok();
        ClipSession::open(bridge.clone(), handle, &clip, Duration::from_secs(5)).unwrap_or_else(|e| panic!("open {name}: {e:?}"))
    }

    /// Two clips on one process: replies come back in the opposite clip order and each lands at
    /// its own handle; per-frame errors stay with their clip; a clip the bridge cannot open fails
    /// alone.
    #[cfg(target_os = "linux")]
    #[test]
    fn multi_clip_replies_reach_their_own_handle() {
        let (dir, bridge, _) = script_bridge("multi-route", MULTI_BRIDGE);
        for shm in [None, Some(32)] {
            let multi = MultiBridge::spawn(&bridge, shm, false, Duration::from_secs(5)).unwrap_or_else(|e| panic!("spawn: {e}"));
            assert!(multi.caps().multi);
            assert_eq!(live_processes(&bridge), 1);
            let mut a = clip_session(&multi, &dir, 0, "a.braw", "plain");
            let mut b = clip_session(&multi, &dir, 1, "b.braw", "plain");
            assert_eq!(a.uses_shm(), shm.is_some());
            // a's request is held until b's arrives; b is answered first, then a.
            let ta = std::thread::spawn(move || {
                let r = a.request_frame(2, Some("full"), Duration::from_secs(5));
                (a, r)
            });
            std::thread::sleep(Duration::from_millis(200));
            let rb = b.request_frame(5, Some("full"), Duration::from_secs(5)).unwrap_or_else(|e| panic!("b: {e}")).unwrap_or_else(|e| panic!("b: {e}"));
            let (mut a, ra) = ta.join().unwrap_or_else(|_| panic!("thread"));
            let ra = ra.unwrap_or_else(|e| panic!("a: {e}")).unwrap_or_else(|e| panic!("a: {e}"));
            assert!(ra.data.iter().all(|&x| x == 3), "a got frame 2: {:?}", ra.data);
            assert!(rb.data.iter().all(|&x| x == 6), "b got frame 5: {:?}", rb.data);
            // A per-frame error is a's alone.
            assert!(a.request_frame(9, Some("full"), Duration::from_secs(5)).is_ok_and(|r| r.is_err()));
            // A clip the bridge refuses fails to open; the others go on.
            let bad = dir.join("bad.braw");
            std::fs::write(&bad, "nope").ok();
            assert!(matches!(ClipSession::open(multi.clone(), 2, &bad, Duration::from_secs(5)), Err(OpenFail::Clip(_))));
            assert!(multi.is_alive());
            // A handle cannot be opened twice.
            assert!(matches!(multi.open(1, &bad, Duration::from_secs(5)), Err(OpenFail::Clip(_))));
            let batch = std::thread::spawn(move || {
                let r = a.request_batch_steered(&[1, 4], Some("full"), 2, || true, Duration::from_secs(5));
                (a, r)
            });
            let (a, r) = batch.join().unwrap_or_else(|_| panic!("thread"));
            let r = r.unwrap_or_else(|e| panic!("batch: {e}"));
            assert!(r.iter().zip([2_u8, 5]).all(|(f, v)| f.as_ref().is_ok_and(|f| f.data.iter().all(|&x| x == v))));
            drop((a, b, multi));
            assert_eq!(live_processes(&bridge), 0, "the process went with the last reference");
        }
        let log = std::fs::read_to_string(bridge.with_extension("log")).unwrap_or_default();
        assert!(log.lines().any(|l| l == "release 1"), "slots are handed back: {log}");
        assert!(log.lines().any(|l| l.starts_with("close ")), "dropped sessions close their handle: {log}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Hostile multi-clip bridges fail cleanly, and every waiting clip learns it: an unsolicited
    /// frame, a frame without a handle, a slot outside the ring, a disagreeing verify, a stall
    /// (killed after the timeout), a `ready` that never comes.
    #[cfg(target_os = "linux")]
    #[test]
    fn hostile_multi_clip_bridges_fail_cleanly() {
        let (dir, bridge, _) = script_bridge("multi-hostile", MULTI_BRIDGE);
        for (mode, shm, expect) in [
            ("stry", None, "nobody asked"),
            ("nohd", None, "without a clip handle"),
            ("wild", Some(32), "slot 5"),
            ("badv", None, "disagree"),
            ("stal", None, "timed out"),
        ] {
            let multi = MultiBridge::spawn(&bridge, shm, false, Duration::from_secs(5)).unwrap_or_else(|e| panic!("{mode}: spawn: {e}"));
            let mut a = clip_session(&multi, &dir, 0, "a.braw", mode);
            let mut b = clip_session(&multi, &dir, 1, "b.braw", "plain");
            let tb = std::thread::spawn(move || b.request_frame(1, Some("full"), Duration::from_secs(10)));
            std::thread::sleep(Duration::from_millis(100));
            let t = Instant::now();
            let ra = a.request_frame(3, Some("full"), Duration::from_millis(700));
            let rb = tb.join().unwrap_or_else(|_| panic!("thread"));
            // The broken stream fails a's request (or, for a late verify, the next one).
            let caught = match &ra {
                Err(e) => e.contains(expect) || (mode == "badv" && e.contains("exited")),
                Ok(_) => mode == "badv",
            };
            assert!(caught, "{mode}: a got {:?}", ra.as_ref().map(|r| r.as_ref().map(|f| f.data.len())));
            assert!(rb.is_err(), "{mode}: b must learn the bridge broke");
            if mode == "badv" {
                let next = a.request_frame(4, Some("full"), Duration::from_secs(2));
                assert!(next.is_err(), "the desynchronised bridge serves no more");
            }
            assert!(t.elapsed() < Duration::from_secs(5), "{mode}: took {:?}", t.elapsed());
            assert!(!multi.is_alive(), "{mode}");
        }
        std::fs::write(bridge.with_extension("noready"), "").ok();
        let r = MultiBridge::spawn(&bridge, None, false, Duration::from_millis(500));
        assert!(r.as_ref().is_err_and(|e| e.contains("ready")), "got {:?}", r.as_ref().err());
        assert_eq!(live_processes(&bridge), 0, "every bridge was reaped");
        std::fs::remove_dir_all(&dir).ok();
    }
}
