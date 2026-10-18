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
//! Everything a bridge says is untrusted: dimensions are capped ([`MAX_FRAME_DIM`],
//! [`MAX_FRAME_BYTES`]) before anything is allocated, pixel reads run on a reader thread so a
//! stalled bridge times out instead of hanging its caller, and every child process is killed
//! and reaped on every exit path.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Largest frame side a bridge may announce (16K; the largest camera sensors are 12K).
pub const MAX_FRAME_DIM: u32 = 16384;
/// Largest frame a bridge may send, in bytes (a 12K RGBA frame is ~320 MB).
pub const MAX_FRAME_BYTES: usize = 512 << 20;

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
}

impl Caps {
    fn parse(list: Option<&str>) -> Self {
        let mut caps = Caps::default();
        for word in list.unwrap_or_default().split(',').map(str::trim) {
            match word {
                "scale" => caps.scale = true,
                "frame-errors" => caps.frame_errors = true,
                "shm" => caps.shm = true,
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
}

impl FrameFormat {
    fn parse(s: Option<&str>) -> Option<Self> {
        match s {
            None => Some(Self::Rgb24),
            Some("rgb24") => Some(Self::Rgb24),
            Some("rgba8") => Some(Self::Rgba8),
            Some("bgra8") => Some(Self::Bgra8),
            _ => None,
        }
    }

    /// Bytes one pixel takes in this layout.
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb24 => 3,
            Self::Rgba8 | Self::Bgra8 => 4,
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
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: FrameFormat,
}

/// One NDJSON event from a bridge's stderr.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Metadata(Metadata),
    /// Before a frame's pixels: the frame the following bytes belong to, its dimensions and
    /// pixel layout.
    Frame {
        index: u64,
        width: u32,
        height: u32,
        format: FrameFormat,
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
        "frame" => Some(Event::Frame { index, width, height, format: FrameFormat::parse(json_str(line, "format").as_deref())? }),
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
        })),
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
    /// Where the last reply's time went: waiting for its announcement (decode), then reading
    /// its pixels (transport). Diagnostics only (`FILMCRAFT_RAWCAM_STATS`, the benches).
    pub last_timing: (Duration, Duration),
}

impl DecodeSession {
    /// Spawn a persistent session at debayer tier `debayer` (the bridge's tier word:
    /// `full`/`half`/`quarter`/`eighth`, `premium` for R3D full quality; with [`Caps::scale`]
    /// only the default for commands without a scale word). Waits for the startup `metadata`
    /// event so a broken bridge fails here, not at the first frame. On every failure the bridge
    /// process is killed and reaped.
    pub fn spawn(bridge: &Path, input: &Path, debayer: &str, timeout: Duration) -> Result<Self, String> {
        let mut cmd = Command::new(bridge);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd.arg("--input").arg(input).arg("--debayer").arg(debayer).arg("--persistent");
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
                    return Ok(Self {
                        child,
                        stdin,
                        pixels,
                        events,
                        pending: std::collections::VecDeque::new(),
                        unverified: None,
                        meta,
                        last_timing: (Duration::ZERO, Duration::ZERO),
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
        while out.len() < indices.len() {
            let mut cmd = String::new();
            while sending && sent < indices.len() && sent - out.len() < window {
                if sent > 0 && !steer_more() {
                    sending = false;
                    break;
                }
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
        let (width, height, format) = loop {
            match self.next_event(deadline) {
                Some(Ok(Event::Frame { index: i, width, height, format })) => {
                    if i != index {
                        return Err(format!("bridge out of sync: answered frame {i} for {index}"));
                    }
                    break (width, height, format);
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
        self.pixels.want.send(bytes).map_err(|_| "bridge frame stream closed".to_string())?;
        let data = match self.pixels.got.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(read) => read?,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Stalled mid-frame: kill it (the reader thread then sees the pipe close).
                reap(&mut self.child);
                return Err("bridge frame timed out".into());
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err("bridge frame stream closed".into()),
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
        Ok(Ok(FrameBytes { data, width, height, format }))
    }
}

impl Drop for DecodeSession {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        reap(&mut self.child);
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
                caps: Caps::default()
            }))
        );
        // Protocol 2: version and capabilities (unknown words are ignored).
        let v2 =
            r#"{"type":"metadata","timecode":"00:00:00:00","fps_num":25,"width":16,"height":8,"frame_count":3,"protocol":2,"caps":"scale,frame-errors,warp"}"#;
        let Some(Event::Metadata(m)) = parse_event(v2) else { panic!("metadata v2") };
        assert_eq!((m.protocol, m.caps), (2, Caps { scale: true, frame_errors: true, shm: false }));
        assert_eq!(
            parse_event(r#"{"type":"error","index":12,"message":"frame index 12 beyond clip"}"#),
            Some(Event::FrameError { index: 12, message: "frame index 12 beyond clip".into() })
        );
        assert_eq!(
            parse_event(r#"{"type":"frame","index":7,"width":4096,"height":2160}"#),
            Some(Event::Frame { index: 7, width: 4096, height: 2160, format: FrameFormat::Rgb24 })
        );
        assert_eq!(
            parse_event(r#"{"type":"frame","index":7,"width":4096,"height":2160,"format":"rgba8"}"#),
            Some(Event::Frame { index: 7, width: 4096, height: 2160, format: FrameFormat::Rgba8 })
        );
        assert_eq!(
            parse_event(r#"{"type":"frame","index":7,"width":4096,"height":2160,"format":"bgra8"}"#),
            Some(Event::Frame { index: 7, width: 4096, height: 2160, format: FrameFormat::Bgra8 })
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
        let mut session = DecodeSession::spawn(&bridge, &clip, "full", Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
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
        let mut session = DecodeSession::spawn(&bridge, &clip, "full", Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
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
        let r = DecodeSession::spawn(&bridge, &clip, "full", Duration::from_secs(10));
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
        let mut session = DecodeSession::spawn(&bridge, &clip, "full", Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
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
        let mut session = DecodeSession::spawn(&bridge, &input, "full", Duration::from_secs(10)).ok().unwrap_or_else(|| panic!("session must start"));
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
        let mut session = DecodeSession::spawn(&bridge, &clip, "full", Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
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
        let mut session = DecodeSession::spawn(&bridge, &clip, "full", Duration::from_secs(10)).unwrap_or_else(|e| panic!("session must start: {e}"));
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
}
