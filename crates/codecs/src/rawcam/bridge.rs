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

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Clip facts from a bridge's `metadata` event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metadata {
    pub timecode: String,
    pub fps_num: u32,
    pub fps_den: u32,
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
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

fn json_dims(line: &str, kind: &str) -> Option<Event> {
    let index = json_u64(line, "index")?;
    let width = json_u64(line, "width")? as u32;
    let height = json_u64(line, "height")? as u32;
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
            fps_num: json_u64(line, "fps_num")? as u32,
            fps_den: json_u64(line, "fps_den").unwrap_or(1) as u32,
            width: json_u64(line, "width")? as u32,
            height: json_u64(line, "height")? as u32,
            frame_count: json_u64(line, "frame_count")?,
        })),
        Some(kind @ ("frame" | "verify")) => json_dims(line, kind),
        Some("progress") => Some(Event::Progress { frame: json_u64(line, "frame")?, total: json_u64(line, "total")? }),
        Some("done") => Some(Event::Done),
        Some("error") => Some(Event::Error(json_str(line, "message").unwrap_or_else(|| "unknown bridge error".into()))),
        _ => None,
    }
}

fn base_command(bridge: &Path) -> Command {
    let mut cmd = Command::new(bridge);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd
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
    let mut child = cmd.spawn().map_err(|e| format!("cannot run {}: {e}", bridge.display()))?;
    let stderr = child.stderr.take().ok_or_else(|| "no stderr pipe".to_string())?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || forward_events(stderr, tx));
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
    let _ = child.kill();
    let _ = child.wait();
    result
}

/// `--extract-audio`: decode the clip's audio into `out` (WAV).
pub fn extract_audio(bridge: &Path, input: &Path, out: &Path, timeout: Duration) -> Result<(), String> {
    let mut cmd = base_command(bridge);
    cmd.arg("--input").arg(input).arg("--extract-audio").arg(out);
    let mut child = cmd.spawn().map_err(|e| format!("cannot run {}: {e}", bridge.display()))?;
    let stderr = child.stderr.take().ok_or_else(|| "no stderr pipe".to_string())?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || forward_events(stderr, tx));
    let status = match wait_with_timeout(&mut child, timeout) {
        Some(s) => s,
        None => {
            let _ = child.kill();
            let _ = child.wait();
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

/// A persistent decode session: the bridge process stays alive (`--persistent`), the SDK is
/// initialised once, and frames are requested one `frame <index>` command at a time (or in
/// batches — [`DecodeSession::request_batch`]). Events read past the reply being handled are
/// parked in [`DecodeSession::pending`]: a pipelined bridge emits the next reply's `frame`
/// announcement while this one is still being read, and losing it would stall the stream.
pub struct DecodeSession {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    events: mpsc::Receiver<Event>,
    pending: std::collections::VecDeque<Event>,
}

impl DecodeSession {
    /// Spawn a persistent session at debayer tier `debayer` (the bridge's tier word:
    /// `full`/`half`/`quarter`/`eighth`, `premium` for R3D full quality). Waits for the startup
    /// `metadata` event so a broken bridge fails here, not at the first frame.
    pub fn spawn(bridge: &Path, input: &Path, debayer: &str, timeout: Duration) -> Result<Self, String> {
        let mut cmd = Command::new(bridge);
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd.arg("--input").arg(input).arg("--debayer").arg(debayer).arg("--persistent");
        let mut child = cmd.spawn().map_err(|e| format!("cannot run {}: {e}", bridge.display()))?;
        let stdin = child.stdin.take().ok_or_else(|| "no stdin pipe".to_string())?;
        let stdout = child.stdout.take().ok_or_else(|| "no stdout pipe".to_string())?;
        let stderr = child.stderr.take().ok_or_else(|| "no stderr pipe".to_string())?;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || forward_events(stderr, tx));
        let deadline = Instant::now() + timeout;
        loop {
            match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(Event::Metadata(_)) => return Ok(Self { child, stdin, stdout, events: rx, pending: std::collections::VecDeque::new() }),
                Ok(Event::Error(e)) => return Err(e),
                Ok(_) => {}
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("{}: bridge did not start up (no metadata event)", input.display()));
                }
            }
        }
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

    /// Request one frame: sends `frame <index>`, waits for the `frame` event (its dimensions
    /// size the read), reads the pixels and checks the trailing `verify` event. On any error the
    /// byte stream can no longer be trusted — the caller must discard the session.
    pub fn request_frame(&mut self, index: u64, timeout: Duration) -> Result<FrameBytes, String> {
        let _ = writeln!(self.stdin, "frame {index}");
        self.stdin.flush().map_err(|e| format!("bridge command failed: {e}"))?;
        let deadline = Instant::now() + timeout;
        self.receive_frame(index, deadline)
    }

    /// Request several frames in one go: all commands are written before the first reply is
    /// read, so a pipelined bridge can overlap their reads with the running decode. Replies
    /// arrive (and are returned) in command order. On any error the stream is desynchronised —
    /// the caller must discard the session.
    pub fn request_batch(&mut self, indices: &[u64], timeout: Duration) -> Result<Vec<FrameBytes>, String> {
        self.request_batch_steered(indices, indices.len().max(1), || true, timeout)
    }

    /// [`Self::request_batch`] in chunks of `chunk` commands, consulting `steer_more` before
    /// each further chunk: the frames already requested are committed (their replies must be
    /// read), but no more are sent once the caller no longer wants them — an interactive
    /// request thus waits at most one chunk behind the read-ahead, not a whole batch.
    pub fn request_batch_steered(
        &mut self,
        indices: &[u64],
        chunk: usize,
        mut steer_more: impl FnMut() -> bool,
        timeout: Duration,
    ) -> Result<Vec<FrameBytes>, String> {
        let chunk = chunk.max(1);
        let mut out = Vec::with_capacity(indices.len());
        let mut at = 0;
        while at < indices.len() {
            let end = (at + chunk).min(indices.len());
            let mut cmd = String::new();
            for &index in &indices[at..end] {
                cmd.push_str("frame ");
                cmd.push_str(&index.to_string());
                cmd.push('\n');
            }
            self.stdin.write_all(cmd.as_bytes()).map_err(|e| format!("bridge command failed: {e}"))?;
            self.stdin.flush().map_err(|e| format!("bridge command failed: {e}"))?;
            let deadline = Instant::now() + timeout;
            for &index in &indices[at..end] {
                out.push(self.receive_frame(index, deadline)?);
            }
            at = end;
            if at < indices.len() && !steer_more() {
                break;
            }
        }
        Ok(out)
    }

    /// Wait for the `frame` event of `index` (its dimensions and layout size the read), read
    /// the pixels, check the trailing `verify` event. Events of later replies are parked, not
    /// consumed.
    fn receive_frame(&mut self, index: u64, deadline: Instant) -> Result<FrameBytes, String> {
        let (width, height, format) = loop {
            match self.next_event(deadline) {
                Some(Ok(Event::Frame { index: i, width, height, format })) => {
                    if i != index {
                        return Err(format!("bridge out of sync: answered frame {i} for {index}"));
                    }
                    break (width, height, format);
                }
                Some(Ok(Event::Error(e))) => return Err(e),
                Some(Ok(_)) => {}
                Some(Err(mpsc::RecvTimeoutError::Disconnected)) => return Err("bridge exited".into()),
                Some(Err(mpsc::RecvTimeoutError::Timeout)) | None => return Err("bridge frame timed out".into()),
            }
        };
        let bytes = width as usize * height as usize * format.bytes_per_pixel();
        let mut buf = vec![0u8; bytes];
        self.stdout.read_exact(&mut buf).map_err(|e| format!("bridge frame stream broke: {e}"))?;
        // The `verify` event trails its pixels back-to-back; check the one event that is next
        // and park anything else — it belongs to a later reply, and draining past it would
        // swallow that reply's `frame` announcement and stall the whole batch.
        if let Ok(event) = self.events.try_recv() {
            match event {
                Event::Verify { index: i, width: w, height: h } => {
                    if i != index {
                        return Err(format!("bridge out of sync: verified frame {i} for {index}"));
                    }
                    if (w, h) != (width, height) {
                        return Err(format!("bridge dimensions disagree: {w}×{h} after {width}×{height}"));
                    }
                }
                Event::Error(e) => return Err(e),
                other => {
                    self.pending.push_back(other);
                }
            }
        }
        Ok(FrameBytes { data: buf, width, height, format })
    }
}

impl Drop for DecodeSession {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
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
            Some(Event::Metadata(Metadata { timecode: "13:37:12:34".into(), fps_num: 24000, fps_den: 1001, width: 8192, height: 4320, frame_count: 1200 }))
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
        let frames = session.request_batch(&[3, 4], Duration::from_secs(5)).unwrap_or_else(|e| panic!("a burst of replies must not stall the batch: {e}"));
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].width, 4, "width comes from the frame event");
        assert_eq!(frames[0].height, 2);
        assert_eq!(frames[0].format, FrameFormat::Rgb24);
        std::fs::remove_dir_all(&dir).ok();
    }
}
