//! Smart rendering: unchanged frames of ProRes / DNx media are copied into ProRes / DNx exports as
//! stored, everything else is encoded.

use super::*;
use filmcraft_media::generators::GeneratorSource;
use filmcraft_media::{FrameRequest, Generator, MediaSource, SharedSource};
use filmcraft_project::{ItemKind, Label, MediaClip, MediaRef, Param, ParamValue, SequenceSettings, TrackKind};
use filmcraft_render::SourceMap;

const RATE: FrameRate = FrameRate::FPS_24;

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("fc-smart-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name).to_string_lossy().to_string()
}

fn media_item(p: &mut Project, media: MediaRef, info: filmcraft_media::MediaInfo) -> ItemId {
    p.add_item(
        &info.name.clone(),
        Label::Iris,
        ItemKind::Media(MediaClip {
            media,
            info,
            interpret: Default::default(),
            mark_in: None,
            mark_out: None,
            markers: vec![],
            offline: false,
            proxy: None,
            identity: None,
        }),
        None,
    )
}

/// A 2 s counting leader (every frame different) exported with `settings` (format, profile), not
/// smart rendered, and opened as a source.
fn source_file(name: &str, settings: ExportSettings) -> SharedSource {
    let mut p = Project::new("src");
    let g = GeneratorSource::new(Generator::CountingLeader, 320, 180, RATE, Tick(2 * filmcraft_time::TICKS_PER_SECOND));
    let item = media_item(&mut p, MediaRef::Generator(g.generator.clone()), g.info().clone());
    let seq = p.new_sequence("s", SequenceSettings { width: 320, height: 180, frame_rate: RATE, ..Default::default() }, 1, 0, None);
    let v = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, RATE.tick_of(48)), RATE).unwrap();
    p.sequence_mut(seq).unwrap().video_tracks[0].items.push(v);
    let mut m = SourceMap::default();
    m.0.insert(item, Arc::new(g));
    let path = tmp(name);
    let s = ExportSettings { path: path.clone(), include_audio: false, smart_render: false, ..settings };
    export(&Arc::new(p), seq, &s, &m, &Progress::default()).unwrap();
    filmcraft_codecs::open_bytes(name, std::fs::read(&path).unwrap().into()).unwrap()
}

/// A sequence of `src` (320×180, 24 fps): its frames 5–24 as they are, then its frames 30–39 at
/// 50 % opacity, on V1 (V2 is empty).
fn edit(src: &SharedSource) -> (Project, ItemId, SourceMap) {
    let mut p = Project::new("edit");
    let item = media_item(&mut p, MediaRef::File { path: "src".into() }, src.info().clone());
    let seq = p.new_sequence("s", SequenceSettings { width: 320, height: 180, frame_rate: RATE, ..Default::default() }, 2, 0, None);
    let a = p.make_track_item(item, TrackKind::Video, Tick::ZERO, TimeRange::new(RATE.tick_of(5), RATE.tick_of(20)), RATE).unwrap();
    let mut b = p.make_track_item(item, TrackKind::Video, RATE.tick_of(20), TimeRange::new(RATE.tick_of(30), RATE.tick_of(10)), RATE).unwrap();
    b.effect_mut("opacity").unwrap().params.insert("opacity".into(), Param::new(ParamValue::Float(50.0)));
    let s = p.sequence_mut(seq).unwrap();
    s.video_tracks[0].items.push(a);
    s.video_tracks[0].items.push(b);
    let mut m = SourceMap::default();
    m.0.insert(item, src.clone());
    (p, seq, m)
}

/// Export `p`'s sequence with `settings`; the report and the output opened as a source.
fn run(p: &Project, seq: ItemId, m: &SourceMap, name: &str, settings: ExportSettings) -> (Report, SharedSource) {
    let path = tmp(name);
    let r = export(&Arc::new(p.clone()), seq, &ExportSettings { path: path.clone(), include_audio: false, ..settings }, m, &Progress::default()).unwrap();
    (r, filmcraft_codecs::open_bytes(name, std::fs::read(&path).unwrap().into()).unwrap())
}

fn stored(s: &SharedSource, frame: i64) -> Arc<[u8]> {
    s.coded_sample(RATE.tick_of(frame)).expect("a stored picture").data
}

/// The output's frames 0–19 are the source's frames 5–24 byte for byte; frames 20–29 (50 %
/// opacity) are encoded, so they are not the source's frames 30–39.
fn assert_copied(src: &SharedSource, out: &SharedSource, r: &Report, what: &str) {
    // every frame of the source is different, so comparing them means something
    assert!(stored(src, 5)[..] != stored(src, 6)[..], "{what}");
    assert_eq!(r.frames, 30, "{what}");
    assert_eq!(r.smart_frames, 20, "{what}: frames copied");
    for k in 0..20 {
        assert!(stored(out, k)[..] == stored(src, k + 5)[..], "{what}: output frame {k} is source frame {}", k + 5);
    }
    for k in 20..30 {
        assert!(stored(out, k)[..] != stored(src, k + 10)[..], "{what}: output frame {k} was encoded");
    }
    // and the file plays: every frame decodes
    for k in 0..30 {
        out.video_frame(FrameRequest::full(RATE.tick_of(k))).unwrap();
    }
}

#[test]
fn unchanged_prores_frames_are_copied_into_a_prores_export() {
    let prores = ExportSettings { format: Format::ProRes, prores_profile: "hq".into(), ..Default::default() };
    let src = source_file("src_hq.mov", prores.clone());
    let (p, seq, m) = edit(&src);
    let (r, out) = run(&p, seq, &m, "out_hq.mov", prores.clone());
    assert_copied(&src, &out, &r, "ProRes HQ");
    assert!(out.info().video.as_ref().unwrap().codec.contains("ProRes 422 HQ"));
    // into MXF OP1a too
    let mxf = ExportSettings { format: Format::MxfOp1a, mxf_video_codec: MxfVideoCodec::ProRes, prores_profile: "hq".into(), ..Default::default() };
    let (r, out) = run(&p, seq, &m, "out_hq.mxf", mxf);
    assert_copied(&src, &out, &r, "ProRes HQ in MXF");
}

#[test]
fn unchanged_dnx_frames_are_copied_into_a_dnx_export() {
    for profile in ["hq", "hqx"] {
        let dnx = ExportSettings { format: Format::DnxHr, dnx_profile: profile.into(), ..Default::default() };
        let src = source_file(&format!("src_dnx_{profile}.mov"), dnx.clone());
        let (p, seq, m) = edit(&src);
        let (r, out) = run(&p, seq, &m, &format!("out_dnx_{profile}.mov"), dnx);
        assert_copied(&src, &out, &r, profile);
        let mxf = ExportSettings { format: Format::MxfOp1a, mxf_video_codec: MxfVideoCodec::Dnxhr, dnx_profile: profile.into(), ..Default::default() };
        let (r, out) = run(&p, seq, &m, &format!("out_dnx_{profile}.mxf"), mxf);
        assert_copied(&src, &out, &r, &format!("{profile} in MXF"));
    }
}

#[test]
fn frames_coded_otherwise_are_encoded() {
    let src = source_file("src_std.mov", ExportSettings { format: Format::ProRes, prores_profile: "standard".into(), ..Default::default() });
    let (p, seq, m) = edit(&src);
    // another ProRes profile, DNx, H.264: nothing is copied
    for (name, s) in [
        ("o_hq.mov", ExportSettings { format: Format::ProRes, prores_profile: "hq".into(), ..Default::default() }),
        ("o_dnx.mov", ExportSettings { format: Format::DnxHr, ..Default::default() }),
        ("o.mp4", ExportSettings { format: Format::H264, ..Default::default() }),
    ] {
        let (r, _) = run(&p, seq, &m, name, s);
        assert_eq!(r.smart_frames, 0, "{name}");
    }
    let same = ExportSettings { format: Format::ProRes, prores_profile: "standard".into(), ..Default::default() };
    // switched off, at another size, with an overlay: nothing either
    for (name, s) in [
        ("off.mov", ExportSettings { smart_render: false, ..same.clone() }),
        ("small.mov", ExportSettings { scale: 0.5, ..same.clone() }),
        ("sized.mov", ExportSettings { frame_size: Some((640, 360)), ..same.clone() }),
        ("tc.mov", {
            let mut s = same.clone();
            s.effects.timecode_overlay.enabled = true;
            s
        }),
    ] {
        let (r, _) = run(&p, seq, &m, name, s);
        assert_eq!(r.smart_frames, 0, "{name}");
    }
    // the same coding is copied
    let (r, _) = run(&p, seq, &m, "same.mov", same);
    assert_eq!(r.smart_frames, 20);
}

#[test]
fn a_changed_or_covered_clip_is_encoded() {
    let prores = ExportSettings { format: Format::ProRes, prores_profile: "hq".into(), ..Default::default() };
    let src = source_file("src_hq2.mov", prores.clone());
    let copied = |edit_fn: &dyn Fn(&mut Project, ItemId)| {
        let (mut p, seq, m) = edit(&src);
        edit_fn(&mut p, seq);
        run(&p, seq, &m, "o.mov", prores.clone()).0.smart_frames
    };
    assert_eq!(copied(&|_, _| {}), 20);
    // a standard effect on the first clip
    assert_eq!(
        copied(&|p, seq| p.sequence_mut(seq).unwrap().video_tracks[0].items[0]
            .effects
            .push(filmcraft_project::find_effect("gaussian_blur").unwrap().instance())),
        0
    );
    // Motion that moves it
    assert_eq!(
        copied(&|p, seq| {
            let it = &mut p.sequence_mut(seq).unwrap().video_tracks[0].items[0];
            it.effect_mut("motion").unwrap().params.insert("scale".into(), Param::new(ParamValue::Float(101.0)));
        }),
        0
    );
    // a clip on V2 over the first five frames: those are its own stored pictures
    let over = |p: &mut Project, seq: ItemId, source_in: i64, opacity: Option<f64>, enabled: bool| {
        let media = p.sequence(seq).unwrap().video_tracks[0].items[0].item;
        let mut top = p.make_track_item(media, TrackKind::Video, Tick::ZERO, TimeRange::new(RATE.tick_of(source_in), RATE.tick_of(5)), RATE).unwrap();
        if let Some(o) = opacity {
            top.effect_mut("opacity").unwrap().params.insert("opacity".into(), Param::new(ParamValue::Float(o)));
        }
        let tr = &mut p.sequence_mut(seq).unwrap().video_tracks[1];
        tr.items.push(top);
        tr.enabled = enabled;
    };
    assert_eq!(copied(&|p, seq| over(p, seq, 40, None, true)), 20);
    // half transparent: those five frames are mixed, not stored pictures
    assert_eq!(copied(&|p, seq| over(p, seq, 40, Some(50.0), true)), 15);
    // on a track switched off: the first clip shows
    assert_eq!(copied(&|p, seq| over(p, seq, 40, Some(50.0), false)), 20);
    // a cross dissolve over the cut (frames 18–21): its first two frames are mixed
    assert_eq!(
        copied(&|p, seq| {
            let tr = &mut p.sequence_mut(seq).unwrap().video_tracks[0];
            let (a, b) = (tr.items[0].id, tr.items[1].id);
            tr.transitions.push(filmcraft_project::Transition {
                id: filmcraft_project::TransitionId(1),
                effect: filmcraft_project::find_effect("cross_dissolve").unwrap().instance(),
                start: RATE.tick_of(18),
                duration: RATE.tick_of(4),
                from: Some(a),
                to: Some(b),
                align: Default::default(),
                reverse: false,
            });
        }),
        18
    );
    // an adjustment layer over the first five frames
    assert_eq!(
        copied(&|p, seq| {
            let adj = p.add_item(
                "Adjustment Layer",
                Label::Lavender,
                ItemKind::AdjustmentLayer { width: 320, height: 180, rate: RATE, duration: RATE.tick_of(48) },
                None,
            );
            let it = p.make_track_item(adj, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, RATE.tick_of(5)), RATE).unwrap();
            p.sequence_mut(seq).unwrap().video_tracks[1].items.push(it);
        }),
        15
    );
}

/// The decoded frames of `file` as ffmpeg's framemd5 lists them, in order.
fn framemd5(ff: &std::path::Path, file: &str) -> Vec<String> {
    let out = std::process::Command::new(ff).args(["-v", "error", "-i", file, "-map", "0:v", "-f", "framemd5", "-"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).lines().filter(|l| !l.starts_with('#')).filter_map(|l| l.rsplit(',').next()).map(|s| s.trim().to_string()).collect()
}

/// Media ffmpeg encoded (prores_ks, its DNxHR) is copied where unchanged too, and the result is
/// one stream ffmpeg decodes without a complaint, the copied frames exactly as in the source.
#[test]
fn ffmpeg_media_copied_into_exports_ffmpeg_oracle() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    for (name, codec, settings) in [
        (
            "ff_prores.mov",
            ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
            ExportSettings { format: Format::ProRes, prores_profile: "hq".into(), ..Default::default() },
        ),
        (
            "ff_dnxhr.mov",
            ["-c:v", "dnxhd", "-profile:v", "dnxhr_hq", "-pix_fmt", "yuv422p"],
            ExportSettings { format: Format::DnxHr, dnx_profile: "hq".into(), ..Default::default() },
        ),
    ] {
        let path = tmp(name);
        let made = std::process::Command::new(&ff)
            .args(["-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24", "-t", "2"])
            .args(codec)
            .arg(&path)
            .output()
            .unwrap();
        assert!(made.status.success(), "{name}: {}", String::from_utf8_lossy(&made.stderr));
        let src = filmcraft_codecs::open_bytes(name, std::fs::read(&path).unwrap().into()).unwrap();
        let (p, seq, m) = edit(&src);
        let out_name = format!("out_{name}");
        let (r, _) = run(&p, seq, &m, &out_name, settings);
        assert_eq!(r.smart_frames, 20, "{name}");
        let out_path = tmp(&out_name);
        let dec = std::process::Command::new(&ff).args(["-v", "error", "-xerror", "-i"]).arg(&out_path).args(["-f", "null", "-"]).output().unwrap();
        assert!(dec.status.success() && dec.stderr.is_empty(), "{name}: {}", String::from_utf8_lossy(&dec.stderr));
        let (a, b) = (framemd5(&ff, &path), framemd5(&ff, &out_path));
        assert_eq!(b.len(), 30, "{name}");
        for k in 0..20 {
            assert_eq!(b[k], a[k + 5], "{name}: output frame {k} decodes as source frame {}", k + 5);
        }
    }
}

/// A nested sequence shown as it is passes its stored pictures through; moved, it is encoded.
#[test]
fn a_nested_sequence_shown_as_it_is_is_copied() {
    let prores = ExportSettings { format: Format::ProRes, prores_profile: "hq".into(), ..Default::default() };
    let src = source_file("src_hq3.mov", prores.clone());
    let (mut p, inner, m) = edit(&src);
    let outer = p.new_sequence("outer", SequenceSettings { width: 320, height: 180, frame_rate: RATE, ..Default::default() }, 1, 0, None);
    let nest = p.make_track_item(inner, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, RATE.tick_of(30)), RATE).unwrap();
    p.sequence_mut(outer).unwrap().video_tracks[0].items.push(nest);
    let (r, out) = run(&p, outer, &m, "nest.mov", prores.clone());
    assert_copied(&src, &out, &r, "nested");
    p.sequence_mut(outer).unwrap().video_tracks[0].items[0].effect_mut("motion").unwrap().params.insert("scale".into(), Param::new(ParamValue::Float(50.0)));
    assert_eq!(run(&p, outer, &m, "nest_moved.mov", prores).0.smart_frames, 0);
}

/// A source that counts the frames asked of it.
struct Counting(SharedSource, std::sync::atomic::AtomicUsize);

impl MediaSource for Counting {
    fn info(&self) -> &filmcraft_media::MediaInfo {
        self.0.info()
    }
    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<filmcraft_frame::VideoFrame>> {
        self.1.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.0.video_frame(req)
    }
    fn coded_sample(&self, time: Tick) -> Option<filmcraft_media::CodedSample> {
        self.0.coded_sample(time)
    }
    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        self.0.audio(start, frames, sample_rate)
    }
}

/// Deciding whether to copy a frame decodes nothing and asks no frame of the source (whose cache
/// would keep a coded picture that a CPU render then decodes in place, past the cache's count):
/// only the frames that are rendered are asked for.
#[test]
fn copying_asks_the_source_for_no_frame() {
    let prores = ExportSettings { format: Format::ProRes, prores_profile: "hq".into(), ..Default::default() };
    let src = source_file("src_hq4.mov", prores.clone());
    let (p, seq, _) = edit(&src);
    let counting = Arc::new(Counting(src, Default::default()));
    let mut m = SourceMap::default();
    let item = p.sequence(seq).unwrap().video_tracks[0].items[0].item;
    m.0.insert(item, counting.clone());
    let (r, _) = run(&p, seq, &m, "count.mov", prores.clone());
    assert_eq!(r.smart_frames, 20);
    // the 10 frames at 50 % opacity, and nothing for the 20 copied ones
    assert_eq!(counting.1.load(std::sync::atomic::Ordering::Relaxed), 10);
    // another profile: every frame is rendered, each asked for once
    counting.1.store(0, std::sync::atomic::Ordering::Relaxed);
    let (r, _) = run(&p, seq, &m, "count_lt.mov", ExportSettings { prores_profile: "lt".into(), ..prores });
    assert_eq!(r.smart_frames, 0);
    assert_eq!(counting.1.load(std::sync::atomic::Ordering::Relaxed), 30);
}
