//! `MediaSource::coded_sample` hands out the stored bytes of exactly the picture `video_frame`
//! shows at the same media time (smart rendering copies them into exports), for ProRes and DNx in
//! QuickTime and MXF, and nothing for other codecs. ffmpeg writes the files.

mod common;

use std::path::{Path, PathBuf};

use common::*;
use filmcraft_frame::PixelData;
use filmcraft_media::{FrameRequest, SharedSource};
use filmcraft_time::Tick;

const V25: &[&str] = &["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25", "-t", "1"];

fn make(ff: &Path, name: &str, codec: &[&str]) -> PathBuf {
    let mut args: Vec<&str> = V25.to_vec();
    args.extend_from_slice(codec);
    fixture(ff, name, &args).unwrap_or_else(|| panic!("could not generate {name}"))
}

fn open(p: &Path) -> SharedSource {
    filmcraft_codecs::open_bytes(p.file_name().unwrap().to_str().unwrap(), bytes(p)).unwrap()
}

/// At every frame's start and middle, and past the end, the stored picture is the one the
/// decoder shows: a GPU compositor gets that picture still coded, as the sample's bytes. The file
/// stores `pictures` different ones.
fn check(file: &Path, pictures: usize) {
    let src = open(file);
    let rate = src.info().frame_rate();
    let n = rate.frame_at(src.info().duration);
    assert!(n >= 20, "{}: {n} frames", file.display());
    let half = Tick(rate.frame_duration().0 / 2);
    let mut seen = std::collections::HashSet::new();
    for k in 0..n + 3 {
        for t in [rate.tick_of(k), rate.tick_of(k) + half] {
            let stored = src.coded_sample(t).unwrap_or_else(|| panic!("{}: no stored picture at {t:?}", file.display()));
            let shown = filmcraft_media::cancel::with_gpu_frames(true, || src.video_frame(FrameRequest::full(t))).unwrap();
            let PixelData::Coded(c) = &shown.data else { panic!("{}: a coded frame", file.display()) };
            assert!(c.data() == &stored.data[..], "{}: frame {k} at {t:?}", file.display());
            // and it is described as the decoder shows it
            assert_eq!((stored.width, stored.height, stored.alpha, stored.color), (shown.width, shown.height, false, shown.color), "{}", file.display());
            seen.insert(stored.data.clone());
        }
    }
    // every frame of the test pattern is different, so the comparison above means something
    assert_eq!(seen.len(), pictures, "{}", file.display());
}

#[test]
fn prores_and_dnx_stored_pictures_are_the_pictures_shown() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let prores = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"];
    let dnx = ["-c:v", "dnxhd", "-profile:v", "dnxhr_lb", "-pix_fmt", "yuv422p"];
    for (name, codec) in [("cs_prores.mov", &prores[..]), ("cs_dnxhr.mov", &dnx[..]), ("cs_prores.mxf", &prores[..]), ("cs_dnxhr.mxf", &dnx[..])] {
        let f = make(&ff, name, codec);
        check(&f, 25);
        let s = open(&f).coded_sample(Tick::ZERO).unwrap();
        let want = if name.contains("prores") { (filmcraft_frame::CodedFormat::ProRes, *b"apch") } else { (filmcraft_frame::CodedFormat::Dnx, *b"AVdh") };
        assert_eq!(s.format, want.0, "{name}");
        if want.0 == filmcraft_frame::CodedFormat::ProRes {
            assert_eq!(&s.fourcc, &want.1, "{name}: profile 3 is HQ");
        }
    }
}

#[test]
fn a_late_starting_prores_track_hands_out_the_picture_shown() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    // an edit list that starts the video 0.2 s in
    let f = fixture(
        &ff,
        "cs_prores_offset.mov",
        &[
            "-itsoffset",
            "0.2",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=25",
            "-t",
            "1",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
        ],
    )
    .expect("fixture");
    // 0.8 s of pictures; the first one is also shown before the video starts
    check(&f, 20);
}

#[test]
fn other_codecs_have_no_stored_pictures() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let f = make(&ff, "cs_h264.mp4", &["-c:v", "libx264", "-pix_fmt", "yuv420p"]);
    assert!(open(&f).coded_sample(Tick::ZERO).is_none());
}
