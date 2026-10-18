//! The drag preview of a drop (`place_preview::preview_place`) shows exactly the clips
//! `timeline.place` then makes: picture, one bar per audio clip, and the tracks the drop adds.

use filmcraft_media::{AudioStreamInfo, MediaInfo, MediaKind, MediaSource};
use filmcraft_project::{AudioChannelMap, AudioChannels, ClipId, ItemId, ItemKind, Label, MediaClip, MediaRef, TrackKind};
use filmcraft_time::FrameRate;
use serde_json::{Value, json};

use crate::Session;
use crate::place_preview::{PlaceGhost, preview_place};

const RATE: FrameRate = FrameRate::FPS_24;

/// A session with a `nv`-video / `na`-audio sequence and no items.
fn session(nv: usize, na: usize) -> Session {
    let mut s = Session::default();
    let mut p = (*s.project).clone();
    let settings = filmcraft_project::SequenceSettings { frame_rate: RATE, ..Default::default() };
    let seq = p.new_sequence("Drop", settings, nv, na, None);
    s.project = std::sync::Arc::new(p);
    s.state.active_sequence = Some(seq);
    s
}

/// Media of 2 s with a picture or not and the given audio streams (channels each).
fn add_media(s: &mut Session, name: &str, video: bool, streams: &[u32], map: Option<AudioChannelMap>) -> ItemId {
    let bars = filmcraft_media::generators::GeneratorSource::new(filmcraft_media::Generator::BarsAndTone, 32, 32, RATE, RATE.tick_of(48));
    let mut info: MediaInfo = bars.info().clone();
    info.name = name.into();
    if !video {
        info.video = None;
        info.kind = MediaKind::AudioOnly;
    }
    info.audio_streams =
        streams.iter().map(|&channels| AudioStreamInfo { sample_rate: 48_000, channels, codec: "pcm_s16le".into(), bits_per_sample: Some(16) }).collect();
    let interpret = filmcraft_project::Interpretation { audio_channels: map, ..Default::default() };
    let clip = MediaClip {
        media: MediaRef::Generator(filmcraft_media::Generator::BarsAndTone),
        info,
        interpret,
        mark_in: None,
        mark_out: None,
        markers: vec![],
        offline: false,
        proxy: None,
        identity: None,
    };
    let mut p = (*s.project).clone();
    let id = p.add_item(name, Label::Iris, ItemKind::Media(clip), None);
    s.project = std::sync::Arc::new(p);
    id
}

fn sorted(mut g: Vec<PlaceGhost>) -> Vec<PlaceGhost> {
    g.sort_by_key(|g| (g.kind == TrackKind::Audio, g.track, g.start));
    g
}

/// Preview `params`, run the drop, and check the clips it made are the preview's bars.
fn drop_matches_preview(s: &mut Session, params: Value) -> Vec<PlaceGhost> {
    let before = s.active_sequence().unwrap().clone();
    let ghosts = sorted(preview_place(s, &params).unwrap());
    let r = s.execute("timeline.place", params).unwrap();
    let q = s.active_sequence().unwrap();
    let made: Vec<PlaceGhost> = r["clips"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let (tid, it) = q.find_item(ClipId(c.as_u64().unwrap())).unwrap();
            let kind = q.track(tid).unwrap().kind;
            let track = q.tracks(kind).iter().position(|t| t.id == tid).unwrap();
            PlaceGhost { kind, track, new_track: track >= before.tracks(kind).len(), start: it.start, duration: it.duration }
        })
        .collect();
    assert_eq!(ghosts, sorted(made), "the preview shows what the drop makes");
    ghosts
}

fn tracks(g: &[PlaceGhost], kind: TrackKind) -> Vec<(usize, bool)> {
    g.iter().filter(|g| g.kind == kind).map(|g| (g.track, g.new_track)).collect()
}

#[test]
fn eight_mono_streams_show_eight_audio_bars_and_the_tracks_the_drop_adds() {
    let mut s = session(3, 3);
    let item = add_media(&mut s, "eight streams", true, &[1; 8], None);
    let g = drop_matches_preview(&mut s, json!({"item": item.0, "track": "V1", "audioTrack": "A1", "time": RATE.tick_of(24).0}));
    assert_eq!(tracks(&g, TrackKind::Video), vec![(0, false)]);
    assert_eq!(tracks(&g, TrackKind::Audio), (0..8).map(|i| (i, i >= 3)).collect::<Vec<_>>(), "A1–A3 exist, A4–A8 are added");
    assert!(g.iter().all(|g| g.start == RATE.tick_of(24) && g.duration == RATE.tick_of(48)), "{g:?}");
    assert_eq!(s.active_sequence().unwrap().audio_tracks.len(), 8);
}

#[test]
fn an_eight_channel_stream_is_one_clip_unless_its_channels_are_mapped_to_mono_clips() {
    let mut s = session(1, 2);
    let one = add_media(&mut s, "8ch", true, &[8], None);
    let g = drop_matches_preview(&mut s, json!({"item": one.0, "track": "V1", "audioTrack": "A2", "time": 0}));
    assert_eq!(tracks(&g, TrackKind::Audio), vec![(1, false)], "one multichannel clip on A2");
    // Modify ▸ Audio Channels ▸ Mono: one clip per channel, from A2 down
    let mono = add_media(&mut s, "8ch mono", true, &[8], Some(AudioChannelMap::for_format(AudioChannels::Mono, 8)));
    let g = drop_matches_preview(&mut s, json!({"item": mono.0, "track": "V1", "audioTrack": "A2", "time": RATE.tick_of(96).0}));
    assert_eq!(tracks(&g, TrackKind::Audio), (1..9).map(|i| (i, i >= 2)).collect::<Vec<_>>());
}

#[test]
fn audio_only_and_video_only_media_show_only_their_own_bars() {
    let mut s = session(2, 2);
    let sound = add_media(&mut s, "sound", false, &[2, 2], None);
    let g = drop_matches_preview(&mut s, json!({"item": sound.0, "track": "V2", "audioTrack": "A2", "time": 0}));
    assert_eq!(tracks(&g, TrackKind::Video), vec![]);
    assert_eq!(tracks(&g, TrackKind::Audio), vec![(1, false), (2, true)]);
    let picture = add_media(&mut s, "picture", true, &[], None);
    let g = drop_matches_preview(&mut s, json!({"item": picture.0, "track": "V2", "audioTrack": "A1", "time": RATE.tick_of(96).0}));
    assert_eq!(tracks(&g, TrackKind::Video), vec![(1, false)]);
    assert_eq!(tracks(&g, TrackKind::Audio), vec![]);
}

#[test]
fn a_drop_on_new_tracks_previews_them_past_the_last_track() {
    let mut s = session(2, 2);
    let item = add_media(&mut s, "four streams", true, &[1; 4], None);
    // below the last audio track: video on V3 (same number), sound on a new A3 and the A4–A6 it needs
    let g = drop_matches_preview(&mut s, json!({"item": item.0, "track": "new", "audioTrack": "new", "time": 0, "insert": true}));
    assert_eq!(tracks(&g, TrackKind::Video), vec![(2, true)]);
    assert_eq!(tracks(&g, TrackKind::Audio), vec![(2, true), (3, true), (4, true), (5, true)]);
    // above the top video track with existing audio tracks: picture on a new V4, sound on A1–A4
    let g = drop_matches_preview(&mut s, json!({"item": item.0, "track": "new", "audioTrack": "A1", "time": RATE.tick_of(96).0}));
    assert_eq!(tracks(&g, TrackKind::Video), vec![(3, true)]);
    assert_eq!(tracks(&g, TrackKind::Audio), vec![(0, false), (1, false), (2, false), (3, false)]);
}

#[test]
fn source_ranges_stream_toggles_and_several_items_match_the_drop() {
    let mut s = session(1, 1);
    let item = add_media(&mut s, "three streams", true, &[2, 1, 1], None);
    // a Source monitor drag of the sound only, half a second of it
    let g = drop_matches_preview(
        &mut s,
        json!({"item": item.0, "track": "V1", "audioTrack": "A1", "time": 0, "sourceIn": RATE.tick_of(6).0, "duration": RATE.tick_of(12).0, "video": false, "audio": true}),
    );
    assert_eq!(tracks(&g, TrackKind::Video), vec![]);
    assert_eq!(tracks(&g, TrackKind::Audio), vec![(0, false), (1, true), (2, true)]);
    assert!(g.iter().all(|g| g.duration == RATE.tick_of(12)));
    // several project items, back to back
    let other = add_media(&mut s, "stereo", true, &[2], None);
    let g = drop_matches_preview(&mut s, json!({"items": [other.0, item.0], "track": "V1", "audioTrack": "A1", "time": RATE.tick_of(48).0}));
    assert_eq!(g.iter().filter(|g| g.kind == TrackKind::Video).map(|g| g.start).collect::<Vec<_>>(), vec![RATE.tick_of(48), RATE.tick_of(96)]);
    assert_eq!(g.iter().filter(|g| g.kind == TrackKind::Audio).count(), 4);
}

#[test]
fn a_drop_the_command_refuses_has_no_preview() {
    let mut s = session(1, 1);
    let item = add_media(&mut s, "a/v", true, &[2], None);
    let mut p = (*s.project).clone();
    let q = p.sequence_mut(s.state.active_sequence.unwrap()).unwrap();
    q.audio_tracks[0].locked = true;
    s.project = std::sync::Arc::new(p);
    let params = json!({"item": item.0, "track": "V1", "audioTrack": "A1", "time": 0});
    assert!(preview_place(&s, &params).is_err());
    assert!(s.execute("timeline.place", params).is_err());
    // hostile parameters give errors, not panics
    for bad in [
        json!({"item": u64::MAX, "track": "V1"}),
        json!({"item": item.0, "track": "V99"}),
        json!({"items": "x"}),
        json!({"items": []}),
        json!({"item": item.0, "time": -5}),
        json!({"item": item.0, "duration": i64::MAX, "time": i64::MAX}),
    ] {
        let _ = preview_place(&s, &bad);
    }
}

#[test]
fn audio_that_arrives_later_shows_up_in_the_next_preview() {
    let mut s = session(1, 4);
    let item = add_media(&mut s, "late", true, &[], None);
    let params = json!({"item": item.0, "track": "V1", "audioTrack": "A1", "time": 0});
    assert_eq!(tracks(&preview_place(&s, &params).unwrap(), TrackKind::Audio), vec![]);
    let mut p = (*s.project).clone();
    if let Some(ItemKind::Media(m)) = p.item_mut(item).map(|i| &mut i.kind) {
        m.info.audio_streams = vec![AudioStreamInfo { sample_rate: 48_000, channels: 1, codec: "pcm".into(), bits_per_sample: None }; 3];
    }
    s.project = std::sync::Arc::new(p);
    assert_eq!(tracks(&drop_matches_preview(&mut s, params), TrackKind::Audio), vec![(0, false), (1, false), (2, false)]);
}

#[test]
fn a_sequence_edited_in_as_its_clips_previews_every_clip_on_its_lane() {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let demo = s.state.active_sequence.unwrap();
    s.execute("file.newSequence", json!({"name": "Into", "video": 1, "audio": 1})).unwrap();
    s.state.sequences_as_clips = true;
    let g = drop_matches_preview(&mut s, json!({"item": demo.0, "track": "V1", "audioTrack": "A1", "time": 0}));
    assert!(g.iter().any(|g| g.kind == TrackKind::Audio && g.new_track), "the demo's audio lanes need tracks the drop adds: {g:?}");
    assert!(g.len() > 2);
}
