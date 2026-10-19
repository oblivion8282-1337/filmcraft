//! What a `timeline.place` will put where, without editing anything. The Timeline draws these
//! bars while a project item or a Source monitor range is dragged over it, so the preview shows
//! the clips the drop makes: the picture on its track and one bar per audio clip (one per
//! Modify ▸ Audio Channels clip and per further container stream, on the tracks below the
//! destination), including the tracks the drop adds. It resolves the parameters with the same
//! code as the command ([`crate::commands::resolve_place`],
//! [`crate::commands::item_audio_specs`], [`crate::sequence_tools::sequence_lanes`]), so the
//! preview and the result cannot drift apart.

use filmcraft_project::TrackKind;
use filmcraft_time::{Tick, TimeRange};
use serde::Serialize;
use serde_json::Value;

use crate::commands::{PlaceDest, bad, item_audio_specs, resolve_place, sequence_time_p};
use crate::sequence_tools::{MAX_TRACKS, sequence_lanes, wants_new_track};
use crate::{EngineError, Result, Session};

/// One clip a `timeline.place` would make.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlaceGhost {
    pub kind: TrackKind,
    /// Index among the sequence's tracks of `kind` (V1 / A1 = 0).
    pub track: usize,
    /// The sequence has no such track yet: the drop adds it.
    pub new_track: bool,
    pub start: Tick,
    pub duration: Tick,
}

impl PlaceGhost {
    pub fn end(&self) -> Tick {
        Tick(self.start.0.saturating_add(self.duration.0))
    }
}

/// The clips `timeline.place` with parameters `p` would make in the active sequence, or the error
/// it would give. Takes the same parameters as the command (`item` or `items`).
pub fn preview_place(s: &Session, p: &Value) -> Result<Vec<PlaceGhost>> {
    match p.get("items") {
        Some(items) => preview_items(s, p, items),
        None => preview_one(s, p),
    }
}

/// `items`: back to back from the drop time, as `place_items` lays them out.
fn preview_items(s: &Session, p: &Value, items: &Value) -> Result<Vec<PlaceGhost>> {
    let ids: Vec<u64> = items
        .as_array()
        .ok_or_else(|| bad("timeline.place", "`items` must be a list of item ids"))?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| bad("timeline.place", "`items` must be a list of item ids")))
        .collect::<Result<_>>()?;
    if p.get("item").is_some() || p.get("sourceIn").is_some() || p.get("duration").is_some() {
        return Err(bad("timeline.place", "give `item` or `items`, and no `sourceIn` or `duration` with `items`"));
    }
    if ids.is_empty() || ids.len() > crate::commands::MAX_PLACE_ITEMS {
        return Err(bad("timeline.place", format!("`items` needs 1 to {} item ids", crate::commands::MAX_PLACE_ITEMS)));
    }
    let mut at = sequence_time_p(s, p, "", "timeline.place")?.unwrap_or(s.playhead());
    let mut out = Vec::new();
    for item in ids {
        let mut one = p.clone();
        if let Some(o) = one.as_object_mut() {
            for k in ["items", "time", "frame", "seconds"] {
                o.remove(k);
            }
            o.insert("item".into(), item.into());
            o.insert("time".into(), at.0.into());
        }
        // a "new" track stays the same index for the next items: the first one added it there
        let ghosts = preview_one(s, &one)?;
        if let Some(end) = ghosts.iter().map(PlaceGhost::end).max() {
            at = end;
        }
        out.extend(ghosts);
    }
    Ok(out)
}

fn preview_one(s: &Session, p: &Value) -> Result<Vec<PlaceGhost>> {
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let r = resolve_place(s, p)?;
    let (new_v, new_a) = (r.video == Some(PlaceDest::NewVideo), r.audio == Some(PlaceDest::NewAudio));
    if wants_new_track(p) && !new_v && !new_a {
        return Err(bad("timeline.place", "no new track to add: the item has no picture or sound for it"));
    }
    if (new_v && q.video_tracks.len() >= MAX_TRACKS) || (new_a && q.audio_tracks.len() >= MAX_TRACKS) {
        return Err(bad("sequence.addTracks", format!("a sequence has at most {MAX_TRACKS} tracks of a kind")));
    }
    let index = |kind: TrackKind, d: Option<PlaceDest>| -> Result<Option<usize>> {
        match d {
            None => Ok(None),
            Some(PlaceDest::NewVideo) => Ok(Some(q.video_tracks.len())),
            Some(PlaceDest::NewAudio) => Ok(Some(q.audio_tracks.len())),
            Some(PlaceDest::Track(t)) => {
                q.tracks(kind).iter().position(|x| x.id == t).map(Some).ok_or_else(|| EngineError::Other("no such destination track".into()))
            }
        }
    };
    let (v, a) = (index(TrackKind::Video, r.video)?, index(TrackKind::Audio, r.audio)?);
    place_layout(s, r.item, r.range, r.at, v, a)
}

/// The clips `place_item` makes for `item`'s `range` at `at`, its picture from video track
/// `vfirst` and its sound from audio track `afirst` (indices; a track at or past the end of its
/// list is one the drop adds).
pub(crate) fn place_layout(
    s: &Session,
    item: filmcraft_project::ItemId,
    range: TimeRange,
    at: Tick,
    vfirst: Option<usize>,
    afirst: Option<usize>,
) -> Result<Vec<PlaceGhost>> {
    let q = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let pi = s.project.item(item).ok_or_else(|| bad("timeline.place", "no such item"))?;
    let ghost =
        |kind: TrackKind, track: usize, start: Tick, duration: Tick| PlaceGhost { kind, track, new_track: track >= q.tracks(kind).len(), start, duration };
    let mut out = Vec::new();
    if s.state.sequences_as_clips
        && let Some(src) = pi.as_sequence().filter(|src| src.multicam.is_none())
    {
        let lanes = sequence_lanes(src, range, vfirst, afirst);
        if lanes.items.is_empty() {
            return Err(EngineError::Other("no destination track for this sequence's clips (check source patching)".into()));
        }
        for (kind, lane, it) in &lanes.items {
            let base = lanes.base.get(&(*kind == TrackKind::Audio)).copied().unwrap_or(0);
            out.push(ghost(*kind, base.saturating_add(*lane), at + (it.start - range.start), it.duration));
        }
    } else {
        let (video, audio) = s.project.source_streams(item);
        let vfirst = vfirst.filter(|_| video);
        let afirst = afirst.filter(|_| audio);
        if vfirst.is_none() && afirst.is_none() {
            return Err(EngineError::Other("no destination track for this media (check source patching)".into()));
        }
        // `make_track_item`: on the sequence's frame grid, at least a frame long
        let rate = q.settings.frame_rate;
        let dur = rate.snap_nearest(range.duration).max(rate.frame_duration());
        if let Some(v) = vfirst {
            out.push(ghost(TrackKind::Video, v, at, dur));
        }
        if let Some(first) = afirst {
            let specs = item_audio_specs(&s.project, item);
            if first.saturating_add(specs.len()) > MAX_TRACKS {
                return Err(bad("Place Clip", format!("a sequence has at most {MAX_TRACKS} audio tracks")));
            }
            out.extend(specs.iter().map(|spec| ghost(TrackKind::Audio, first.saturating_add(spec.track_offset), at, dur)));
        }
    }
    // a clip on a locked track fails the whole edit
    if out.iter().any(|g| q.tracks(g.kind).get(g.track).is_some_and(|t| t.locked)) {
        return Err(filmcraft_edit::EditError::Locked.into());
    }
    Ok(out)
}
