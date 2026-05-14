#TESTING
import json
from pathlib import Path
## END


### Assigning individual whisper segments speakers

def overlap(w_start, w_end, d_start, d_end):
    return max(0.0, min(w_end, d_end) - max(w_start, d_start))

def assign_speaker(whisper_seg, diar_segments):
    probable_speaker = "UNKNOWN"
    probable_overlap = 0.0

    for seg in diar_segments:
        #Skipping cases in which overlap would be <0.0 or 0.0 after overlap()
        if min(seg["end"],whisper_seg.end) < max(seg["start"],whisper_seg.start):
            continue
        new_ov = overlap(whisper_seg.start,whisper_seg.end,seg["start"],seg["end"])
        #Optimization - just one area of overlaps possible, when it no longer overlaps we stop searching
        if probable_overlap and not new_ov:
            break
        if new_ov > probable_overlap:
            probable_overlap = new_ov
            probable_speaker = seg["speaker"]

    return probable_speaker

# For every whisper segment appends speaker to list of speakers by order of appearing
def assign_segments_speakers(whisper_segs, diar_segs):
    speakers = []
    #TESTING
    text = ""
    current_file = Path(__file__).resolve()
    json_path = current_file.parent.parent /"dev" / "dis_temp.json"
    with open(json_path, "r", encoding="utf-8") as file:
        diar_segs = json.load(file)
    ## END
    for seg in whisper_segs:
        speakers.append(assign_speaker(seg, diar_segs))
    return speakers