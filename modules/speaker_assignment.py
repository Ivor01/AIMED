### Assigning individual whisper segments speakers

def overlap(w_start, w_end, d_start, d_end):
    return max(0.0, min(w_end, d_end) - max(w_start, d_start))

def assign_speaker(whisper_seg, diar_segments):
    probable_speaker = "UNKNOWN"
    probable_overlap = 0.0

    for seg in diar_segments:
        #Skipping cases in which overlap would be <0.0 or 0.0 after overlap()
        if min(seg["end"],whisper_seg["end"]) > max(seg["start"],whisper_seg["start"]):
            continue
        new_ov = overlap(whisper_seg["start"],whisper_seg["end"],seg["start"],seg["end"])
        #Optimization - just one area of overlaps possible, when it no longer overlaps we stop searching
        if probable_overlap and not new_ov:
            break  
        if new_ov > probable_overlap:
            probable_overlap = new_ov
            probable_speaker = seg["speaker"]

    return probable_speaker


