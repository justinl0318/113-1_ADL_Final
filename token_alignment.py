import json

def align_segments_with_audio(segments, word_timings):
    alignments = []
    ptr1, ptr2 = 0, 0

    segments = segments.split()
    current_word_segment = ""
    ptr2_start = ptr2 # start of current word timing
    ptr2_start_index = 0 # the starting index of word_timings[ptr2_start]

    while ptr1 < len(segments) and ptr2 < len(word_timings):
        segment = segments[ptr1]
        timing = word_timings[ptr2]

        # add current word to build current_word_segment
        current_word_segment += timing["word"]

        # match found
        if segment in current_word_segment:
            alignments.append({
                "segment": segment,
                "start": word_timings[ptr2_start]["start"],
                "end": timing["end"] 
            })

            ptr1 += 1 # move to next segment

            if len(segment) == len(current_word_segment[ptr2_start_index:]): # exact match
                ptr2 += 1 # advance ptr2
                ptr2_start = ptr2
                ptr2_start_index = 0
            elif len(segment) < len(current_word_segment[ptr2_start_index:]): # partial match, leftover is used at the next starting index
                ptr2_start_index = len(current_word_segment[ptr2_start_index:]) - len(segment)
                ptr2_start = ptr2

            current_word_segment = ""

        # reset if we've accumulated too much words without a match
        elif ptr2 - ptr2_start > 3:
            ptr2 = ptr2_start + 1
            ptr2_start = ptr2
            ptr2_start_index = 0
            current_word_segment = ""

        # no match, move to next word timing
        else:
            ptr2 += 1

        # handle remaining segments
        if ptr2 >= len(word_timings) and ptr1 < len(segments):
            last_timing = word_timings[-1]
            alignments.append({
                "segment": segments[ptr1],
                "start": last_timing["start"],
                "end": last_timing["end"]
            })  
            ptr1 += 1

    return alignments

def process_json_file(data):
    for word_entry in data:
        for i, transcript in enumerate(word_entry["transcriptions"]):
            b_segmented = transcript["B"]["segmented"]
            b_word_timings = word_entry["word_level_transcriptions"][i]["B"]

            alignments = align_segments_with_audio(b_segmented, b_word_timings)

            transcript["B"]["alignments"] = alignments

    return data

if __name__ == "__main__":
    input_file = "./adl_dataset/segmented_phoneme.json"
    output_file = "./adl_dataset/aligned_phoneme.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = process_json_file(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)



