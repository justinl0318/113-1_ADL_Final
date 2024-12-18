import json

def align_segments_with_audio(segments, word_timings):
    alignments = []
    ptr1, ptr2 = 0, 0

    segments = segments.split()
    current_word_segment = ""
    ptr2_start = ptr2 # start of current word timing

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

            # Calculate how much of the current_word_segment has been consumed
            consumed_length = current_word_segment.find(segment) + len(segment)
            leftover_length = len(current_word_segment) - consumed_length

            if leftover_length > 0:
                # ff there's leftover, adjust ptr2_start to skip consumed word_timings
                # Calculate how many word_timings have been consumed to form the segment, so that we can correctly skip the consumed timings
                # This is essential to correctly skip the consumed timings
                consumed_word = segment
                consumed_timings = 0
                temp_length = 0

                for i in range(ptr2_start, ptr2 + 1):
                    temp_length += len(word_timings[i]["word"])
                    consumed_timings += 1
                    if temp_length >= len(consumed_word):
                        break

                # Update ptr2 to skip the consumed word_timings
                ptr2_start += consumed_timings - 1
            else:
                # Exact match, move ptr2 forward
                ptr2 += 1
                ptr2_start = ptr2  # Update the start pointer for the next segment

            current_word_segment = ""

            if segment == "遲到":
                print(ptr2_start, ptr2, consumed_length, leftover_length)

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
            a_segmented = transcript["A"]["segmented"]
            a_word_timings = word_entry["word_level_transcriptions"][i]["A"]
            alignments = align_segments_with_audio(a_segmented, a_word_timings)
            transcript["A"]["alignments"] = alignments

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