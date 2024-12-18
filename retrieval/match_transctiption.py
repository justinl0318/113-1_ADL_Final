import json
from typing import List, Tuple
import re


def find_best_match(extraction_item, alignments):
    response = json.loads(extraction_item["generated_response"])

    best_word = (
        response.get("phrases", [])[0]
        if response and response.get("phrases", [])
        else ""
    )

    split_length = len(alignments)

    start = 0
    end = split_length
    for i in range(split_length):
        for j in range(i + 1, split_length + 1):
            merged_alignment = "".join(
                [alignment["segment"] for alignment in alignments[i:j]]
            )
            if (best_word in merged_alignment) and j - i < end - start:
                start = i
                end = j
                break

    best_match = {
        "word": best_word,
        "start_index": start,
        "segment_length": end - start,
        "start_time": alignments[start]["start"],
        "end_time": alignments[end - 1]["end"],
    }

    return best_match


if __name__ == "__main__":
    with open("../preprocessed_dataset/aligned_phoneme.json", "r", encoding="utf-8") as f:
        transctiption_data = json.load(f)
    with open("./results_only_keyword.json", "r", encoding="utf-8") as f:
        extraction_data = json.load(f)

    index = 0
    for i, word_entry in enumerate(transctiption_data):
        for transcript in word_entry["transcriptions"]:
            for c in ["A", "B"]:
                alignments = transcript[c]["alignments"]
                best_match = find_best_match(extraction_data[index], alignments)

                transcript[c]["best_match"] = best_match

                index += 1

    output_file = "./keyword_query_gemini.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(transctiption_data, f, ensure_ascii=False, indent=2)
