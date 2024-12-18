import sqlite3
from pypinyin import pinyin, Style
from difflib import SequenceMatcher
import json

def word_to_pinyin(word):
    phones = pinyin(word, style=Style.TONE3)
    return ' '.join([p[0] for p in phones])

def get_similarity(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

def query_word(query_word, threshold=0.8):
    conn = sqlite3.connect('words.db')
    c = conn.cursor()
    
    query_phoneme = word_to_pinyin(query_word)
    
    # Get all words and phonemes
    c.execute('SELECT word, phoneme FROM words')
    results = c.fetchall()
    
    best_match = None
    best_score = 0
    
    for word, phoneme in results:
        score = get_similarity(query_phoneme, phoneme)
        if score > best_score:
            best_score = score
            best_match = (word, phoneme)
    
    conn.close()
    
    if best_match:
        return {
            'query_word': query_word,
            'query_phoneme': query_phoneme,
            'matched_word': best_match[0],
            'matched_phoneme': best_match[1],
            'score': best_score,
            'is_active': best_score >= threshold
        }
    return None

def get_best_match(segmented, alignments, LEN, max_segment_length=None):
    if not max_segment_length: # set default value
        max_segment_length = LEN

    best_score, best_match = 0, None

    # iterate through all possible lengths starting from 1
    for currlen in range(1, max_segment_length + 1):
        for start_index in range(0, LEN - currlen + 1):
            start_time = alignments[start_index]["start"]
            end_time = alignments[start_index + currlen - 1]["end"]
            concated_word = "".join(segmented[start_index:start_index + currlen])

            # print(f"Word: {concated_word}, Start: {start_time}, End: {end_time}")
            result = query_word(concated_word)
            if result and result["score"] > best_score:
                best_score = result["score"]
                best_match = {
                    "word": concated_word,
                    "start_index": start_index,
                    "segment_length": currlen,
                    "start_time": start_time,
                    "end_time": end_time,
                    "score": best_score
                }

            # if result:
            #     print(f"Query result: {result}")

    return best_match

def process_data(data):
    for i, word_entry in enumerate(data):
        for transcript in word_entry["transcriptions"]:
            for c in ["A", "B"]:
                segmented = transcript[c]["segmented"].split()
                alignments = transcript[c]["alignments"]
                LEN = len(segmented)
                print(f"word: {word_entry['word']}, speaker: {c}, index{i}")
                best_match = get_best_match(segmented, alignments, LEN, max_segment_length=3)
                # print(f"Best match: {best_match}")
                transcript[c]["best_match"] = best_match

    return data

# Example usage
if __name__ == "__main__":
    input_file = "./adl_dataset/aligned_phoneme.json"
    output_file = "./adl_dataset/keyword_query.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = process_data(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)


# # RAG output:
# 校正後的句子，跟活網用詞的定義 text version
# 原本user的語音，跟活網用詞的定義 text version

