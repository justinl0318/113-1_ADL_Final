import json
from pypinyin import pinyin, Style
from phonemizer import phonemize
import ipdb

def convert_pinyin(segmented_text):
    word_list = segmented_text.split()
    phoneme_list = []
    for word in word_list:
        phonemes = pinyin(word, style=Style.TONE3, strict=False)
        # ipdb.set_trace()
        phoneme_list.append(''.join([p[0] for p in phonemes]))
    return ' '.join(phoneme_list)

def convert_phonemizer(segmented_text):
    word_list = segmented_text.split()
    phoneme_list = []
    for word in word_list:
        phonemes = phonemize(
            word, 
            language='zh-cn',
            backend='espeak',
            preserve_punctuation=True,
            strip=True
        )
        phoneme_list.append(phonemes.strip())
    return ' '.join(phoneme_list)

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result = []
    for entry in data:
        processed_entry = {
            "word": entry["word"],
            "transcriptions": []
        }
        
        for trans in entry["transcriptions"]:
            processed_trans = {"A": [], "B": []}
            
            for speaker in ["A", "B"]:
                for utt in trans[speaker]:
                    processed_utt = {
                        "start": utt["start"],
                        "end": utt["end"],
                        "text": utt["text"],
                        "segmented": utt["segmented"],
                        "pinyin_phoneme": convert_pinyin(utt["segmented"])
                        # "esp_phoneme": convert_phonemizer(utt["segmented"])
                    }
                    processed_trans[speaker].append(processed_utt)
            
            processed_entry["transcriptions"].append(processed_trans)
        
        result.append(processed_entry)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "./adl_dataset/processed_transcription.json"
    output_file = "./adl_dataset/segmented_phoneme.json"
    process_file(input_file, output_file)