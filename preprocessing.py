import json
import jieba

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_text(text):
    return list(jieba.cut(text))

def process_utterances(utterances):
    """Process a list of utterances."""
    processed = {}
    processed["text"] = utterances
    processed["segmented"] = " ".join(process_text(utterances))
    return processed

def process_transcriptions(data):
    """Process all transcriptions in the dataset."""
    processed_data = []
    
    for word_entry in data:
        processed_transcriptions = []
        
        for transcript in word_entry["sentences"]:
            # process both speakers' utterances
            processed_dialogue = {
                "A": process_utterances(transcript["A"]),
                "B": process_utterances(transcript["B"])
            }
            processed_transcriptions.append(processed_dialogue)
        
        processed_data.append({
            "word": word_entry["word"],
            "transcriptions": processed_transcriptions,
            "word_level_transcriptions": word_entry["word_level_transcriptions"]
        })
    
    return processed_data

def save_data(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    input_file = "./adl_dataset/transcription.json"
    output_file = "./adl_dataset/processed_transcription.json"
    
    data = load_data(input_file)
    
    processed_data = process_transcriptions(data)
    
    save_data(processed_data, output_file)

if __name__ == "__main__":
    main()