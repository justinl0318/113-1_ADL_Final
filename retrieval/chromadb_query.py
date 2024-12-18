from transformers import ClapModel, ClapProcessor, AutoTokenizer
from datasets import Dataset, Audio
from typing import Optional, Union, cast
import json
import pandas as pd
from pydub import AudioSegment
import chromadb
from chromadb.api.types import (
    Document,
    Documents,
    Embedding,
    EmbeddingFunction,
    Embeddings,
)

class CLAPEmbeddingFunction(EmbeddingFunction[Union[Documents, Documents]]):
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: Optional[str] = 0,
    ) -> None:

        model = ClapModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = ClapProcessor.from_pretrained(model_name)
        
        self._model = model
        self._model.to(device)
        self._processor = processor
        self._tokenizer = tokenizer
        self._device = device

    def _encode_audio(self, audio: Document) -> Embedding:
        audio_file = Dataset.from_dict({"audio": [audio]}).cast_column("audio", Audio(sampling_rate=48000))
        inputs =  self._processor(audios=audio_file[0]['audio']['array'], sampling_rate=audio_file[0]['audio']['sampling_rate'], return_tensors="pt").to(0)
        audio_embed = self._model.get_audio_features(**inputs)
        
        return cast(Embedding, audio_embed.squeeze().cpu().detach().numpy())

    def _encode_text(self, text: Document) -> Embedding:
        inputs = self._tokenizer(text, padding='max_length', return_tensors="pt", truncation=True, max_length=512).to(self._device)
        text_embed = self._model.get_text_features(**inputs)
        
        return cast(Embedding, text_embed.squeeze().cpu().detach().numpy())

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        for item in input:
            if item.endswith('.wav'):
                embeddings.append(self._encode_audio(cast(Document, item)))
            else:
                embeddings.append(self._encode_text(cast(Document, item)))
        return embeddings
    
class Chromadb:
    def __init__(self):
        chroma_client = chromadb.Client()
        embedding_function = CLAPEmbeddingFunction()
        self.collection = chroma_client.get_or_create_collection(name="audio_text_collection", embedding_function=embedding_function)
        self.df = pd.read_csv('../raw_dataset/活網用語辭典.csv')
    
    def store_text(self):
        path_to_data = '../raw_dataset/audio.json'
        
        with open(path_to_data, 'r') as f:
            data = json.load(f)
        
        items = []    
        ids = []
        for i in range(len(data)):
            items.append(data[i]['word'])
            ids.append(str(i))

        self.collection.add(
            documents=items,
            ids=ids
        )
        
    def store_audio(self):
        path_to_data = '../raw_dataset/audio.json'
        
        with open(path_to_data, 'r') as f:
            data = json.load(f)
        
        items = []    
        ids = []
        for i in range(len(data)):
            items.append('../raw_dataset/' + data[i]['word_audio_path'])
            ids.append(str(i))

        self.collection.add(
            documents=items,
            ids=ids
        )
        
    def retrieve_from_text(self, text, k):
        results = self.collection.query(
            query_texts=[text],
            n_results=k
        )
        return results

    def retrieve_from_audio(self, audio, k):
        results = self.collection.query(
            query_texts=[audio],
            n_results=k
        )
        return results

def query_text(db, k):
    with open('keyword_query_gemini.json', 'r') as f:
        data = json.load(f)
    
    for i in range(len(data)):
        if i % 20 == 0:
            print(i)
        for j in range(len(data[i]['transcriptions'])):
            for c in ['A', 'B']:
                results = db.retrieve_from_text(data[i]['transcriptions'][j][c]['best_match']['word'], k)
                data[i]['transcriptions'][j][c]['retrieve'] = results
                if str(i) in results['ids'][0]:
                    data[i]['transcriptions'][j][c]['correct'] = '1'
                else:
                    data[i]['transcriptions'][j][c]['correct'] = '0'
    
    return data                
    
        
def segment_audio(input_file, start_time, end_time, output_file):
    audio = AudioSegment.from_wav(input_file)
    
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    
    segmented_audio = audio[start_ms:end_ms]
    
    segmented_audio.export(output_file, format="wav")
                    
    
def query_audio(db, k):
    with open('keyword_query_gemini.json', 'r') as f:
        data = json.load(f)
    with open('../raw_dataset/audio.json', 'r') as f:
        path = json.load(f)
        
    for i in range(len(data)):
        for j in range(len(data[i]['transcriptions'])):
            for c in ['A', 'B']:
                if i % 20 == 0:
                    print(i)
                input_audio = '../raw_dataset/' + path[i]['paths'][j][c]
                segment_audio(input_audio, data[i]['transcriptions'][j][c]['best_match']['start_time'], data[i]['transcriptions'][j][c]['best_match']['end_time'], 'tmp.wav')
                results = db.retrieve_from_audio('tmp.wav', k)
                data[i]['transcriptions'][j][c]['retrieve'] = results
                if str(i) in results['ids'][0]:
                    data[i]['transcriptions'][j][c]['correct'] = '1'
                else:
                    data[i]['transcriptions'][j][c]['correct'] = '0'
                    
def main():
    db = Chromadb()
    db.store_text()
    data = query_text(db, 10)
    with open('retrieve_text2text.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    
    db = Chromadb()
    db.store_audio()
    data = query_text(db, 10)
    with open('retrieve_text2audio.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    
if __name__ == "__main__":
    main()