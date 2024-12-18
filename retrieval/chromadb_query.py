from transformers import ClapModel, ClapProcessor, AutoTokenizer
from datasets import load_dataset, Dataset, Audio
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
        self.df = pd.read_csv('adl_dataset/活網用語辭典.csv')
    
    def store_text(self):
        path_to_data = 'adl_dataset/concat_data.json'
        
        with open(path_to_data, 'r') as f:
            data = json.load(f)
        
        items = []    
        ids = []
        for i in range(len(data)):
            items.append(data[i]['concat_text'])
            ids.append(str(i))

        # items = []
        # ids = []
        # for i in range(len(self.df)):
        #     items.append(self.df.loc[i]['詞彙'])
        #     ids.append(str(i))
        
        self.collection.add(
            documents=items,
            ids=ids
        )
        
    def store_audio(self):
        path_to_data = 'adl_dataset/concat_data.json'
        
        with open(path_to_data, 'r') as f:
            data = json.load(f)
        
        items = []    
        ids = []
        for i in range(len(data)):
            items.append('adl_dataset/' + data[i]['concat_audio_path'])
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
    with open('adl_dataset/keyword_query.json', 'r') as f:
        data = json.load(f)
    correct = 0
    for i in range(len(data)):
        for j in range(len(data[i]['transcriptions'])):
            results = db.retrieve_from_text(data[i]['transcriptions'][j]['A']['text'], k)
            data[i]['transcriptions'][j]['A']['retrieve'] = results
            if str(i) in results['ids'][0]:
                correct += 1
           
            results = db.retrieve_from_text(data[i]['transcriptions'][j]['B']['text'], k)
            data[i]['transcriptions'][j]['B']['retrieve'] = results
            if str(i) in results['ids'][0]:
                correct += 1
    
            
def main():
    db = Chromadb()
    db.store_text()
    query_text(db, 30)
    # query_audio(db)
    
    
    db = Chromadb()
    db.store_text()
    query_text(db, 30)
    # query_audio(db)
    
    
if __name__ == "__main__":
    main()