from transformers import ClapModel, ClapProcessor, AutoTokenizer
from datasets import load_dataset, Dataset, Audio
from typing import Optional, Union, cast
import os
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
        model_name: str = "laion/larger_clap_general",
        device: Optional[str] = 0,
    ) -> None:

        model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
        tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_general")
        processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        
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
        inputs = self._tokenizer(text, padding=True, return_tensors="pt").to(self._device)
        text_embed = self._model.get_text_features(**inputs)
        
        return cast(Embedding, text_embed.squeeze().cpu().detach().numpy())

    def __call__(self, input: Union[Documents, Documents]) -> Embeddings:
        embeddings: Embeddings = []
        for item in input:
            if item.endswith('.wav'):
                embeddings.append(self._encode_audio(cast(Document, item)))
            else:
                embeddings.append(self._encode_text(cast(Document, item)))
        return embeddings


chroma_client = chromadb.Client()
embedding_function = CLAPEmbeddingFunction()
path_to_audio = 'adl_dataset/audio_files'

audio_filenames = os.listdir(path_to_audio)
ids = [file[:-4] for file in audio_filenames]
audio_filenames = [os.path.join(path_to_audio, file) for file in audio_filenames]

collection = chroma_client.get_or_create_collection(name="audio_text_collection", embedding_function=embedding_function)
collection.add(
    documents=audio_filenames,
    ids=ids
)

results = collection.query(
    query_texts=["破大防"],
    n_results=5
)

print(results)


