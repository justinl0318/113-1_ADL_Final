# 113-1 ADL_final Group 28

### Environment setup

Run `pip install -r requirements.txt` to install the necessary packages. 

### Dataset

```bash
├── README.md
├── raw_dataset
├── ├── audio_files/ # folder containing training data in .wav format
│   ├── 活網用語辭典.csv # gen-alpha slangs, contain (keyword, definition, sentence)
│   ├── transcription.json # training data that has been transcribed to text by an ASR model
```

### Preprocessing

Run `./preprocessing/preprocess.sh`. After that, a file named `aligned_phoneme.json` will be created under the `preprocessed_dataset/` folder, which contains the corresponding audio segments for each text segments in the training data.

### Retrieval

We present two strategies to perform retrieval. 

1. Segmentation (斷詞) -> Identify the segment that best matches the keyword in the databse -> Extract keyword + definition + example 
2. 

### Training



### Inference



