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

#### 1. Prepare Axolotl Environment

```bash
conda create -n adlhw3 python=3.10
conda activate adlhw3
pip install torch==2.4.1 transformers==4.45.1 bitsandbytes==0.44.1 peft==0.13.0

conda install -c conda-forge cudatoolkit-dev
git clone https://github.com/axolotl-ai-cloud/axolotl
cd axolotl

pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'
```

#### 2. Start Training

##### `train.yml`


```
base_model: models/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1
datasets:
  - path: ./raw_dataset/alpaca_dataset.json
output_dir: ./lora
```

##### command

```bash
accelerate launch -m axolotl.cli.train train_bonus.yml 
```

### Inference



