from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import json

################### Select device ####################
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


################### ASR: Whisper model ####################
# model_id = "models/models--andybi7676--cool-whisper/snapshots/9243ae0ca0cff575d2ca8a5c5698e232bf47821c"
# asr_model = WhisperModel(model_id, device=device, compute_type="float16")


################### Chat LLM: TAIDE ####################
chat_tokenizer = AutoTokenizer.from_pretrained(
    # "/home/nlp/b10902031/emer_llm/models/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
    "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
    # local_files_only=True
)
if chat_tokenizer.pad_token is None:
    chat_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch_dtype
)
chat_model = AutoModelForCausalLM.from_pretrained(
    # "/home/nlp/b10902031/emer_llm/models/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
    "taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
    quantization_config=quantization_config, 
    device_map="auto"
)

def get_chat_output(input, tokenizer, model, device):
    chat = [
        {
            "role": "system", 
            "content": """你是一位熟悉台灣鄉民梗的網民，要根據輸入的對話，輸出詳細解釋。"""
        },
        {
            "role": "user",
            "content": input
        },
    ]
    tokenized_input = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(device)

    # Generate model output
    outputs = model.generate(
        input_ids=tokenized_input,
        max_new_tokens=256,
        # do_sample=True,
        # temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated_text = generated_text.replace("<|eot_id|>", "").replace("\n", "").split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

    return generated_text

with open ("raw_dataset/text.json") as f:
    data = json.load(f)

for word_entry in data:
    dialogues = word_entry["dialogues"]
    for dialogue in dialogues:
        A = dialogue["A"]
        B = dialogue["B"]
        print(f"A: {A}\nB: {B}")
        raw_output = get_chat_output(f"解釋這段對話：「{A + B}」", chat_tokenizer, chat_model, device)
        print(f"raw output: {raw_output}")

# with open("adl_dataset/keyword_query.json") as f:
#     data = json.load(f)[word_idx]

# with open("adl_dataset/text.json") as f:
#     text = json.load(f)[word_idx]
# definition = text["definition"]


# word = data["word"]
# A = data["transcriptions"][exp_idx]["A"]["text"]
# B = data["transcriptions"][exp_idx]["B"]["text"]
# replace_segment = data["transcriptions"][exp_idx]["B"]["segmented"].split(" ")
# replace_idx = data["transcriptions"][exp_idx]["B"]["best_match"]["start_index"]
# replace_len = data["transcriptions"][exp_idx]["B"]["best_match"]["segment_length"]
# for i in range(replace_idx, replace_idx + replace_len):
#     if i == replace_idx:
#         replace_segment[i] = word
#     else:
#         replace_segment[i] = ""
# fixB = "".join(replace_segment)

# print(f"A: {A}\nB: {B}\nfixB: {fixB}")

# raw_output = get_chat_output(f"解釋這段對話：「{A + B}」", chat_tokenizer, chat_model, device)
# fix_output = get_chat_output(f"解釋這段對話：「{A + fixB}」\n詞彙釋義：{word}。{definition}", chat_tokenizer, chat_model, device)

# print(f"raw output: {raw_output}\nfix output: {fix_output}")

# A: 有,你昨天有看到阿明的IG線動嗎?
# B: 有,他居然逃去沖繩玩,完全沒跟我們說,傻抱屁眼。
# fixB: 有,他居然逃去沖繩玩,完全沒跟我們說,傻爆屁眼。
# raw output: 在這段對話中，話題是關於一位名為阿明的朋友，在Instagram上分享他到沖繩旅遊的點滴。以下是對話中可能提及的幾個網路流行文化元素： 一、IG：Instagram的簡稱，是一個社群媒體平台，使用者可上傳並分享照片及影片，搭配文字說明。 二、線動：可能是「線動」的拼寫錯誤，應該是指Instagram的動態貼文，使用者可分享生活點滴、照片、影片等，其他使用者都可以看到且互動。 三、傻子：這個詞語在這個脈絡中，很可能是用來形容阿明，在他到沖繩旅遊前，並未先告知朋友，而朋友在看到他在Instagram上分享旅遊點滴後，才感到驚訝。詞語「傻子」在中文中常被用來形容一個人行為愚蠢、不明智。 四、屁眼：這個詞語在中文中常被用來作為對某人
# fix output: 這段對話中，「傻爆屁眼」這個成語被用來表達強烈的無奈或驚訝。網友在談話中，表示對阿明未告知就前往沖繩玩一事感到十分無奈，這種無奈之情，被誇張化，以「傻爆屁眼」來表達，強調其強烈程度。這句成語常被用在網路討論區、論壇等網路語言盛行的場合，以求表現說話者情緒的強度和語氣的強烈。