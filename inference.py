import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import csv
# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_entry(model, tokenizer, entry, max_length=512):
    """
    單一推論方法，處理單筆資料，並生成模型的回應。
    """
    instruction = entry["instruction"]
    input_text = entry["input"]
    
    # 構建輸入
    formatted_input = f"{instruction}\n{input_text}"
    encoded_input = tokenizer(
        formatted_input, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length
    )

    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)
    # 生成輸出
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=70, 
            pad_token_id=tokenizer.eos_token_id
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # 提取模型的回應部分
    response = output_text
    return response

# 測試函數
if __name__ == "__main__":
    # 加載模型和 Tokenizer
    base_model_path = "taide"
    peft_path = "lora"
    
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(model, peft_path)
    model.eval()

    with open("retrieval/keyword_query.json") as f:
        test_data = json.load(f)
    
    with open("raw_dataset/活網用語辭典.csv", newline="") as f:
        reader = csv.DictReader(f)
        keyword_list = [(row["詞彙"],row["釋義"]) for row in reader]
    
    output_list = []    
    for d in test_data:
        responses_list = []
        responses = {}   
        for entry in d["transcriptions"]:
            conversation = entry["A"]
            text = conversation["text"]
            best_matches_ids = [id["matched_index"] for id in conversation["top_matches"]]
            best_matches_words = [keyword_list[i][0] for i in best_matches_ids]
            best_matches_def = [keyword_list[i][1] for i in best_matches_ids]

            # 組合詞彙和釋義為詞語定義
            definitions = "\n".join(
                [f"詞語定義{i+1}: {word} - {definition}" for i, (word, definition) in enumerate(zip(best_matches_words, best_matches_def))]
            )
            input_text = f"{definitions}\nA: {text}\nB: "

            #print("Input:\n", input_text)
            test_entry = {
                "instruction": "根據以下對話生成適當的回應。",
                "input": input_text
            }
            print("Input:\nA: ", text)
            response = inference_entry(model, tokenizer, test_entry).split("B: ")[1]  
            print("Response:\nB: ", response)
            responses["input"]  = text
            responses["response"] = response
            responses_list.append(responses)
        output_list.append(responses_list)
      
    with open("inference_output.json", "w") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=4)   
