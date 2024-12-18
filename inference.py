import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_single_entry(model, tokenizer, entry, max_length=256):
    """
    單一推論方法，處理單筆資料，並生成模型的回應。
    """
    instruction = entry["instruction"]
    input_text = entry["input"]
    
    # 構建輸入
    formatted_input = f"{instruction}\\n{input_text}\\n"
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
            max_new_tokens=50, 
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

    test_entry = {
        "instruction": "根據以下對話生成適當的回應。",
        "input": "你再這樣下去，大家會把你當成公司的遲刻魔\n",
    }
    
    # 推論
    model.eval()
    generated_response = inference_single_entry(model, tokenizer, test_entry)
    print("Without LoRA:\n", generated_response)
    model = PeftModel.from_pretrained(model, peft_path)
    model = model.to(device)
    generated_response = inference_single_entry(model, tokenizer, test_entry)
    print("With LoRA:\n", generated_response)
    
