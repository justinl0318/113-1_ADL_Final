import json

# 原始 dataset.json 資料
with open("data/text.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)
# 生成 Alpaca 格式資料
def convert_to_alpaca_format(data):
    alpaca_data = []
    for item in data:
        word = item["word"]
        definition = item["definition"]
        for dialogue in item["dialogues"]:
            instruction = "根據以下對話生成適當的回應。"
            input_text = f"定義: {word} 的意思是 \"{definition}\"。\nA: {dialogue['A']}\nB: "
            output_text = dialogue["B"]
            alpaca_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })
    return alpaca_data

# 轉換資料
alpaca_dataset = convert_to_alpaca_format(original_data)

# 儲存為 JSON 文件
output_file = "alpaca_dataset.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(alpaca_dataset, f, ensure_ascii=False, indent=4)

print(f"Alpaca 格式的數據已儲存至 {output_file}")
