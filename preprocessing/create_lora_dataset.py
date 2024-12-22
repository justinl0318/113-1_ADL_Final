import json
import random
# 原始 dataset.json 資料
with open("raw_dataset/text.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)


# 生成 Alpaca 格式資料
def convert_to_alpaca_format(data):
    alpaca_data = []
    alpaca_data_valid = []
    for item in data:
        
        word = item["word"]
        definition = item["definition"]
        counter = 0
        for dialogue in item["dialogues"]:
            words = []
            if word in dialogue["A"]:
                words.append(item)
                words += random.sample([d for d in data if d["word"] != word], 2)
            else:
                words += random.sample([d for d in data], 3)

            instruction = "根據以下對話生成適當的回應。"
            int2zh = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五"}
            input_text = ""
            for i in range(len(words)):
                input_text += f"定義{int2zh[i+1]}: {words[i]['word']} 的意思是 {words[i]['definition']}\n"
            input_text += f"\nA: {dialogue['A']}\nB: "
        
            output_text = dialogue["B"]
            if counter % 5 != 4:
                alpaca_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                })
            else:
                alpaca_data_valid.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                })
            counter += 1
    return alpaca_data, alpaca_data_valid

# 轉換資料
alpaca_dataset_train, alpaca_dataset_valid = convert_to_alpaca_format(original_data)

# 儲存為 JSON 文件
output_file_train = "raw_dataset/alpaca_dataset_train.json"
output_file_valid = "raw_dataset/alpaca_dataset_valid.json"

with open(output_file_train, "w", encoding="utf-8") as f:
    json.dump(alpaca_dataset_train, f, ensure_ascii=False, indent=4)

with open(output_file_valid, "w", encoding="utf-8") as f:
    json.dump(alpaca_dataset_valid, f, ensure_ascii=False, indent=4)

print(f"轉換完成，訓練資料集儲存於 {output_file_train}，驗證資料集儲存於 {output_file_valid}。")
