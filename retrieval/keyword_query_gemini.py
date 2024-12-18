import json
import time
from tqdm import tqdm
import google.generativeai as genai

# Configure the generative AI API
genai.configure(api_key="AIzaSyB3j4gnmRmZ8aIMJ7wL8U2ziriT3YUA9KQ")

# Create the model
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Load the transcription data
with open("../raw_dataset/transcription.json", "r", encoding="utf-8") as file:
    transcription_data = json.load(file)

results = []


# Function to call the model with retries
def call_model_with_retries(prompt, max_retries=60, delay=2):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            tqdm.write(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise e  # Re-raise the exception if max retries reached


# Calculate total number of sentences for progress bar
total_sentences = sum(len(entry.get("sentences", [])) for entry in transcription_data)

# Initialize progress bar
start = False
with tqdm(total=total_sentences, desc="Processing", unit="sentence") as pbar:
    # Process each text entry
    for entry in transcription_data:
        for transcription in entry.get("sentences", []):
            for speaker, texts in transcription.items():
                text = texts

                if text == "你覺得我該怎麼辦?他一直用句臉讓我很困惑":
                    start = True
                if not start:
                    continue
                prompt = (
                    f'請從以下句子找出1個可能較難以理解用法的流行用語，標注出來到"phrases"。\n'
                    f'Example:["破大房"]\n\n"""{text}"""'
                )

                try:
                    # Call the model with retries
                    generated_response = call_model_with_retries(prompt)
                    results.append(
                        {"input_text": text, "generated_response": generated_response}
                    )

                    # Print the generated response
                    tqdm.write(f"Input text: {text}")
                    tqdm.write(f"Generated response: {generated_response}")
                except Exception as e:
                    tqdm.write(f"Error processing text after retries: {text} -> {e}")
                    results.append({"input_text": text, "error": str(e)})
                    tqdm.write("Max retries reached. Exiting.")
                    with open(
                        "results_new_flash_2.0_exp.json", "w", encoding="utf-8"
                    ) as file:
                        json.dump(results, file, ensure_ascii=False, indent=4)
                    exit(1)
                finally:
                    pbar.update(1)  # Update the progress bar

# Save results to a JSON file
output_file = "results_only_keyword.json"
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=4)

tqdm.write(f"Results saved to {output_file}")
