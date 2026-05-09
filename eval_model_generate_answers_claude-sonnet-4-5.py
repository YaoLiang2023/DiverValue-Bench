from openai import OpenAI
import json
import time
from tqdm import tqdm

# Set up your OpenAI API key
client = OpenAI(
    base_url="",
    api_key=""
)

MODEL_NAME = "claude-sonnet-4-5"

def load_dataset(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def build_profile_no_pref(sample):
    profile = [
        f"Age: {sample.get('age', '')}",
        f"Gender: {sample.get('gender', '')}",
        f"Education: {sample.get('education', '')}",
        f"Birth Country: {sample.get('birth_country/region', '')}",
        f"Marital Status: {sample.get('marital_status', '')}"
    ]
    return "\n".join(profile)

def build_prompt_with_profile(sample):
    user_profile = build_profile_no_pref(sample)
    prompt = (
        f"User Profile:\n{user_profile}\n"
        f"Question: {sample['question']}\n"
        f"Please answer this question as helpfully as possible, taking into account the user's background information."
    )
    return prompt

def query_openai(prompt, model=MODEL_NAME, max_tokens=512, temperature=0.7, retry=3):
    for i in range(retry):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful, unbiased, and polite AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error: {e}, retrying ({i+1}/{retry})...")
            time.sleep(3)
    return None

def main():
    input_file = "data/generated_multi_value_dataset_with_info.json"
    output_file = "data/claude-sonnet-4-5_multi_value_evaluation_result_pref_match.json"

    data = load_dataset(input_file)
    eval_results = []
    total = len(data)

    print(f"Total samples: {total}")
    for sample in tqdm(data, desc="Evaluating"):
        # 1. Generate model answers using only the profile and questions.
        prompt = build_prompt_with_profile(sample)
        model_answer = query_openai(prompt)
        user_profile = build_profile_no_pref(sample)
        # 2. In the discrimination stage, provide both `preference` and `stated_prefs` to the evaluation model.
        judge_result = ""

        eval_results.append({
            "id": sample["id"],
            "question": sample["question"],
            "profile": user_profile,
            "preference": sample.get("preference", ""),
            "stated_prefs": sample.get("stated_prefs", {}),
            "answer_w": sample["answer_w"],
            "model_answer": model_answer,
            "judge_match": judge_result
        })


    # Save all evaluation results.
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    print(f"All results saved in {output_file}")

if __name__ == "__main__":
    main()
