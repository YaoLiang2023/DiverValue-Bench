import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ========== 1. Load all subsets of UF-P-4 ==========
data_files = [
    'data/uf-p-4/helpfulness.json',
    'data/uf-p-4/honesty.json',
    'data/uf-p-4/instruction_following.json',
    'data/uf-p-4/truthfulness.json'
]

all_entries = []
for file in data_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            if all(k in entry for k in ("prompt", "chosen", "rejected")):
                profile = entry.get("profile", "")
                # Splice the profile and prompt, separated by '\ n'. If the profile does not exist, only the prompt is used
                if profile:
                    full_prompt = profile.strip() + "\n" + entry["prompt"].strip()
                else:
                    full_prompt = entry["prompt"].strip()
                all_entries.append({
                    "prompt": full_prompt,
                    "chosen": entry["chosen"],
                    "rejected": entry["rejected"]
                })

print(f"Loaded {len(all_entries)} samples from UF-P-4.")

# ========== 2. Load LoRA model and base model ==========
model_id = "meta-llama/Llama-2-7b-hf"
adapter_dir = "./output/llama2-7b_dpo_finetuned_final_seed12"  # Modify the adapter path according to your saved settings

# Strategy model (with LoRA adapter)
policy_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
policy_model = PeftModel.from_pretrained(policy_model, adapter_dir)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# ========== 3. Evaluation function ==========
def get_log_prob(model, tokenizer, prompt, answer):
    full_input = prompt + answer
    inputs = tokenizer(full_input, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        answer_ids = input_ids[0, prompt_len:]
        answer_log_probs = log_probs[0, prompt_len-1:-1, :]
        log_prob = 0.0
        for i, token_id in enumerate(answer_ids):
            log_prob += answer_log_probs[i, token_id].item()
        return log_prob

def get_log_prob_normalized(model, tokenizer, prompt, answer):
    full_input = prompt + answer
    inputs = tokenizer(full_input, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        answer_ids = input_ids[0, prompt_len:]
        answer_log_probs = log_probs[0, prompt_len-1:-1, :]
        log_prob = 0.0
        for i, token_id in enumerate(answer_ids):
            log_prob += answer_log_probs[i, token_id].item()
    answer_length = len(answer_ids)
    if answer_length == 0:
        # In theory, it won't happen. If the answer is empty, it will return a very small number
        print("prompt : {}; answer : {}".format(prompt, answer))
        return -1e9
    return log_prob / answer_length  # 平均log-prob

# 4. Evaluation accuracy (PREF indicator)
def evaluate_alignment_accuracy(policy_model, dataset, tokenizer):
    policy_model.eval()
    align_correct = 0
    print("============== Start evaluating Alignment Accuracy on UF-P-4 ==============")
    for idx, sample in enumerate(dataset):
        prompt, y_w, y_l = sample['prompt'], sample['chosen'], sample['rejected']
        lp_chosen = get_log_prob(policy_model, tokenizer, prompt, y_w)
        lp_rejected = get_log_prob(policy_model, tokenizer, prompt, y_l)
        if lp_chosen > lp_rejected:
            align_correct += 1
        if idx < 5 or idx % 500 == 0:
            print(f"[{idx+1}/{len(dataset)}] logP_chosen: {lp_chosen:.4f}, logP_rejected: {lp_rejected:.4f} => {'OK' if lp_chosen > lp_rejected else 'FAIL'}")
    alignment_acc = align_correct / len(dataset)
    print(f"\nAlignment Accuracy on UF-P-4: {alignment_acc:.4f}")

def evaluate_robust_alignment_accuracy(model, dataset, tokenizer):
    """
    calculation:
    -Normal PROF Accuracy (delta=0)
    -Strict Accuracy under Various Deltas
    -And the Robust Alignment Accuracy (RAA) after multi threshold averaging
    """
    model.eval()
    margins = []
    deltas = (0.0, 0.2, 2.0, 4.0, 6.0, 8.0)
    print("================================== Start evaluating Robust Alignment Accuracy ==================================")
    print(f"Using delta set: {deltas}")

    for idx, sample in enumerate(dataset):
        prompt, y_w, y_l = sample['prompt'], sample['chosen'], sample['rejected']

        lp_chosen = get_log_prob_normalized(model, tokenizer, prompt, y_w)
        lp_rejected = get_log_prob_normalized(model, tokenizer, prompt, y_l)
        margin = lp_chosen - lp_rejected
        margins.append(margin)

        if idx < 5 or idx % 500 == 0:
            print(f"[{idx+1}/{len(dataset)}] "
                  f"logP_chosen: {lp_chosen:.4f}, logP_rejected: {lp_rejected:.4f}, "
                  f"margin: {margin:.4f}")

    N = len(margins)
    # Normal PROF Accuracy (delta=0)
    acc_0 = sum(1 for m in margins if m > 0.0) / N

    # Strict Accuracy under Various Deltas
    acc_per_delta = {}
    for delta in deltas:
        acc_d = sum(1 for m in margins if m > delta) / N
        acc_per_delta[delta] = acc_d

    # Multi threshold averaging: Robust Alignment Accuracy
    # Note: Delta sets usually contain 0, so RAA will be<=acc_0
    raa = sum(acc_per_delta[d] for d in deltas) / len(deltas)

    print("\n---------------- Evaluation Result ----------------")
    print(f"PREF Accuracy (delta=0): {acc_0:.4f}")
    for d in deltas:
        print(f"Strict Alignment Accuracy (delta={d:.1f}): {acc_per_delta[d]:.4f}")
    print(f"Robust Alignment Accuracy (RAA): {raa:.4f}")
    print("--------------------------------------------------")


# 5. Execute evaluation
# evaluate_pref_accuracy(policy_model, test_dataset, tokenizer)
evaluate_robust_alignment_accuracy(policy_model, all_entries, tokenizer)
