# Install dependencies if needed
# !pip install transformers==4.31.0 datasets torch==2.0.1 peft trl==0.6.0 accelerate scikit-learn sentence-transformers

import torch
import random
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from peft import LoraConfig
from trl import DPOTrainer
import json
from sklearn.model_selection import train_test_split  # 现在不再使用，但保留以尽量少改动
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

seed = 12
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# =========================
# Use DiverValue-Bench data that has been correctly split by user_id
# =========================
train_path = 'data/DVB-train.json'
test_path = 'data/DVB-test.json'

with open(train_path, 'r', encoding='utf-8') as f:
    train_json = json.load(f)

with open(test_path, 'r', encoding='utf-8') as f:
    test_json = json.load(f)

def preprocess_entries(data_json):
    """Convert the original DiverValue Punch entry into the {prompt, choose, rejected} format required by DPO"""
    data_entries = []
    for entry in data_json:
        user_profile = (
            f"Age: {entry['age']}, Gender: {entry['gender']}, "
            f"Employment: {entry['employment_status']}, "
            f"Education: {entry['education']}, Marital Status: {entry['marital_status']}, "
            f"Birth Country: {entry['birth_country']}."
        )

        prompt = f"User profile: {user_profile}\nQuestion: {entry['question']}"

        data_entries.append({
            "prompt": prompt,
            "chosen": entry["answer_w"],
            "rejected": entry["answer_l"]
        })
    return data_entries

# Preprocess the training and testing sets separately
train_data = preprocess_entries(train_json)
test_data = preprocess_entries(test_json)

# Convert to Huggingface Dataset
datasets = DatasetDict({
    'train': Dataset.from_list(train_data),
    'test': Dataset.from_list(test_data)
})

# Load Llama-2-13b-hf model and tokenizer
model_id = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Setup LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./output/llama2-13b_dpo_finetuned_seed{}".format(seed),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="epoch",
    report_to="none",
    bf16=True,
    seed=seed,
)
set_seed(seed)

# Initialize DPO Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    beta=0.1,
    train_dataset=datasets['train'],
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_prompt_length=512,
    max_length=1024,
)

# Start Training
trainer.train()

# Evaluate using Semantic Preference Matching Rate
sentence_model = SentenceTransformer('./all-MiniLM-L6-v2')

def evaluate_semantic_matching_rate(model, dataset, tokenizer, sentence_model):
    model.eval()
    matched_count = 0
    print("=====================================Start evaluating========================================== ")
    for sample in dataset:
        inputs = tokenizer(sample['prompt'], return_tensors='pt').to(model.device)
        outputs = model.generate(**inputs, max_length=1024)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("========================inputs : {}, generated_text : {}==========================".format(sample['prompt'], generated_text))
        gen_vec = sentence_model.encode([generated_text])[0]
        chosen_vec = sentence_model.encode([sample['chosen']])[0]

        similarity = cosine_similarity([gen_vec], [chosen_vec])[0][0]

        if similarity > 0.8:
            matched_count += 1

    matching_rate = matched_count / len(dataset)
    print(f"Semantic Preference Matching Rate on test set: {matching_rate:.4f}")

# If evaluation is needed, cancel the next line of comments
# evaluate_semantic_matching_rate(model, datasets['test'], tokenizer, sentence_model)

# Save the fine-tuned model
trainer.save_model("./output/llama2-13b_dpo_finetuned_final_seed{}".format(seed))
