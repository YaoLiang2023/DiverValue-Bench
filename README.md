# DiverValue-Bench: A Benchmark and Fine-Tuning Framework for Aligning Large Language Models with Diverse Human Values

## 📌 Overview

This repository contains the core source code for the paper:  
**“DiverValue-Bench: A Benchmark for Evaluating Large Language Models on Multi-Value Preference Alignment”**.

Our codebase supports the full pipeline for:
- Generating multi-value-aligned QA data (`DiverValue-Bench`) from real user survey responses and conversations.
- Training and evaluating alignment models (e.g., LLaMA2 with DPO + LoRA).
- Computing evaluation metrics such as:
  - **Preference Matching Accuracy**
  - **Semantic Preference Matching Rate**
  - **Cross-national and regional alignment breakdowns**

---

## 📁 Directory Structure

```
.
├── merge_data.py                           # Merge survey and conversation JSONL files into unified training data
├── chatbot_final.py                        # Convert user value scores into English preference summaries
├── add_values_final.py                     # Convert 7-dim user value scores into English preference descriptions using GPT-4o
├── generate_data_final.py                  # Generate DiverValue-Bench data from survey + chat history
├── train_llama2-7b_final.py                # Finetune LLaMA2 using DPO (LoRA + SFT)
├── eval_model_gpt-4o.py                    # Evaluate GPT-4o on DiverValue-Bench using reference answers
├── eval_self.py                            # Evaluate alignment accuracy using log-prob comparison
├── eval_uf-p-4.py                          # Evaluate policy model on UF-P-4 benchmark
├── llama2-7b_spmr.py                       # Evaluate Semantic Preference Matching Rate (SPMR)
├── statistics_birth_country_rada_final.py  # Generate radar charts: cross-national alignment
├── statistics_region_final.py              # Generate radar charts: regional-demographic alignment
├── ...                                     # Other source code files
```

---

## ⚙️ Installation Instructions

Please follow the step-by-step instructions below to install the required environment:

```bash
# Step 1: Install core dependencies
pip install torch==2.0.1
pip install -r requirements.txt
pip install -e ./transformers-4.31.0
pip install -e ./peft
pip install trl==0.6.0
pip install scikit-learn==1.3.0

# Step 2: Additional packages for evaluation
pip install sentence-transformers==2.2.2 --no-deps
pip install huggingface_hub==0.14.1
```

Note:
- Editable installs (`-e`) are required for both `transformers` and `peft` packages.
- `--no-deps` is used for `sentence-transformers` to prevent version conflicts.
- **Python ≥ 3.11**
- **GPU with bfloat16 or float16 support** (for LLaMA2)
- **OpenAI API key** (for GPT-4o evaluations)
---


## 🔁 User Value Preference Mapping

We use GPT-4o to transform user-provided value scores (from 0 to 100) across seven value dimensions into concise English preference descriptions. This process serves as the foundation for personalized Q&A pair generation and alignment evaluation.

### 🧩 Mapping Logic

Each user submits a value score dictionary like the following:

```json
{
  "creativity": 30,
  "fluency": 84,
  "factuality": 74,
  "diversity": 84,
  "safety": 39,
  "personalisation": 9,
  "helpfulness": 93
}
```

The model interprets the scores as:
- Scores ≥ 80 → High preference
- Scores ≤ 60 → Low preference
- Scores in [61–79] are ignored unless needed for context

Example output:
```
High helpfulness, High fluency, High diversity, low creativity, low safety, low personalisation.
```

### ⚙️ Execution

To generate value preference summaries for all users:

```bash
python merge_data.py
```

- Input: `data/survey.jsonl`, `data/conversations.jsonl`
- Output: `data/merged.json`

```bash
python add_values_final.py
```

- Input: `data/merged.jsonl`
- Output: `data/labeled_prism.json`
- OpenAI GPT-4o API is required (`api_key` must be set)

Core logic is implemented in `chatbot_final.py`.

---

## 🔄 Data Generation

To generate the DiverValue-Bench dataset from labeled JSONL files (survey responses + chats):

```bash
python generate_data_final.py
```

- Input: `data/labeled_prism.jsonl`
- Output: `data/generated_multi_value_dataset_with_info.json`

---

## 🧠 Model Training (LLaMA2 DPO)

To finetune LLaMA-2-7b with DPO (Direct Preference Optimization + LoRA):

```bash
python train_llama2-7b_final.py
```

- Model: `meta-llama/Llama-2-7b-hf`
- Output: `./output/llama2_dpo_finetuned_final`

---

## 📊 Evaluation Modules

### 1. Evaluate GPT-4o on DiverValue-Bench:

```bash
python eval_model_gpt-4o.py
```

### 2. Evaluate Policy vs Reference Model (Log-Prob):

```bash
python eval_self.py
```

### 3. Evaluate on UF-P-4:

```bash
python eval_uf-p-4.py
```

### 4. Semantic Preference Matching Rate (SPMR):

```bash
python llama2-7b_spmr.py
```

---

## 🌐 Regional & National Breakdown

### A. By **Birth Country**:

```bash
python statistics_birth_country_rada_final.py
```

### B. By **Region & Demographic Group**:

```bash
python statistics_region_final.py
```

---