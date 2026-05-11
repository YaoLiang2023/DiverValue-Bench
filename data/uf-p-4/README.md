# UF-P-4 Evaluation Data

This directory is reserved for the UF-P-4 evaluation data used in the out-of-domain preference-alignment evaluation reported in our paper:

**DiverValue-Bench: A Benchmark and Fine-Tuning Framework for Aligning Large Language Models with Diverse Human Values**

## Important Notice

UF-P-4 is a third-party evaluation benchmark. It is **not introduced by DiverValue-Bench** and is used in this repository only for reproducing the out-of-domain OPA evaluation described in our paper.

For licensing and attribution reasons, we do **not** redistribute the UF-P-4 JSON files in this repository. Users should obtain the UF-P-4 files from the original public source and comply with the original dataset license, citation requirements, and terms of use.

## Original Reference

The UF-P-4 benchmark is introduced in the following paper:

```bibtex
@misc{poddar2024personalizing,
  title        = {Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning},
  author       = {Sriyash Poddar and Yanming Wan and Hamish Ivison and Abhishek Gupta and Natasha Jaques},
  year         = {2024},
  eprint       = {2408.10075},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2408.10075}
}
````

UF-P-4 is based on UltraFeedback and constructs personalized preference data using four fine-grained attributes:

* helpfulness
* honesty
* instruction following
* truthfulness

In UF-P-4, these four attributes are treated as different user preference types, making it suitable for evaluating whether models can handle divergent user preferences.

## Source Used in Our Experiments

The UF-P-4 files used in our experiments were downloaded from the AlignX repository:

```text
https://github.com/JinaLeejnl/AlignX/tree/main/benchmark/UF-P-4
```

The expected files are:

```text
data/uf-p-4/
├── helpfulness.json
├── honesty.json
├── instruction_following.json
└── truthfulness.json
```

## Why the JSON Files Are Not Included Here

Although the UF-P-4 files are publicly accessible from the source above, they are third-party benchmark files. To avoid ambiguity about redistribution rights, this repository does not directly host the UF-P-4 JSON files.

Instead, we provide:

* the evaluation scripts under `scripts/eval_ufp4/`;
* the expected directory structure;
* the expected data format;
* the original data source;
* the citation information for the original UF-P-4 paper.

This allows users to reproduce our out-of-domain evaluation while obtaining the third-party benchmark data from its original source.

## Expected Data Format

Each JSON file should contain preference-pair examples. The evaluation scripts expect each entry to include at least the following fields:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

Some versions may also include a `profile` field:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "profile": "..."
}
```

If `profile` is available, our evaluation scripts prepend it to the prompt before computing the model likelihood scores.

## How to Prepare the Data

Please download the following files from the original AlignX source:

```text
helpfulness.json
honesty.json
instruction_following.json
truthfulness.json
```

Then place them in this directory:

```text
data/uf-p-4/
```

After placement, the directory should look like:

```text
data/uf-p-4/
├── README.md
├── helpfulness.json
├── honesty.json
├── instruction_following.json
└── truthfulness.json
```

## Usage in This Repository

The scripts under the following directory use UF-P-4 for out-of-domain OPA evaluation:

```text
scripts/eval_ufp4/
```

For example:

```bash
python scripts/eval_ufp4/eval_uf-p-4_llama2-7b_opa_non_train_profile.py
python scripts/eval_ufp4/eval_uf-p-4_llama2-7b_opa_profile_seed12.py
```

The same evaluation protocol is provided for LLaMA-2-13B, Qwen-7B, and Qwen-14B.

## Evaluation Protocol

For each UF-P-4 example, the evaluation scripts compute the normalized log probability of the preferred response and the dispreferred response:

* `chosen`: the preferred response;
* `rejected`: the dispreferred response.

A model is counted as correct if it assigns a higher normalized log probability to `chosen` than to `rejected`.

The resulting score is reported as OPA, following the evaluation setting used in our paper.

## Citation Requirement

If you use UF-P-4, please cite the original UF-P-4 paper:

```bibtex
@misc{poddar2024personalizing,
  title        = {Personalizing Reinforcement Learning from Human Feedback with Variational Preference Learning},
  author       = {Sriyash Poddar and Yanming Wan and Hamish Ivison and Abhishek Gupta and Natasha Jaques},
  year         = {2024},
  eprint       = {2408.10075},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2408.10075}
}
```

If you obtain the files from AlignX, please also cite or acknowledge the AlignX repository according to its instructions.

## Disclaimer

UF-P-4 is provided by third-party sources. The authors of DiverValue-Bench do not claim ownership of UF-P-4. Users are responsible for ensuring that their use of UF-P-4 complies with the original dataset license, terms of use, and citation requirements.