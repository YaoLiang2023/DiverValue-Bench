import json
import os
from collections import defaultdict

import pandas as pd
import plotly.express as px


# =========================
# Config
# =========================
# Evaluation results of 5 models (poly audit) JSON: Sample by sample record+final_1abel (W/L/TIE)
# Explanation: By default, they are all located in the data/audit/directory.
MODEL_JSONS = {
    "gpt-4o": "data/judge/gpt-4o_multi_value_judge_pairwise_polyjudge_recomputed.json",
    "doubao-1.5-pro-32k": "data/judge/doubao-1.5-pro-32k_multi_value_judge_pairwise_polyjudge.json",
    "deepseek-v3-fast": "data/judge/deepseek-v3-fast_multi_value_judge_pairwise_polyjudge.json",
    "claude-sonnet-4-5": "data/judge/claude-sonnet-4-5_multi_value_judge_pairwise_polyjudge.json",
    "mistral-medium": "data/judge/mistral-medium_multi_value_judge_pairwise_polyjudge.json",
}

# output file
OUT_CSV_DETAIL = "country_paa_5models_detail.csv"
OUT_CSV_MEAN = "country_paa_5models_mean.csv"
OUT_FIG_PDF = "data/judge/chart/country_paa_map_5models_mean.pdf"

# Standardization of Country/Region Names (for ease of Plotly Map Matching)
COUNTRY_NORMALIZE = {
    "Russian Federation": "Russia",
    "Korea, Republic of": "South Korea",
    "Viet Nam": "Vietnam",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Hong Kong SAR, China": "Hong Kong",
    "United States of America": "United States",
}


def extract_birth_country(profile):
    if profile is None:
        return None

    if isinstance(profile, dict):
        # Common key name compatibility
        for k in [
            "birth_country/region",
            "Birth Country/Region",
            "birth_country",
            "birth_country_region",
            "birthCountry",
        ]:
            if k in profile and profile[k] is not None:
                return str(profile[k]).strip()
        return None

    if isinstance(profile, str):
        for line in profile.splitlines():
            s = line.strip()
            if not s:
                continue
            # 兼容 "Birth Country:" / "Birth Country/Region:"
            if s.lower().startswith("birth country"):
                parts = s.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return None

    return None


def normalize_country(name):
    """标准化国家名；返回 None 表示跳过（例如未声明国家）。"""
    if not name:
        return None
    name = str(name).strip()
    name = COUNTRY_NORMALIZE.get(name, name)

    low = name.lower()
    if low in {"prefer not to say", "unstated", "unknown", "n/a", "none", "null"}:
        return None
    return name


def load_items(json_path):
    """兼容 JSON 顶层为 list 或 dict 的情况。"""
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    # 有些导出格式可能是 {"data":[...]} 或 {id: sample, ...}
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        # fallback：把 value 当作样本
        return list(obj.values())

    raise ValueError("Unsupported JSON top-level type: {}".format(type(obj)))


def country_paa_from_items(items):
    """返回：country -> {N, aligned, paa}，其中 paa 为百分比（0-100）。"""
    totals = defaultdict(int)
    aligned = defaultdict(int)

    for inst in items:
        country_raw = extract_birth_country(inst.get("profile"))
        country = normalize_country(country_raw)
        if country is None:
            continue

        totals[country] += 1

        # conservative: 仅 final_label == "W" 计为 aligned；TIE / L 都视为非对齐
        label = inst.get("final_label") or inst.get("label")
        if label == "W":
            aligned[country] += 1

    out = {}
    for country, n in totals.items():
        paa = 100.0 * aligned[country] / n if n else 0.0
        out[country] = {"N": n, "aligned": aligned[country], "paa": paa}
    return out


def resolve_path(p):
    """允许用户直接给文件名（无目录）；若在当前目录找不到，则原样返回。"""
    if os.path.exists(p):
        return p
    basename = os.path.basename(p)
    if os.path.exists(basename):
        return basename
    return p


# =========================
# Load + Aggregate (per model)
# =========================
model_country = {}  # model -> country -> {N, aligned, paa}

missing = [m for m, p in MODEL_JSONS.items() if not os.path.exists(resolve_path(p))]
if missing:
    raise FileNotFoundError(
        "Missing JSON files for models: {}\n"
        "Please check MODEL_JSONS paths (current working dir: {}).".format(
            ", ".join(missing), os.getcwd()
        )
    )

for model, path in MODEL_JSONS.items():
    path = resolve_path(path)
    items = load_items(path)
    model_country[model] = country_paa_from_items(items)


# =========================
# Merge to DataFrame
# =========================
# 取各模型国家集合并集
all_countries = sorted({c for d in model_country.values() for c in d.keys()})

rows = []
for country in all_countries:
    row = {"Country": country}

    # 记录每个模型的 PAA 与 N
    paas = []
    Ns = []
    aligned_sum = 0
    total_sum = 0
    present_models = 0

    for model in MODEL_JSONS.keys():
        rec = model_country[model].get(country)
        if rec is None:
            row[f"PAA_{model}"] = None
            row[f"N_{model}"] = None
            continue

        paa = rec["paa"]
        n = rec["N"]
        row[f"PAA_{model}"] = round(paa, 2)
        row[f"N_{model}"] = n

        paas.append(paa)
        Ns.append(n)
        aligned_sum += rec["aligned"]
        total_sum += n
        present_models += 1

    # (1) macro mean: 对每个模型的 country-level PAA 做简单平均（与你“5个模型PAA平均值”的表述最一致）
    row["PAA_mean"] = round(sum(paas) / len(paas), 2) if paas else None

    # (2) micro mean: 将 5 个模型在该国家的 aligned/total 汇总后再算（等价于按 N 加权）
    row["PAA_micro"] = round(100.0 * aligned_sum / total_sum, 2) if total_sum else None

    # N：为了图与表直观，给出该国家“每个模型平均 N”
    row["N_mean"] = round(sum(Ns) / len(Ns), 2) if Ns else None
    row["Models_present"] = present_models

    rows.append(row)


df = pd.DataFrame(rows)

# 细表（含每模型列）
df.sort_values("PAA_mean", ascending=False).to_csv(
    "country_paa_5models_detail.csv", index=False, encoding="utf-8-sig"
)

# 简表（只保留平均值 + 关键统计）
cols_mean = ["Country", "PAA_mean", "PAA_micro", "N_mean", "Models_present"]
df[cols_mean].sort_values("PAA_mean", ascending=False).to_csv(
    "country_paa_5models_mean.csv", index=False, encoding="utf-8-sig"
)

# 同时在控制台输出每个国家/地区的平均值（按 PAA_mean 从高到低）
print("\nPer-country mean PAA across 5 models (sorted by PAA_mean):")
for _, r in df[cols_mean].sort_values("PAA_mean", ascending=False).iterrows():
    print(
        f"{r['Country']}: PAA_mean={r['PAA_mean']:.2f}%, "
        f"PAA_micro={r['PAA_micro']:.2f}%, N_mean={r['N_mean']}, models={int(r['Models_present'])}"
    )


# =========================
# Plot (use PAA_mean)
# =========================
fig = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="PAA_mean",
    hover_name="Country",
    hover_data={
        "PAA_mean": True,
        "PAA_micro": True,
        "N_mean": True,
        "Models_present": True,
        # 如需在 hover 里展示每个模型的 PAA，可取消注释下一行
        # **{f"PAA_{m}": True for m in MODEL_JSONS.keys()},
    },
    color_continuous_scale=px.colors.sequential.YlGnBu,
    range_color=(0, 100),
    labels={
        "PAA_mean": "Mean PAA (%)",
        "PAA_micro": "Micro PAA (%)",
        "N_mean": "Mean N",
        "Models_present": "#Models",
    },
)

fig.update_layout(
    title_text="Mean Preference Alignment Accuracy (PAA) Across 5 Models by Birth Country/Region",
    title_x=0.5,
    geo=dict(showframe=False, showcoastlines=True),
    font=dict(family="Arial", size=18),
    coloraxis_colorbar=dict(title="Mean PAA (%)", tickvals=[0, 20, 40, 60, 80, 100]),
    margin=dict(l=20, r=20, t=80, b=20),
)

# 保存为高分辨率图片，适合论文
os.makedirs(os.path.dirname("data/judge/chart/country_paa_map_5models_mean.pdf"), exist_ok=True)
fig.write_image(
    "data/judge/chart/country_paa_map_5models_mean.pdf",
    format="pdf",
    scale=8,
    width=2200,
    height=1200,
)
fig.show()
