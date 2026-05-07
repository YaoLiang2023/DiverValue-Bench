import json
import re
import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------- 1. Regional Mapping Table (Customizable Extension) -------------------
country_region_map = {
    # Western
    "United States": "Western", "Canada": "Western", "United Kingdom": "Western", "Australia": "Western",
    "Ireland": "Western", "Belgium": "Western", "France": "Western", "Finland": "Western",
    "Denmark": "Western", "Latvia": "Western", "Switzerland": "Western", "Iceland": "Western",
    "Austria": "Western", "Slovenia": "Western", "Luxembourg": "Western", "Netherlands": "Western",
    "Germany": "Western", "Spain": "Western", "Italy": "Western", "Portugal": "Western",

    # East Asia
    "China": "East Asia", "Hong Kong": "East Asia", "Japan": "East Asia", "Korea, Republic of": "East Asia",

    # Southeast Asia
    "Philippines": "Southeast Asia", "Malaysia": "Southeast Asia", "Indonesia": "Southeast Asia",
    "Vietnam": "Southeast Asia",

    # South Asia
    "India": "South Asia", "Bangladesh": "South Asia", "Nepal": "South Asia", "Sri Lanka": "South Asia", "Pakistan": "South Asia",

    # Africa
    "South Africa": "Africa", "Zambia": "Africa", "Kenya": "Africa", "Nigeria": "Africa",
    "Ghana": "Africa", "Sudan": "Africa", "Malawi": "Africa", "Côte d'Ivoire": "Africa",

    # Eastern Europe
    "Poland": "Eastern Europe", "Romania": "Eastern Europe", "Russian Federation": "Eastern Europe",
    "Estonia": "Eastern Europe", "Czechia": "Eastern Europe", "Hungary": "Eastern Europe",
    "Slovakia": "Eastern Europe", "Bulgaria": "Eastern Europe", "Ukraine": "Eastern Europe",
    "Lithuania": "Eastern Europe", "Belarus": "Eastern Europe",

    # Middle East
    "Israel": "Middle East", "Kuwait": "Middle East", "Türkiye": "Middle East",

    # Latin America
    "Mexico": "Latin America", "Chile": "Latin America", "Brazil": "Latin America", "Argentina": "Latin America",
    "Colombia": "Latin America", "Venezuela, Bolivarian Republic of": "Latin America",
    "Dominican Republic": "Latin America", "Jamaica": "Latin America", "Guyana": "Latin America",
    "Cuba": "Latin America", "Honduras": "Latin America",

    # Oceania
    "New Zealand": "Oceania",

    # Other or Unspecified
    "Prefer not to say": "Unspecified"
}
def get_region(country):
    return country_region_map.get(country, "Other")

# ------------------- 2. General Profile Field Extraction -------------------
def extract_profile(profile_str):
    res = {}
    patterns = {
        'Age': r'Age: ([^\n]+)',
        'Gender': r'Gender: ([^\n]+)',
        'Education': r'Education: ([^\n]+)',
        'Birth Country': r'Birth Country: ([^\n]+)',
        'Marital Status': r'Marital Status: ([^\n]+)'
    }
    for k, p in patterns.items():
        m = re.search(p, profile_str)
        res[k] = m.group(1).strip() if m else ''
    return res

# ------------------- 3. Load and preprocess data -------------------
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for r in data:
        if "profile" in r and r["profile"]:
            profile = extract_profile(r["profile"])
            record = {**profile, "final_label": r.get("final_label", "")}
            records.append(record)
    return records

# ------------------- 4. Regional grouping preference consistency rate -------------------
def group_stat_region(records, group_key, region_key='Birth Country'):
    stat = {}
    for r in records:
        region = get_region(r[region_key])
        if region not in stat:
            stat[region] = {}
        key = r[group_key]
        if key not in stat[region]:
            stat[region][key] = {'matched': 0, 'total': 0}
        if r["final_label"] and "W" in r["final_label"]:
            stat[region][key]['matched'] += 1
        stat[region][key]['total'] += 1
    region_rate = {
        region: {k: (v['matched']/v['total'] if v['total']>0 else 0) for k,v in d.items()}
        for region, d in stat.items()
    }
    return region_rate

# ------------------- 5. Read all model results -------------------
model_files = {
    "GPT-4o": "data/rejudge/gpt-4o_multi_value_rejudge_pairwise_polyjudge_recomputed.json",
    "Doubao-1.5-Pro": "data/rejudge/doubao-1.5-pro-32k_multi_value_rejudge_pairwise_polyjudge.json",
    "DeepSeek-v3": "data/rejudge/deepSeek-v3-fast_multi_value_rejudge_pairwise_polyjudge.json",
    "Claude-Sonnet-4.5": "data/rejudge/claude-sonnet-4-5_multi_value_rejudge_pairwise_polyjudge.json",
    "Mistral-Medium-3": "data/rejudge/mistral-medium_multi_value_rejudge_pairwise_polyjudge.json",
}
model_results = {model: load_data(path) for model, path in model_files.items()}

# ------------------- 6. Statistics of labels and consistency rates for all regions -------------------
fields = ["Age", "Gender", "Education", "Marital Status"]
all_labels = {field: set() for field in fields}
all_rates_region = {field: {} for field in fields}
for field in fields:
    for model, records in model_results.items():
        rates_region = group_stat_region(records, field)
        for region, rates in rates_region.items():
            all_labels[field].update(rates.keys())
            if region not in all_rates_region[field]:
                all_rates_region[field][region] = {}
            all_rates_region[field][region][model] = rates

for field in fields:
    all_labels[field] = sorted(list(all_labels[field]))

# ------------------- 7. Draw regional radar map -------------------
def plot_radar(field, labels, rates_dict, region, save_path):
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8), dpi=300)
    plt.subplot(111, polar=True)
    model_colors = {
        "GPT-4o": "#4E79A7",
        "Doubao-1.5-Pro": "#59A14F",
        "DeepSeek-v3": "#F28E2B",
        "Claude-Sonnet-4.5": "#B07AA1",
        "Mistral-Medium-3": "#76B7B2"
    }

    for model, rates in rates_dict.items():
        values = [rates.get(l, 0) for l in labels]
        values += values[:1]
        plt.plot(angles, values, label=model, linewidth=2, color=model_colors.get(model, None))
        plt.fill(angles, values, alpha=0.15, color=model_colors.get(model, None))

    plt.xticks(angles[:-1], labels, size=16)
    plt.yticks(np.arange(0, 1.1, 0.2), [f"{int(v * 100)}%" for v in np.arange(0, 1.1, 0.2)], color="grey", size=16)
    plt.title(f"{field} ({region})", size=20, y=1.08)
    plt.ylim(0, 1.0)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ------------------- 8. Loop output all regional/dimensional radar maps -------------------
output_dir = "data/rejudge/chart/"
for field in fields:
    for region in all_rates_region[field]:
        plot_radar(
            field,
            all_labels[field],
            {model: all_rates_region[field][region].get(model, {}) for model in model_files},
            region=region,
            save_path=f"{output_dir}radar_{field.replace(' ', '_').lower()}_{region}.pdf"
        )

print("Three model radar maps of each area/group have been saved to", output_dir)

for region in sorted({r for field in fields for r in all_rates_region[field]}):
    print(f"\n==============================")
    print(f"Region：{region}")
    print(f"==============================")
    for field in fields:
        print(f"\n  [Dimension] {field}:")
        if region not in all_rates_region[field]:
            print("    (No data)")
            continue
        for model in model_files:
            print(f"    - {model}:")
            rates = all_rates_region[field][region].get(model, {})
            for group in all_labels[field]:
                rate = rates.get(group, 0)
                print(f"        {group}: {rate:.4f}")
