import json
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# ------- Country Name Mapping (Abbreviation/Replacement of Long Names) -------
COUNTRY_ABBREV = {
    'Prefer not to say': 'Unstated',
    'Dominican Republic': 'Do.',
    'Romania': 'Rom.',
    'Finland': 'Fin.',
    'United Arab Emirates': 'UAE',
    'Russian Federation': 'Rus.',
    # 'Estonia': 'Est.',
    'Portugal': 'Por.',
    'Korea, Republic of': 'S. Korea',
    'Venezuela, Bolivarian Republic of': 'Venezuela',
    'Bosnia and Herzegovina': 'Bosnia-Herz.',
    'Hong Kong': 'Hong Kong SAR, China',
    # 'France': 'Fra.',
    # 'Denmark': 'Den.',
}
def abbreviate_country(name):
    return COUNTRY_ABBREV.get(name, name)

def stat_from_json(filename):
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    # Compatible with list/dict
    if isinstance(data, dict) and 'results' in data:
        records = data['results']
    else:
        records = data

    total, match = {}, {}
    for r in records:
        profile = r.get('profile', '')
        match_obj = re.search(r'Birth Country/Region:\s*([^\n]+)', profile)
        if not match_obj:
            continue
        country = match_obj.group(1).strip()
        country = abbreviate_country(country)
        judge = r.get('final_label')
        if country not in total:
            total[country] = 0
            match[country] = 0
        total[country] += 1
        if judge and "W" in str(judge):
            match[country] += 1
    stat = {}
    for country in total:
        acc = match[country] / total[country]
        stat[country] = acc
    return stat

# ------ Read data ------
files = [
    ('GPT-4o', 'data/judge/gpt-4o_multi_value_judge_pairwise_polyjudge_recomputed.json'),
    ('Doubao-1.5-Pro', 'data/judge/doubao-1.5-pro-32k_multi_value_judge_pairwise_polyjudge.json'),
    ('DeepSeek-v3', 'data/judge/deepSeek-v3-fast_multi_value_judge_pairwise_polyjudge.json'),
    ('Claude-Sonnet-4.5', 'data/judge/claude-sonnet-4-5_multi_value_judge_pairwise_polyjudge.json'),
    ('Mistral-Medium-3', 'data/judge/mistral-medium_multi_value_judge_pairwise_polyjudge.json'),
]

all_stats = {}
all_countries = set()
for model, file in files:
    stat = stat_from_json(file)
    all_stats[model] = stat
    all_countries.update(stat.keys())

# Maintain consistency in the order of countries in each model
country_list = sorted(list(all_countries))

# --------- Draw radar chart function ---------
def plot_radar(country_rates_dict, countries, save_path="radar_multi_model.pdf"):
    N = len(countries)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(max(8, 0.25*N), 8), dpi=300)
    ax = plt.subplot(111, polar=True)

    # Support multiple models
    model_colors = ['#4E79A7', '#59A14F', '#F28E2B', '#B07AA1', '#76B7B2']
    for i, (model_name, rates) in enumerate(country_rates_dict.items()):
        values = [rates.get(c, 0) for c in countries]
        values += values[:1]
        ax.plot(angles, values, label=model_name, linewidth=2, color=model_colors[i % len(model_colors)])
        ax.fill(angles, values, alpha=0.15, color=model_colors[i % len(model_colors)])

    ax.set_thetagrids(np.degrees(angles[:-1]), countries, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(0)
    ax.yaxis.set_tick_params(labelsize=10)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"])
    plt.title("Preference Alignment Rates of LLMs Across Countries/Regions", fontsize=14, pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()

# --------- Main program call ---------
print("all_stats : {}".format(all_stats))
plot_radar(all_stats, country_list, save_path="data/judge/chart/alignment_radar_multi_model_judge_final.pdf")
