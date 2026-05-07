import json
from collections import defaultdict

import pandas as pd
import plotly.express as px


# =========================
# Config
# =========================
# Evaluation result (poly report) JSON: Sample by sample record+final_1abel (W/L/TIE)
JSON_PATH = "data/rejudge/gpt-4o_multi_value_rejudge_pairwise_polyjudge_recomputed.json"

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
    """
    A profile can be either a string (multiple lines of 'Key: Value') or a dict.
    Return the original value of the Birth Country field (not standardized).
    """
    if profile is None:
        return None

    if isinstance(profile, dict):
        # Common key name compatibility
        for k in [
            "Birth Country",
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
            if s.lower().startswith("birth country"):
                parts = s.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return None

    return None


def normalize_country(name):
    """Standardized country name; Returning None indicates skipping (e.g. unspecified country/region)."""
    if not name:
        return None
    name = str(name).strip()
    name = COUNTRY_NORMALIZE.get(name, name)

    low = name.lower()
    if low in {"prefer not to say", "unstated", "unknown", "n/a", "none", "null"}:
        return None
    return name


def load_items(json_path):
    """Compatible with JSON top-level lists or dicts."""
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    # Some export formats may be {"data": [...]} or {id: sample,...}
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        # fallback：Treat value as a sample
        return list(obj.values())

    raise ValueError("Unsupported JSON top-level type: {}".format(type(obj)))


# =========================
# Aggregate: country -> PAA (%)
# =========================
items = load_items(JSON_PATH)

totals = defaultdict(int)
aligned = defaultdict(int)

for inst in items:
    country_raw = extract_birth_country(inst.get("profile"))
    country = normalize_country(country_raw)
    if country is None:
        continue

    totals[country] += 1

    # Conservative: Only final_1abel=="W" is counted as aligned; TIE/L are considered non aligned
    label = inst.get("final_label") or inst.get("label")
    if label == "W":
        aligned[country] += 1

data = []
for country, n in totals.items():
    paa = 100.0 * aligned[country] / n if n else 0.0
    data.append({"Country": country, "Agreement": round(paa, 2), "N": n})

df = pd.DataFrame(data).sort_values("Agreement", ascending=False)

# Optional: Save aggregation results for easy reproduction and verification
df.to_csv("gpt4o_country_paa.csv", index=False, encoding="utf-8-sig")


# =========================
# Plot
# =========================
fig = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="Agreement",
    hover_name="Country",
    hover_data={"Agreement": True, "N": True},
    color_continuous_scale=px.colors.sequential.YlGnBu,
    range_color=(0, 100),
    labels={"Agreement": "Alignment Rate (%)", "N": "N (instances)"},
)

fig.update_layout(
    title_text="Alignment Rates of GPT-4o with Human Value Preferences Across Countries/Regions of Birth",
    title_x=0.5,
    geo=dict(showframe=False, showcoastlines=True),
    font=dict(family="Arial", size=18),
    coloraxis_colorbar=dict(title="Alignment Rate (%)", tickvals=[0, 20, 40, 60, 80, 100]),
    margin=dict(l=20, r=20, t=80, b=20),
)


fig.write_image("data/rejudge/chart/country_agreement_map_gpt-4o.pdf", format="pdf", scale=8, width=2200, height=1200)
# fig.write_image("data/rejudge/chart/country_agreement_map_gpt-4o.png", scale=8, width=2200, height=1200)
# fig.write_image("data/rejudge/chart/country_agreement_map_gpt-4o.svg", scale=2, width=2200, height=1200)

fig.show()
