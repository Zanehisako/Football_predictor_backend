from pathlib import Path
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rapidfuzz import process, fuzz
import pandas as pd
import numpy as np
import joblib
import warnings
import re

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "football_model_final.pkl"



# ==============================================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==============================================================================
# ==============================================================================
# 1. LOAD MODEL & DATA
# ==============================================================================
try:
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts['model']
    features = artifacts['features']
    current_elos = artifacts['elo_dict']
    df_recent = artifacts['df_recent']
    print(f"✅ Model Loaded. Features: {len(features)}")
except Exception as e:
    print("❌ Error loading pickle:")
    traceback.print_exc()
    raise RuntimeError("Model failed to load") from e


ALIASES = {
    "barca": "Barcelona",
    "fcb": "Barcelona",
    "man u": "Manchester United",
    "man utd": "Manchester United",
    "psg": "Paris Saint Germain",
    "ath madrid": "Atletico Madrid",
    "inter": "Inter Milan",
}

def normalize(s):
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()


def find_team(team):
    team_n = normalize(team)

    teams_raw = pd.concat([
        df_recent['home_team_name'],
        df_recent['away_team_name']
    ]).unique()

    teams = {normalize(t): t for t in teams_raw}

    # 1️⃣ alias match
    if team_n in ALIASES:
        alias = normalize(ALIASES[team_n])
        if alias in teams:
            return teams[alias]

    # 2️⃣ exact normalized match
    if team_n in teams:
        return teams[team_n]

    # 3️⃣ containment match (very strong signal)
    for t_n, t in teams.items():
        if team_n in t_n or t_n in team_n:
            return t

    # 4️⃣ fuzzy match (STRICT)
    match, score, _ = process.extractOne(
        team_n,
        teams.keys(),
        scorer=fuzz.token_set_ratio
    )

    print(f"Fuzzy match: '{team}' → '{teams[match]}' ({score})")
    if score >= 50:
        return teams[match]

    return None


def get_stats(team:str):

    best_match = find_team(team)
    if best_match is None:
        raise ValueError(f"No reliable match for '{team}'")

    print(f"Matched '{team}' → '{best_match}'")


    rows = df_recent[
        (df_recent['home_team_name'] == best_match) |
        (df_recent['away_team_name'] == best_match)
    ]
    if len(rows) == 0: return None
    last = rows.sort_values('date').iloc[-1]
    prefix = 'home_' if last['home_team_name'] == team else 'away_'
    
    stats = {}
    # DYNAMIC EXTRACTION for 100 features
    valid_cols = [c for c in df_recent.columns if 'roll_' in c]
    for col in valid_cols:
        if col.startswith(prefix):
            generic_name = col.replace(prefix, '')
            stats[generic_name] = last[col]
    return stats



@app.get("/predict/{home_team}/{away_team}")
async def predict(home_team: str, away_team: str):
    h_team = find_team(home_team)
    a_team = find_team(away_team)
    h_elo = current_elos.get(h_team, 1500)
    a_elo = current_elos.get(a_team, 1500)
    h_stats, a_stats = get_stats(h_team), get_stats(a_team)


    # --- INPUT ---
    input_data = {
        'diff_elo': (h_elo + 70) - a_elo, 
        'home_elo': h_elo, 'away_elo': a_elo, 'diff_rest': 0, 
    }
    
    # Fill Features
    for feat in features:
        if feat in input_data: continue
        if 'home_roll_' in feat:
            key = feat.replace('home_roll_', '')
            input_data[feat] = h_stats.get(key, 0)
        elif 'away_roll_' in feat:
            key = feat.replace('away_roll_', '')
            input_data[feat] = a_stats.get(key, 0)
        elif 'diff_' in feat:
            key = feat.replace('diff_', 'roll_')
            input_data[feat] = h_stats.get(key, 0) - a_stats.get(key, 0)

    # Predict
    input_df = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)
    probs = model.predict_proba(input_df)[0]
    p_away, p_draw, p_home = probs[0], probs[1], probs[2]
    winner ="Draw" if p_draw > p_home and p_draw > p_away else "Undecided" if p_draw > p_home and p_draw > p_away else home_team if p_home > p_away else away_team if p_away > p_home else "Undecided"
    return {"probabilities": probs.tolist(),
            "prediction": find_team(winner) if winner != "Draw" and winner != "Undecided" else winner,
            "message": f"Match between {find_team(home_team)} and {find_team(away_team)}"
            }