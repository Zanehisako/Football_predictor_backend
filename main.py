from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rapidfuzz import process, fuzz
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')


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
    artifacts = joblib.load('models/football_model_final.pkl')
    model = artifacts['model']
    features = artifacts['features']
    current_elos = artifacts['elo_dict']
    df_recent = artifacts['df_recent']
    print(f"✅ Model Loaded. Features: {len(features)}")
    # print("sample features:", features[:10])
    # print("sample elos:", list(current_elos.items())[:5])
    # print("recent data shape:", df_recent['home_team_name'].unique())
except:
    print("❌ Error loading pickle.")
    exit()

def get_stats(team:str):
    # collect all unique team names
    teams = pd.concat([
        df_recent['home_team_name'],
        df_recent['away_team_name']
    ]).unique()

    # find closest match
    best_match, score, _ = process.extractOne(
        team.strip().lower(),
        teams,
        scorer=fuzz.token_sort_ratio
    )
    # if score < 70:
        # raise ValueError(f"No good match found for '{team}'")

    print(f"Matched '{team}' → '{best_match}' (score={score})")

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
async def predict(home_team: str, away_team: str, Odds_1: float, Odds_X: float, Odds_2: float):
    h_team, a_team = home_team, away_team
    f = {'Odds_1': Odds_1, 'Odds_X': Odds_X, 'Odds_2': Odds_2}
    h_stats, a_stats = get_stats(h_team), get_stats(a_team)
    h_elo, a_elo = current_elos.get(h_team, 1500), current_elos.get(a_team, 1500)
    

    # --- INPUT ---
    input_data = {
        'diff_elo': (h_elo + 70) - a_elo, 
        'home_elo': h_elo, 'away_elo': a_elo, 'diff_rest': 0, 
    }
    
    # Fill Market Odds (Required for this model)
    imp_h = 1 / f['Odds_1']
    imp_d = 1 / f['Odds_X']
    imp_a = 1 / f['Odds_2']
    m_sum = imp_h + imp_d + imp_a
    input_data['market_prob_home'] = imp_h / m_sum
    input_data['market_prob_draw'] = imp_d / m_sum
    input_data['market_prob_away'] = imp_a / m_sum
    input_data['has_odds'] = 1

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
    winner = home_team if p_home > p_away else away_team if p_away > p_home else "Draw"
    return {"probabilities": probs.tolist(),
            "prediction": winner,
            }