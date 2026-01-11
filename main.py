from fastapi import FastAPI

app = FastAPI()


@app.get("/predict/{home_team}/{away_team}")
async def predict(home_team: str, away_team: str):
    return {"message": f"Predicting match between {home_team} and {away_team}"}