from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os


try:
    from src.data_loader import preprocess_data
except ImportError:
    try:
        from data_loader import preprocess_data
    except ImportError as e:
        raise ImportError(
            "Failed to import preprocess_data: {}".format(str(e))
        )

app = FastAPI(
    title="Personality Type Predictor",
)

MODEL_PATH = "models/personality_model.joblib"
FEATURE_INFO_PATH = "models/feature_info.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
if not os.path.exists(FEATURE_INFO_PATH):
    raise FileNotFoundError(
        f"Feature info file not found at {FEATURE_INFO_PATH}"
    )

try:
    model = joblib.load(MODEL_PATH)
    feature_info = joblib.load(FEATURE_INFO_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")


class PersonalityFeatures(BaseModel):
    Time_spent_Alone: float
    Stage_fear: str
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: str
    Friends_circle_size: float
    Post_frequency: float


@app.post("/predict")
async def predict_personality(data: PersonalityFeatures):
    input_data = {
        'Time_spent_Alone': [data.Time_spent_Alone],
        'Stage_fear': [data.Stage_fear],
        'Social_event_attendance': [data.Social_event_attendance],
        'Going_outside': [data.Going_outside],
        'Drained_after_socializing': [data.Drained_after_socializing],
        'Friends_circle_size': [data.Friends_circle_size],
        'Post_frequency': [data.Post_frequency]
    }
    df_raw = pd.DataFrame(input_data)

    try:
        X, _ = preprocess_data(df_raw)
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Data preprocessing error: {str(e)}"
        )

    try:
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].max()
        personality_type = "Extrovert" if prediction == 1 else "Introvert"
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

    return {
        "personality_type": personality_type,
        "probability": float(probability),
        "features_importance": dict(zip(
            feature_info['feature_names'],
            feature_info['feature_importances']
        ))
    }


@app.get("/report")
async def get_report():
    report_path = "report.html"
    if not os.path.exists(report_path):
        raise HTTPException(
            status_code=404,
            detail="Report file not found. Generate report first."
        )
    return FileResponse(report_path, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
