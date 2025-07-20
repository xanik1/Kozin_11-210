import os
import joblib
from datetime import datetime
from data_loader import load_data, preprocess_data

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'models', 'personality_model.joblib'
)
PRED_PATH = 'predictions.csv'
REPORT_PATH = 'report.html'

GOOGLE_DRIVE_FILE_ID = '11fowrJG3CIhtk0_owLFFukzGJqQfY5eM'
CSV_URL = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    raw_data = load_data(CSV_URL).head(10)
    X, _ = preprocess_data(raw_data)
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    preds = model.predict(X)
    raw_data['Predicted_Personality'] = [
        'Extrovert' if p == 1 else 'Introvert' for p in preds
    ]
    raw_data.to_csv(PRED_PATH, index=False)
    print(f"✅ Предсказания сохранены в {PRED_PATH}")

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Отчёт о предсказаниях</title>
</head>
<body>
    <h1>🧠 Personality Prediction Report</h1>
    <p><strong>Дата:</strong> {
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }</p>
    <p><strong>Количество людей:</strong> {len(preds)}</p>
    <p><strong>Предсказания:</strong></p>
    {raw_data[['Predicted_Personality']].to_html(index=False)}
</body>
</html>
"""
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"📄 Отчёт сохранён в {REPORT_PATH}")


if __name__ == "__main__":
    main()
