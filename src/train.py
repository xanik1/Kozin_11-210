import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_loader import load_sample_data

GOOGLE_DRIVE_FILE_ID = '11fowrJG3CIhtk0_owLFFukzGJqQfY5eM'
CSV_URL = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'personality_model.joblib')

os.makedirs(MODEL_DIR, exist_ok=True)

try:
    X_train, X_test, y_train, y_test = load_sample_data(CSV_URL)
except Exception as e:
    raise e

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=7,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\n📌 Accuracy: {accuracy:.4f}')

try:
    joblib.dump(model, MODEL_PATH)
    feature_info = {
        'feature_names': model.feature_names_in_.tolist(),
        'feature_importances': model.feature_importances_.tolist()
    }
    joblib.dump(feature_info, os.path.join(MODEL_DIR, 'feature_info.joblib'))
except Exception as e:
    raise e

features = pd.DataFrame({
    'Feature': model.feature_names_in_,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
