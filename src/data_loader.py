import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(path: str):
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
    binary_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    if 'Personality' in df.columns:
        le = LabelEncoder()
        y = le.fit_transform(df['Personality'])
        X = df.drop(columns=['Personality'])
    else:
        X = df
        y = None

    return X, y


def load_and_preprocess(path: str):
    df = load_data(path)
    return preprocess_data(df)


def load_sample_data(path: str, test_size=0.2, random_state=42):
    X, y = load_and_preprocess(path)
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
