import pandas as pd
from src.data_loader import preprocess_data


def test_preprocess_data():
    # Пример минимального DataFrame для тестирования
    data = {
        'Time_spent_Alone': [4.0, 9.0],
        'Stage_fear': ['No', 'Yes'],
        'Social_event_attendance': [4.0, 0.0],
        'Going_outside': [6.0, 0.0],
        'Drained_after_socializing': ['No', 'Yes'],
        'Friends_circle_size': [13.0, 0.0],
        'Post_frequency': [5.0, 3.0],
        'Personality': ['Extrovert', 'Introvert']
    }

    df = pd.DataFrame(data)
    X, y = preprocess_data(df)

    # Проверки
    assert X.shape[0] == 2
    assert 'Stage_fear' in X.columns
    assert X['Stage_fear'].dtype == int
    assert 'Personality' not in X.columns
    assert y is not None
    assert len(y) == 2
    assert set(y) == {0, 1}
