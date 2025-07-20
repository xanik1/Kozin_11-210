from src.data_loader import load_sample_data

TEST_GOOGLE_DRIVE_FILE_ID = '11fowrJG3CIhtk0_owLFFukzGJqQfY5eM'
TEST_CSV_URL = f'https://drive.google.com/uc?id={TEST_GOOGLE_DRIVE_FILE_ID}'


def test_data_loading_from_google_drive():
    X_train, X_test, y_train, y_test = load_sample_data(TEST_CSV_URL)

    assert X_train.shape[0] > 0, "Тренировочная выборка пуста"
    assert X_test.shape[0] > 0, "Тестовая выборка пуста"
    assert X_train.shape[1] == X_test.shape[1], (
        "Количество признаков в train и test не совпадает"
    )
    assert len(y_train) == X_train.shape[0], (
        "Размерности X_train и y_train не совпадают"
    )
    assert len(y_test) == X_test.shape[0], (
        "Размерности X_test и y_test не совпадают"
    )

    expected_features = [
        'Time_spent_Alone',
        'Stage_fear',
        'Social_event_attendance',
        'Going_outside',
        'Drained_after_socializing',
        'Friends_circle_size',
        'Post_frequency'
    ]
    for feature in expected_features:
        assert feature in X_train.columns, f"Признак {feature} отсутствует"

    print(f"✅ Успешно загружено: {X_train.shape[0]} train, "
          f"{X_test.shape[0]} test, {X_train.shape[1]} признаков")


def test_class_distribution():
    X_train, X_test, y_train, y_test = load_sample_data(TEST_CSV_URL)

    assert set(y_train) == {0, 1}, "В трен данных должны быть оба класса"
    assert set(y_test) == {0, 1}, "В тестовых данных должны быть оба класса"

    assert 0.3 < (
        y_train == 1
    ).mean() < 0.7, "Неразумное соотношение классов в train"
    assert 0.3 < (
        y_test == 1
    ).mean() < 0.7, "Неразумное соотношение классов в test"
