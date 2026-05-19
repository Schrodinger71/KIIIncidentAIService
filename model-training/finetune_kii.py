from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


SECTORS = [
    "energy",
    "health",
    "finance",
    "transport",
    "government",
    "communications",
    "industry",
]
OWNERSHIP_LEVELS = ["federal", "regional", "municipal", "private"]
SERVICE_SCALES = ["local", "regional", "federal", "intersectoral"]
CATEGORY_NAMES = ["Без категории", "Третья", "Вторая", "Первая"]

SECTOR_CODE_MAP = {
    0: "Энергетика",
    1: "Транспорт",
    2: "Связь",
    3: "Здравоохранение",
    4: "Банковская сфера",
    5: "Оборонная промышленность",
    6: "Государственное управление",
    7: "Наука",
    8: "Топливная промышленность",
}
LEVEL_CODE_MAP = {0: "Федеральный", 1: "Региональный", 2: "Муниципальный", 3: "Объектовый"}
SCALE_CODE_MAP = {
    0: "Вся страна",
    1: "Федеральный округ",
    2: "Субъект РФ",
    3: "Несколько муниципалитетов",
    4: "Один город",
}
CRITICALITY_CODE_MAP = {0: "Критическая", 1: "Высокая", 2: "Средняя", 3: "Низкая"}

SECTOR_TO_API = {
    "Энергетика": "energy",
    "Транспорт": "transport",
    "Связь": "communications",
    "Здравоохранение": "health",
    "Банковская сфера": "finance",
    "Оборонная промышленность": "industry",
    "Государственное управление": "government",
    "Наука": "industry",
    "Топливная промышленность": "energy",
    "energy": "energy",
    "health": "health",
    "finance": "finance",
    "transport": "transport",
    "government": "government",
    "communications": "communications",
    "industry": "industry",
}
OWNERSHIP_TO_API = {
    "Федеральный": "federal",
    "Региональный": "regional",
    "Муниципальный": "municipal",
    "Объектовый": "private",
    "federal": "federal",
    "regional": "regional",
    "municipal": "municipal",
    "private": "private",
}
SERVICE_SCALE_TO_API = {
    "Вся страна": "federal",
    "Федеральный округ": "federal",
    "Субъект РФ": "regional",
    "Несколько муниципалитетов": "local",
    "Один город": "local",
    "local": "local",
    "regional": "regional",
    "federal": "federal",
    "intersectoral": "intersectoral",
}
CRITICALITY_TO_SCORE = {
    "Критическая": 10,
    "Высокая": 8,
    "Средняя": 5,
    "Низкая": 2,
}

RAW_TO_API_COLUMNS = {
    "level": "ownership_level",
    "num_users": "supported_users",
    "num_territories": "territories_count",
    "critical_processes_count": "critical_processes",
    "integrations_count": "interactions_count",
    "affected_employees": "employees_affected",
    "uses_automated_control_system": "scada_used",
    "provides_gov_services": "government_services",
    "life_health_impact": "life_safety_impact",
    "transport_impact": "transport_disruption",
    "communication_impact": "communications_disruption",
    "sensitive_info": "classified_info",
}
RUSSIAN_HEADERS = {
    "Наименование": "object_name",
    "Сектор": "sector",
    "Уровень": "ownership_level",
    "Масштаб услуг": "service_scale",
    "Критичность": "process_criticality",
    "Пользователи": "supported_users",
    "Территории": "territories_count",
    "Финущерб": "predicted_financial_damage",
    "Восстановление": "recovery_time_hours",
    "Крит.процессы": "critical_processes",
    "Интеграции": "interactions_count",
    "Субъекты ПДн": "personal_data_subjects",
    "Сотрудники": "employees_affected",
    "Непрерывный": "continuous_operation",
    "АСУ ТП": "scada_used",
    "Госуслуги": "government_services",
    "Жизнь/здоровье": "life_safety_impact",
    "Экология": "ecological_impact",
    "Оборона": "defense_impact",
    "Общ.порядок": "public_order_impact",
    "Транспорт": "transport_disruption",
    "Связь": "communications_disruption",
    "Чувств.инфо": "classified_info",
    "Категория": "category_level",
}
BOOLEAN_COLUMNS = [
    "continuous_operation",
    "scada_used",
    "government_services",
    "life_safety_impact",
    "ecological_impact",
    "defense_impact",
    "public_order_impact",
    "transport_disruption",
    "communications_disruption",
    "classified_info",
]
NUMERIC_COLUMNS = [
    "supported_users",
    "territories_count",
    "annual_financial_loss_million",
    "recovery_time_hours",
    "critical_processes",
    "interactions_count",
    "personal_data_subjects",
    "employees_affected",
]
REQUIRED_COLUMNS = [
    "sector",
    "ownership_level",
    "service_scale",
    "process_criticality",
    "supported_users",
    "territories_count",
    "annual_financial_loss_million",
    "recovery_time_hours",
    "critical_processes",
    "interactions_count",
    "personal_data_subjects",
    "employees_affected",
    *BOOLEAN_COLUMNS,
    "category_level",
]


def detect_default_data_path() -> Path:
    candidates = [
        Path("synthetic_kii_data.csv"),
        Path("../Tools/DataConverter/kii_converted.xlsx"),
        Path("../Tools/DataConverter/kii_data.xlsx"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_PATH = Path(os.getenv("DATA_PATH", str(detect_default_data_path())))
BASE_MODEL_PATH = Path(os.getenv("BASE_MODEL", "../inference-api/models/model.keras"))
OUTPUT_MODEL_KERAS = Path(os.getenv("OUTPUT_MODEL_KERAS", "../inference-api/models/model.keras"))
OUTPUT_MODEL_H5 = Path(os.getenv("OUTPUT_MODEL_H5", "../inference-api/models/model.h5"))
EPOCHS = int(os.getenv("EPOCHS", "50"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.2"))


def read_table(filepath: Path) -> pd.DataFrame:
    print(f"📂 Загрузка датасета: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    if filepath.suffix.lower() == ".csv":
        return pd.read_csv(filepath)
    if filepath.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(filepath)
    raise ValueError(f"Неподдерживаемый формат файла: {filepath.suffix}")


def maybe_decode_code(value: Any, mapping: dict[int, str]) -> Any:
    if pd.isna(value):
        return value
    if isinstance(value, str) and value.strip() == "":
        return value
    try:
        numeric_value = int(float(value))
    except (TypeError, ValueError):
        return value
    return mapping.get(numeric_value, value)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    return normalized in {"1", "+", "true", "yes", "y", "да", "истина"}


def normalize_process_criticality(value: Any) -> int:
    decoded = maybe_decode_code(value, CRITICALITY_CODE_MAP)
    if isinstance(decoded, (int, float)) and 1 <= int(decoded) <= 10:
        return int(decoded)
    if str(decoded).strip() in CRITICALITY_TO_SCORE:
        return CRITICALITY_TO_SCORE[str(decoded).strip()]
    raise ValueError(f"Не удалось распознать критичность процесса: {value}")


def normalize_categorical(value: Any, mapping: dict[str, str], decoder: dict[int, str] | None = None) -> str:
    decoded = maybe_decode_code(value, decoder or {})
    normalized = str(decoded).strip()
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"Не удалось распознать категориальное значение: {value}")


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.rename(columns=RUSSIAN_HEADERS).rename(columns=RAW_TO_API_COLUMNS).copy()

    if "predicted_financial_damage" in standardized.columns and "annual_financial_loss_million" not in standardized.columns:
        standardized["annual_financial_loss_million"] = (
            pd.to_numeric(standardized["predicted_financial_damage"], errors="coerce") / 1_000_000.0
        )

    for column in NUMERIC_COLUMNS:
        if column in standardized.columns:
            standardized[column] = pd.to_numeric(standardized[column], errors="coerce").fillna(0)

    if "category_level" in standardized.columns:
        standardized["category_level"] = pd.to_numeric(standardized["category_level"], errors="coerce")

    for column in BOOLEAN_COLUMNS:
        if column in standardized.columns:
            standardized[column] = standardized[column].apply(as_bool)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in standardized.columns]
    if missing_columns:
        raise ValueError(f"В датасете не хватает колонок: {', '.join(missing_columns)}")

    standardized["sector"] = standardized["sector"].apply(
        lambda value: normalize_categorical(value, SECTOR_TO_API, SECTOR_CODE_MAP)
    )
    standardized["ownership_level"] = standardized["ownership_level"].apply(
        lambda value: normalize_categorical(value, OWNERSHIP_TO_API, LEVEL_CODE_MAP)
    )
    standardized["service_scale"] = standardized["service_scale"].apply(
        lambda value: normalize_categorical(value, SERVICE_SCALE_TO_API, SCALE_CODE_MAP)
    )
    standardized["process_criticality"] = standardized["process_criticality"].apply(normalize_process_criticality)
    standardized["category_level"] = standardized["category_level"].fillna(-1).astype(int)

    standardized = standardized[standardized["category_level"].between(0, 3)].copy()
    if standardized.empty:
        raise ValueError("После очистки в датасете не осталось валидных строк с category_level в диапазоне 0..3.")

    return standardized


def api_feature_vector(row: pd.Series) -> np.ndarray:
    base_features = [
        row["process_criticality"] / 10.0,
        min(row["supported_users"] / 10_000_000.0, 1.0),
        min(row["territories_count"] / 89.0, 1.0),
        min(row["annual_financial_loss_million"] / 10_000.0, 1.0),
        min(row["recovery_time_hours"] / 168.0, 1.0),
        min(row["critical_processes"] / 50.0, 1.0),
        min(row["interactions_count"] / 200.0, 1.0),
        min(row["personal_data_subjects"] / 10_000_000.0, 1.0),
        min(row["employees_affected"] / 1_000_000.0, 1.0),
        float(row["continuous_operation"]),
        float(row["scada_used"]),
        float(row["government_services"]),
        float(row["life_safety_impact"]),
        float(row["ecological_impact"]),
        float(row["defense_impact"]),
        float(row["public_order_impact"]),
        float(row["transport_disruption"]),
        float(row["communications_disruption"]),
        float(row["classified_info"]),
    ]

    sector_flags = [1.0 if row["sector"] == item else 0.0 for item in SECTORS]
    ownership_flags = [1.0 if row["ownership_level"] == item else 0.0 for item in OWNERSHIP_LEVELS]
    service_scale_flags = [
        1.0 if row["service_scale"] == "regional" else 0.0,
        1.0 if row["service_scale"] == "federal" else 0.0,
        1.0 if row["service_scale"] == "intersectoral" else 0.0,
    ]

    return np.array(base_features + sector_flags + ownership_flags + service_scale_flags, dtype=np.float32)


def build_training_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([api_feature_vector(row) for _, row in df.iterrows()])
    y = df["category_level"].to_numpy(dtype=np.int32)
    print(f"  Объектов: {len(df)}, признаков на входе модели: {X.shape[1]}")

    unique, counts = np.unique(y, return_counts=True)
    distribution = ", ".join(f"{int(label)}={int(count)}" for label, count in zip(unique, counts))
    print(f"  Распределение классов: {distribution}")
    return X, y


def load_base_model(path: Path, expected_input_dim: int) -> tf.keras.Model | None:
    if not path.exists():
        print("🧠 Базовая модель не найдена, будет создана новая.")
        return None

    print(f"🧠 Попытка загрузки базовой модели: {path}")
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        print(f"  ⚠ Не удалось загрузить базовую модель: {exc}")
        return None

    input_dim = int(model.input_shape[-1])
    if input_dim != expected_input_dim:
        print(f"  ⚠ Размер входа базовой модели = {input_dim}, а нужен {expected_input_dim}. Создаю новую модель.")
        return None

    print("  ✓ Базовая модель совместима по размерности входа.")
    return model


def create_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(4, activation="softmax"),
        ]
    )
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fine_tuning: bool,
) -> tf.keras.Model:
    learning_rate = 1e-4 if fine_tuning else 1e-3
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    OUTPUT_MODEL_KERAS.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_MODEL_KERAS),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n🚀 Запуск обучения...")
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    return model


def evaluate_model(model: tf.keras.Model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    predictions = model.predict(X_val, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    print("\n📊 Матрица ошибок:")
    print(confusion_matrix(y_val, predicted_classes, labels=[0, 1, 2, 3]))

    print("\n📊 Отчет по классам:")
    print(
        classification_report(
            y_val,
            predicted_classes,
            labels=[0, 1, 2, 3],
            target_names=CATEGORY_NAMES,
            zero_division=0,
        )
    )

    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Validation loss: {loss:.4f}")
    print(f"  Validation accuracy: {accuracy:.2%}")
    return float(accuracy)


def save_artifacts(model: tf.keras.Model) -> None:
    print("\n💾 Сохранение артефактов...")
    OUTPUT_MODEL_KERAS.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_MODEL_KERAS)
    print(f"  ✓ Keras-модель: {OUTPUT_MODEL_KERAS}")

    try:
        model.save(OUTPUT_MODEL_H5)
        print(f"  ✓ H5-модель: {OUTPUT_MODEL_H5}")
    except Exception as exc:
        print(f"  ⚠ H5 сохранить не удалось: {exc}")


def main() -> None:
    print("=" * 68)
    print("ДООБУЧЕНИЕ МОДЕЛИ КИИ С ПРИЗНАКАМИ, СОВМЕСТИМЫМИ С INFERENCE API")
    print("=" * 68)

    raw_df = read_table(DATA_PATH)
    dataset = standardize_dataframe(raw_df)
    X, y = build_training_arrays(dataset)

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=VALIDATION_SPLIT,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        print("  ⚠ Стратифицированное разбиение недоступно, использую обычное.")
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=VALIDATION_SPLIT,
            random_state=42,
        )

    print(f"\n  Обучение: {len(X_train)} записей")
    print(f"  Валидация: {len(X_val)} записей")

    base_model = load_base_model(BASE_MODEL_PATH, expected_input_dim=X.shape[1])
    if base_model is None:
        model = create_model(X.shape[1])
        fine_tuning = False
        print("\n  Создана новая модель.")
    else:
        model = base_model
        fine_tuning = True
        print("\n  Загружена совместимая базовая модель, запускаю дообучение.")

    model.summary()
    train_model(model, X_train, y_train, X_val, y_val, fine_tuning=fine_tuning)
    evaluate_model(model, X_val, y_val)
    save_artifacts(model)

    print("\n" + "=" * 68)
    print("ГОТОВО. Модель сохранена в inference-api/models и совместима по схеме входа.")
    print("Для проверки через API запустите model-training/validate_inference.py")
    print("=" * 68)


if __name__ == "__main__":
    main()
