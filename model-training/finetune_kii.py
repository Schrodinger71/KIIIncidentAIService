from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kii_methodology import (  # noqa: E402
    CATEGORY_NAMES,
    CRITERIA,
    FEATURE_INPUT_DIM,
    OBJECT_TYPE_LABELS,
    OWNERSHIP_LABELS,
    SERVICE_SCALE_LABELS,
    SECTOR_LABELS,
    as_bool,
    clamp_level,
    criterion_applicable_field,
    criterion_level_field,
    derive_category_level,
    feature_vector,
    normalize_choice,
    normalize_record,
    normalize_sectors,
)

RUSSIAN_HEADER_ALIASES = {
    "Наименование": "object_name",
    "Сферы": "sectors",
    "Сектор": "sectors",
    "Уровень": "ownership_level",
    "Масштаб": "service_scale",
    "Масштаб услуг": "service_scale",
    "Тип объекта": "object_type",
    "Критичность": "process_criticality",
    "Критичных процессов": "critical_process_count",
    "Интеграции": "interaction_count",
    "Интеграций и связей": "interaction_count",
    "Непрерывный режим": "continuous_operation",
    "АСУ ТП": "scada_used",
    "Чувствительная информация": "classified_info",
    "Категория": "category_level",
}

for criterion in CRITERIA:
    RUSSIAN_HEADER_ALIASES[f"{criterion['code']} применимо"] = criterion_applicable_field(criterion["id"])
    RUSSIAN_HEADER_ALIASES[f"{criterion['code']} уровень"] = criterion_level_field(criterion["id"])
    RUSSIAN_HEADER_ALIASES[criterion["title"]] = criterion_level_field(criterion["id"])

REQUIRED_COLUMNS = [
    "sectors",
    "ownership_level",
    "service_scale",
    "object_type",
    "process_criticality",
    "critical_process_count",
    "interaction_count",
    "continuous_operation",
    "scada_used",
    "classified_info",
]
OPTIONAL_TARGET_COLUMN = "category_level"


def detect_default_data_path() -> Path:
    candidates = [
        Path("synthetic_kii_data.csv"),
        Path("../model-training/synthetic_kii_data.csv"),
        Path("../Tools/DataConverter/kii_converted.xlsx"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_PATH = Path(os.getenv("DATA_PATH", str(detect_default_data_path())))
BASE_MODEL_PATH = Path(os.getenv("BASE_MODEL", "../inference-api/models/model.keras"))
OUTPUT_MODEL_KERAS = Path(os.getenv("OUTPUT_MODEL_KERAS", "../inference-api/models/model.keras"))
OUTPUT_MODEL_H5 = Path(os.getenv("OUTPUT_MODEL_H5", "../inference-api/models/model.h5"))
EPOCHS = int(os.getenv("EPOCHS", "40"))
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


def maybe_to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=RUSSIAN_HEADER_ALIASES).copy()
    if "sector" in renamed.columns and "sectors" not in renamed.columns:
        renamed["sectors"] = renamed["sector"]
    return renamed


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    standardized = rename_columns(df)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in standardized.columns]
    if missing_columns:
        raise ValueError(f"В датасете не хватает колонок: {', '.join(missing_columns)}")

    standardized["object_name"] = standardized.get("object_name", "Объект КИИ").fillna("Объект КИИ")
    standardized["sectors"] = standardized["sectors"].apply(normalize_sectors)
    standardized["ownership_level"] = standardized["ownership_level"].apply(
        lambda value: normalize_choice(value, OWNERSHIP_LABELS, "ownership_level")
    )
    standardized["service_scale"] = standardized["service_scale"].apply(
        lambda value: normalize_choice(value, SERVICE_SCALE_LABELS, "service_scale")
    )
    standardized["object_type"] = standardized["object_type"].apply(
        lambda value: normalize_choice(value, OBJECT_TYPE_LABELS, "object_type")
    )

    for column in ("process_criticality", "critical_process_count", "interaction_count"):
        standardized[column] = standardized[column].apply(maybe_to_int)

    standardized["process_criticality"] = standardized["process_criticality"].clip(lower=1, upper=10)
    standardized["critical_process_count"] = standardized["critical_process_count"].clip(lower=0)
    standardized["interaction_count"] = standardized["interaction_count"].clip(lower=0)

    for column in ("continuous_operation", "scada_used", "classified_info"):
        standardized[column] = standardized[column].apply(as_bool)

    for criterion in CRITERIA:
        applicable_field = criterion_applicable_field(criterion["id"])
        level_field = criterion_level_field(criterion["id"])

        if applicable_field not in standardized.columns:
            standardized[applicable_field] = False
        if level_field not in standardized.columns:
            standardized[level_field] = 0

        standardized[applicable_field] = standardized[applicable_field].apply(as_bool)
        standardized[level_field] = standardized[level_field].apply(clamp_level)

    if OPTIONAL_TARGET_COLUMN in standardized.columns:
        standardized[OPTIONAL_TARGET_COLUMN] = standardized[OPTIONAL_TARGET_COLUMN].apply(maybe_to_int)
    else:
        standardized[OPTIONAL_TARGET_COLUMN] = standardized.apply(derive_category_level, axis=1)

    standardized[OPTIONAL_TARGET_COLUMN] = standardized.apply(derive_category_level, axis=1)
    standardized = standardized[standardized[OPTIONAL_TARGET_COLUMN].between(0, 3)].copy()
    if standardized.empty:
        raise ValueError("После очистки в датасете не осталось валидных строк с category_level в диапазоне 0..3.")

    standardized["sectors"] = standardized["sectors"].apply(lambda values: "|".join(values))
    return standardized


def row_to_record(row: pd.Series) -> dict[str, Any]:
    record = row.to_dict()
    record["sectors"] = normalize_sectors(record.get("sectors"))
    return normalize_record(record)


def build_training_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    records = [row_to_record(row) for _, row in df.iterrows()]
    X = np.stack([feature_vector(record) for record in records]).astype(np.float32)
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
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.15),
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
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
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
    print("=" * 72)
    print("ДООБУЧЕНИЕ МОДЕЛИ КИИ ПО ПОКАЗАТЕЛЯМ МЕТОДИКИ 8.3.1-8.3.14")
    print("=" * 72)
    print(f"Ожидаемая размерность входа модели: {FEATURE_INPUT_DIM}")
    print(f"Поддерживаемые сферы: {', '.join(SECTOR_LABELS.values())}")

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

    print("\n" + "=" * 72)
    print("ГОТОВО. Модель сохранена в inference-api/models и совместима с новой схемой.")
    print("Для проверки API запустите model-training/validate_inference.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
