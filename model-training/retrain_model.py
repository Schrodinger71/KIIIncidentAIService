import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import sys

# --- Конфигурация ---
NEW_DATA_PATH = os.getenv('DATA_PATH', 'synthetic_kii_data.csv')
OUTPUT_MODEL_PATH = os.getenv('OUTPUT_MODEL_PATH', '../inference-api/models/model_v2.keras')
OUTPUT_MODEL_H5_PATH = os.getenv('OUTPUT_MODEL_H5_PATH', '../inference-api/models/model_v2.h5')
OUTPUT_SCALER_PATH = '../inference-api/models/scaler.pkl'
OUTPUT_ENCODERS_PATH = '../inference-api/models/label_encoders.pkl'

def load_and_preprocess_data(filepath):
    """Загружает и предобрабатывает данные."""
    print(f"Загрузка данных из {filepath}...")
    df = pd.read_csv(filepath)
    
    # Разделение на признаки и целевую переменную
    # Убираем object_name, он не нужен для обучения
    feature_cols = [col for col in df.columns if col not in ['category_level', 'object_name']]
    X = df[feature_cols].copy()
    y = df['category_level'].values
    
    # Кодирование категориальных признаков
    categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    print(f"Категориальные признаки: {categorical_cols}")
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} уникальных значений")
    
    # Масштабирование числовых признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Итого признаков: {X_scaled.shape[1]}")
    return X_scaled, y, scaler, label_encoders

def create_model(input_dim):
    """Создаёт новую модель с правильной архитектурой."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 класса
    ])
    return model

def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ НОВОЙ МОДЕЛИ ДЛЯ КАТЕГОРИРОВАНИЯ КИИ")
    print("=" * 60)
    
    # 1. Загрузка и предобработка данных
    try:
        X, y, scaler, label_encoders = load_and_preprocess_data(NEW_DATA_PATH)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл '{NEW_DATA_PATH}' не найден.")
        print("Сначала запустите: python generate_synthetic_data.py")
        sys.exit(1)
    
    # 2. Разделение на выборки
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nОбучающая выборка: {X_train.shape[0]} записей")
    print(f"Валидационная выборка: {X_val.shape[0]} записей")
    
    # Проверка баланса классов
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nРаспределение классов в обучении:")
    for u, c in zip(unique, counts):
        names = ['Без категории', 'Третья', 'Вторая', 'Первая']
        print(f"  Класс {u} ({names[u]}): {c} записей ({c/len(y_train)*100:.1f}%)")
    
    # 3. Создание модели
    print(f"\nСоздание модели (входных признаков: {X_train.shape[1]})...")
    model = create_model(X_train.shape[1])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # 4. Обучение с колбэками
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        OUTPUT_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
    )
    
    print("\nЗапуск обучения...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    # 5. Оценка на валидации
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nРезультаты на валидации:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_accuracy:.2%}")
    
    # 6. Сохранение артефактов
    print(f"\nСохранение артефактов...")
    
    # Сохраняем в двух форматах для совместимости
    model.save(OUTPUT_MODEL_PATH)
    print(f"  Модель (Keras v3): {OUTPUT_MODEL_PATH}")
    
    try:
        model.save(OUTPUT_MODEL_H5_PATH)
        print(f"  Модель (H5): {OUTPUT_MODEL_H5_PATH}")
    except:
        print("  H5 формат недоступен, пропускаем")
    
    joblib.dump(scaler, OUTPUT_SCALER_PATH)
    print(f"  Скейлер: {OUTPUT_SCALER_PATH}")
    
    joblib.dump(label_encoders, OUTPUT_ENCODERS_PATH)
    print(f"  Кодировщики: {OUTPUT_ENCODERS_PATH}")
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print(f"Новая модель: {OUTPUT_MODEL_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
    