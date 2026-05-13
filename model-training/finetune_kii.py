import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# === КОНФИГУРАЦИЯ ===
DATA_PATH = os.getenv('DATA_PATH', '../Tools/DataConverter/kii_converted.xlsx')
BASE_MODEL_PATH = os.getenv('BASE_MODEL', '../inference-api/models/model.keras')
OUTPUT_MODEL_H5 = os.getenv('OUTPUT_MODEL', '../inference-api/models/model_finetuned.h5')
OUTPUT_MODEL_KERAS = '../inference-api/models/model_finetuned.keras'
OUTPUT_SCALER = '../inference-api/models/scaler_finetuned.pkl'
OUTPUT_ENCODERS = '../inference-api/models/encoders_finetuned.pkl'

# Категориальные признаки, которые нужно кодировать
CATEGORICAL_COLS = ['sector', 'level', 'service_scale', 'process_criticality']
SECTOR_MAP = {
    0: "Энергетика", 1: "Транспорт", 2: "Связь", 3: "Здравоохранение",
    4: "Банковская сфера", 5: "Оборонная промышленность",
    6: "Государственное управление", 7: "Наука", 8: "Топливная промышленность"
}

# === ЗАГРУЗКА ДАННЫХ ===
def load_data(filepath):
    print(f"📂 Загрузка: {filepath}")
    df = pd.read_excel(filepath)
    
    # Убираем object_name и category_level
    feature_cols = [c for c in df.columns if c not in ['object_name', 'category_level']]
    X = df[feature_cols].copy()
    y = df['category_level'].values.astype(int)
    
    print(f"  Признаков: {X.shape[1]}, Объектов: {X.shape[0]}")
    print(f"  Распределение категорий: 0={sum(y==0)}, 1={sum(y==1)}, 2={sum(y==2)}, 3={sum(y==3)}")
    
    return X, y

# === ПОДГОТОВКА ПРИЗНАКОВ ===
def preprocess(X, y, scaler=None, encoders=None, fit=True):
    """Кодирует категориальные признаки и масштабирует числовые."""
    X_proc = X.copy()
    
    # Инициализируем энкодеры
    if encoders is None:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col in X_proc.columns:
                le = LabelEncoder()
                if fit:
                    X_proc[col] = le.fit_transform(X_proc[col].astype(str))
                else:
                    X_proc[col] = le.transform(X_proc[col].astype(str))
                encoders[col] = le
    
    # Масштабирование
    if scaler is None and fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_proc)
    elif scaler is not None:
        X_scaled = scaler.transform(X_proc)
    else:
        X_scaled = X_proc.values
    
    return X_scaled, scaler, encoders

# === ЗАГРУЗКА БАЗОВОЙ МОДЕЛИ ===
def load_base_model(path):
    print(f"🧠 Загрузка базовой модели: {path}")
    
    # Пробуем H5 сначала (более совместимый)
    h5_path = path.replace('.keras', '.h5')
    
    if os.path.exists(h5_path):
        print("  ✓ Найден H5 формат, загружаю...")
        return tf.keras.models.load_model(h5_path, compile=False)
    
    if os.path.exists(path):
        try:
            print("  Загружаю .keras...")
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            print(f"  ⚠ Ошибка загрузки .keras: {e}")
            print("  Создаю новую модель с нуля...")
            return None
    
    print("  ⚠ Модель не найдена, создаю новую...")
    return None

# === СОЗДАНИЕ НОВОЙ МОДЕЛИ (если базовой нет) ===
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    return model

# === ДООБУЧЕНИЕ ===
def finetune(model, X_train, y_train, X_val, y_val):
    """Дообучает модель на новых данных с маленьким learning rate."""
    
    # Компилируем с маленьким шагом обучения (fine-tuning)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Маленький lr!
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Колбэки
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, 
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, 
            min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_MODEL_H5, monitor='val_accuracy',
            save_best_only=True, verbose=1
        )
    ]
    
    print("\n🚀 Дообучение...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,  # Маленький батч для маленького датасета
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# === ОЦЕНКА ===
def evaluate_model(model, X_val, y_val):
    """Детальная оценка на валидации."""
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_pred = model.predict(X_val, verbose=0)
    y_pred_class = np.argmax(y_pred, axis=1)
    
    print("\n📊 Матрица ошибок:")
    print(confusion_matrix(y_val, y_pred_class))
    
    print("\n📊 Отчёт по классам:")
    print(classification_report(
        y_val, y_pred_class,
        target_names=['Без категории', 'Третья', 'Вторая', 'Первая'],
        zero_division=0
    ))
    
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n  Loss: {loss:.4f}, Accuracy: {acc:.2%}")
    
    return acc

# === СРАВНЕНИЕ ДО И ПОСЛЕ ===
def compare_models(base_model, finetuned_model, X_val, y_val):
    """Сравнивает точность базовой и дообученной модели."""
    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 50)
    
    if base_model is not None:
        base_loss, base_acc = base_model.evaluate(X_val, y_val, verbose=0)
        print(f"Базовая модель:     Accuracy = {base_acc:.2%}")
    else:
        print("Базовая модель:     не загружена")
    
    ft_loss, ft_acc = finetuned_model.evaluate(X_val, y_val, verbose=0)
    print(f"Дообученная модель: Accuracy = {ft_acc:.2%}")
    
    if base_model is not None:
        improvement = (ft_acc - base_acc) * 100
        print(f"Улучшение:          {improvement:+.1f}%")

# === MAIN ===
def main():
    print("=" * 60)
    print("ДООБУЧЕНИЕ МОДЕЛИ КИИ НА НОВЫХ ДАННЫХ")
    print("=" * 60)
    
    # 1. Загрузка данных
    X, y = load_data(DATA_PATH)
    
    # 2. Подготовка признаков
    X_scaled, scaler, encoders = preprocess(X, y, fit=True)
    print(f"  Размерность после обработки: {X_scaled.shape}")
    
    # 3. Разделение на обучение и валидацию
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Обучение: {X_train.shape[0]}, Валидация: {X_val.shape[0]}")
    
    # 4. Загрузка базовой модели
    base_model = load_base_model(BASE_MODEL_PATH)
    
    if base_model is None:
        print("\n  Создаю новую модель...")
        model = create_model(X_train.shape[1])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
    else:
        print("\n  Использую базовую модель для дообучения...")
        model = base_model
        model.summary()
    
    # 5. Дообучение
    model, history = finetune(model, X_train, y_train, X_val, y_val)
    
    # 6. Оценка
    evaluate_model(model, X_val, y_val)
    
    # 7. Сравнение
    compare_models(base_model, model, X_val, y_val)
    
    # 8. Сохранение артефактов
    print("\n💾 Сохранение...")
    
    # Сохраняем в H5 (совместимый формат)
    model.save(OUTPUT_MODEL_H5, save_format='h5')
    print(f"  ✓ Модель (H5): {OUTPUT_MODEL_H5}")
    
    # Пробуем сохранить в .keras
    try:
        model.save(OUTPUT_MODEL_KERAS)
        print(f"  ✓ Модель (Keras): {OUTPUT_MODEL_KERAS}")
    except Exception as e:
        print(f"  ⚠ Keras сохранение: {e}")
    
    joblib.dump(scaler, OUTPUT_SCALER)
    print(f"  ✓ Скейлер: {OUTPUT_SCALER}")
    
    joblib.dump(encoders, OUTPUT_ENCODERS)
    print(f"  ✓ Кодировщики: {OUTPUT_ENCODERS}")
    
    print("\n" + "=" * 60)
    print("ГОТОВО! Дообученная модель сохранена.")
    print(f"Используйте: {OUTPUT_MODEL_H5}")
    print("=" * 60)

if __name__ == "__main__":
    main()
    