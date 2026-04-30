import os
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from tensorflow import keras
    TENSORFLOW_IMPORT_ERROR = None
except Exception as exc:
    keras = None
    TENSORFLOW_IMPORT_ERROR = str(exc)

try:
    import h5py
except Exception:
    h5py = None


app = FastAPI(
    title="KII Significance Inference API",
    version="2.1.0",
    description="AI service for determining the significance category of a KII object.",
)

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

CATEGORY_MAP = {
    0: "Без категории",
    1: "Третья категория",
    2: "Вторая категория",
    3: "Первая категория",
}

RECOMMENDATIONS = {
    0: [
        "Оставить объект в реестре наблюдения и проводить периодическую переоценку.",
        "Уточнить критичность бизнес-процессов и полноту исходных характеристик.",
        "Подготовить базовый комплект организационных мер без усиленного режима.",
    ],
    1: [
        "Оформить паспорт объекта и закрепить ответственных за его защиту.",
        "Реализовать базовые меры ИБ и регламент резервного восстановления.",
        "Проводить регулярную переоценку значимости при изменении характеристик объекта.",
    ],
    2: [
        "Усилить меры защиты, сегментацию и контроль доступа к объекту КИИ.",
        "Подготовить сценарии отказа и план обеспечения непрерывности процессов.",
        "Зафиксировать интеграции и критичные зависимости в технологическом контуре.",
    ],
    3: [
        "Применить максимальный приоритет защиты и непрерывности функционирования объекта.",
        "Подготовить расширенный план реагирования и аварийного восстановления.",
        "Согласовать регламент контроля с учетом наиболее значимых последствий для КИИ.",
    ],
}

MODEL = None
MODEL_SOURCE = "heuristic_only"
MODEL_LOAD_ERRORS: list[str] = []
ACTIVE_MODEL_CHOICE = os.getenv("DEFAULT_MODEL_CHOICE", "keras")
MODEL_USABLE = False
MODEL_DIAGNOSTIC = "not_loaded"


class KIIObjectPayload(BaseModel):
    object_name: str = Field(..., min_length=3, max_length=200)
    sector: str
    ownership_level: str
    service_scale: str
    process_criticality: int = Field(..., ge=1, le=10)
    supported_users: int = Field(0, ge=0)
    territories_count: int = Field(1, ge=1)
    annual_financial_loss_million: float = Field(0, ge=0)
    recovery_time_hours: int = Field(0, ge=0)
    critical_processes: int = Field(0, ge=0)
    interactions_count: int = Field(0, ge=0)
    personal_data_subjects: int = Field(0, ge=0)
    employees_affected: int = Field(0, ge=0)
    continuous_operation: bool = False
    scada_used: bool = False
    government_services: bool = False
    life_safety_impact: bool = False
    ecological_impact: bool = False
    defense_impact: bool = False
    public_order_impact: bool = False
    transport_disruption: bool = False
    communications_disruption: bool = False
    classified_info: bool = False


class ModelSelectionPayload(BaseModel):
    model_choice: str


def named_probabilities(values: np.ndarray) -> dict[str, float]:
    return {
        CATEGORY_MAP[index]: float(round(values[index], 4))
        for index in range(min(len(values), len(CATEGORY_MAP)))
    }


def configured_model_paths() -> list[Path]:
    explicit_model = os.getenv("MODEL_PATH")
    fallback_model = os.getenv("FALLBACK_MODEL_PATH")
    allow_fallback = os.getenv("ALLOW_MODEL_FALLBACK", "false").lower() == "true"

    paths: list[Path] = []
    predefined_models = {
        "keras": Path("/app/models/model.keras"),
        "h5": Path("/app/models/model.h5"),
    }

    if ACTIVE_MODEL_CHOICE in predefined_models:
        paths.append(predefined_models[ACTIVE_MODEL_CHOICE])
    elif explicit_model:
        paths.append(Path(explicit_model))
    else:
        paths.append(Path("/app/models/model.keras"))

    if allow_fallback and fallback_model:
        fallback_path = Path(fallback_model)
        if fallback_path not in paths:
            paths.append(fallback_path)

    return paths


def load_model() -> None:
    global MODEL
    global MODEL_SOURCE
    global MODEL_LOAD_ERRORS
    global MODEL_USABLE
    global MODEL_DIAGNOSTIC

    MODEL = None
    MODEL_LOAD_ERRORS = []
    MODEL_USABLE = False
    MODEL_DIAGNOSTIC = "not_loaded"

    if keras is None:
        MODEL_SOURCE = "tensorflow_unavailable"
        if TENSORFLOW_IMPORT_ERROR:
            MODEL_LOAD_ERRORS.append(f"tensorflow import failed: {TENSORFLOW_IMPORT_ERROR}")
        MODEL_DIAGNOSTIC = "tensorflow_unavailable"
        return

    for path in configured_model_paths():
        if not path.exists():
            continue

        try:
            MODEL = keras.models.load_model(path, compile=False)
            MODEL_SOURCE = str(path)
            MODEL_USABLE, MODEL_DIAGNOSTIC = evaluate_model_health(MODEL)
            return
        except Exception as exc:
            MODEL_LOAD_ERRORS.append(f"{path}: {exc}")
            if path.suffix == ".keras":
                try:
                    MODEL = load_keras_archive_manually(path)
                    MODEL_SOURCE = f"{path} (manual archive load)"
                    MODEL_USABLE, MODEL_DIAGNOSTIC = evaluate_model_health(MODEL)
                    return
                except Exception as manual_exc:
                    MODEL_LOAD_ERRORS.append(f"{path} manual load: {manual_exc}")

    if MODEL_LOAD_ERRORS:
        MODEL_SOURCE = "model_load_failed"
        MODEL_DIAGNOSTIC = "load_failed"
        return

    MODEL_SOURCE = "model_file_not_found"
    MODEL_DIAGNOSTIC = "file_not_found"


def load_keras_archive_manually(path: Path):
    if h5py is None:
        raise ValueError("h5py is unavailable, so manual .keras loading cannot be used.")

    if not zipfile.is_zipfile(path):
        raise ValueError("File is not a valid .keras archive.")

    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        if "config.json" not in names or "model.weights.h5" not in names:
            raise ValueError("The .keras archive does not contain config.json or model.weights.h5.")

        config = json.loads(archive.read("config.json").decode("utf-8"))

        # Rebuild the model from serialized config and then map archived weights by layer name.
        if hasattr(keras, "saving") and hasattr(keras.saving, "deserialize_keras_object"):
            model = keras.saving.deserialize_keras_object(config)
        else:
            model = keras.models.model_from_json(json.dumps(config))

        build_input_shape = config.get("build_config", {}).get("input_shape")
        if build_input_shape:
            shape = tuple(1 if item is None else item for item in build_input_shape)
            model(np.zeros(shape, dtype=np.float32))

        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
            tmp.write(archive.read("model.weights.h5"))
            tmp_path = tmp.name

    try:
        assigned_layers = 0
        with h5py.File(tmp_path, "r") as weights_file:
            for layer in model.layers:
                if not layer.weights:
                    continue

                layer_group = None
                for candidate in (
                    f"_layer_checkpoint_dependencies\\{layer.name}",
                    f"_layer_checkpoint_dependencies/{layer.name}",
                    layer.name,
                ):
                    if candidate in weights_file:
                        layer_group = weights_file[candidate]
                        break

                if layer_group is None or "vars" not in layer_group:
                    raise ValueError(f"Missing archived weights for layer '{layer.name}'.")

                vars_group = layer_group["vars"]
                weight_keys = sorted(
                    [key for key in vars_group.keys() if str(key).isdigit()],
                    key=int,
                )
                layer_weights = [np.array(vars_group[key]) for key in weight_keys]

                if not layer_weights:
                    raise ValueError(f"No tensors found in archived weights for layer '{layer.name}'.")

                layer.set_weights(layer_weights)
                assigned_layers += 1

        if assigned_layers == 0:
            raise ValueError("No layer weights were assigned from the .keras archive.")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return model


def select_model(model_choice: str) -> dict:
    global ACTIVE_MODEL_CHOICE

    normalized = model_choice.strip().lower()
    if normalized not in {"keras", "h5"}:
        raise HTTPException(status_code=400, detail="Unsupported model choice. Use 'keras' or 'h5'.")

    ACTIVE_MODEL_CHOICE = normalized
    load_model()

    if MODEL is None:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Failed to load model '{normalized}'.",
                "model_source": MODEL_SOURCE,
                "model_load_errors": MODEL_LOAD_ERRORS,
            },
        )

    return {
        "status": "ok",
        "active_model_choice": ACTIVE_MODEL_CHOICE,
        "model_loaded": True,
        "model_source": MODEL_SOURCE,
        "model_usable": MODEL_USABLE,
        "model_diagnostic": MODEL_DIAGNOSTIC,
    }


def evaluate_model_health(model) -> tuple[bool, str]:
    test_vectors = np.array(
        [
            [0.0] * 33,
            [1.0] * 33,
            [0.1] * 33,
            [0.0, 1.0] * 16 + [0.0],
            [1.0, 0.0] * 16 + [1.0],
        ],
        dtype=np.float32,
    )

    try:
        predictions = np.array(model.predict(test_vectors, verbose=0), dtype=np.float32)
    except Exception as exc:
        return False, f"prediction_failed: {exc}"

    if predictions.ndim != 2 or predictions.shape[1] != 4:
        return False, "unexpected_output_shape"

    top_classes = np.argmax(predictions, axis=1)
    top_confidences = np.max(predictions, axis=1)

    if len(set(top_classes.tolist())) == 1 and float(np.min(top_confidences)) >= 0.99:
        return False, "degenerate_constant_prediction"

    return True, "ok"


def feature_vector(payload: KIIObjectPayload) -> np.ndarray:
    base_features = [
        payload.process_criticality / 10.0,
        min(payload.supported_users / 10_000_000.0, 1.0),
        min(payload.territories_count / 89.0, 1.0),
        min(payload.annual_financial_loss_million / 10_000.0, 1.0),
        min(payload.recovery_time_hours / 168.0, 1.0),
        min(payload.critical_processes / 50.0, 1.0),
        min(payload.interactions_count / 200.0, 1.0),
        min(payload.personal_data_subjects / 10_000_000.0, 1.0),
        min(payload.employees_affected / 1_000_000.0, 1.0),
        float(payload.continuous_operation),
        float(payload.scada_used),
        float(payload.government_services),
        float(payload.life_safety_impact),
        float(payload.ecological_impact),
        float(payload.defense_impact),
        float(payload.public_order_impact),
        float(payload.transport_disruption),
        float(payload.communications_disruption),
        float(payload.classified_info),
    ]

    sector_flags = [1.0 if payload.sector == item else 0.0 for item in SECTORS]
    ownership_flags = [1.0 if payload.ownership_level == item else 0.0 for item in OWNERSHIP_LEVELS]
    service_scale_flags = [
        1.0 if payload.service_scale == "regional" else 0.0,
        1.0 if payload.service_scale == "federal" else 0.0,
        1.0 if payload.service_scale == "intersectoral" else 0.0,
    ]

    values = base_features + sector_flags + ownership_flags + service_scale_flags
    return np.array([values], dtype=np.float32)


def heuristic_score(payload: KIIObjectPayload) -> int:
    score = 0.0
    score += payload.process_criticality * 5
    score += min(payload.supported_users / 500_000.0, 15)
    score += min(payload.territories_count * 2.0, 12)
    score += min(payload.annual_financial_loss_million / 500.0, 12)
    score += min(payload.recovery_time_hours / 8.0, 10)
    score += min(payload.critical_processes * 1.5, 10)
    score += min(payload.interactions_count / 20.0, 8)
    score += min(payload.personal_data_subjects / 200_000.0, 8)
    score += min(payload.employees_affected / 10_000.0, 6)

    score += 7 if payload.life_safety_impact else 0
    score += 7 if payload.defense_impact else 0
    score += 6 if payload.public_order_impact else 0
    score += 5 if payload.ecological_impact else 0
    score += 5 if payload.transport_disruption else 0
    score += 5 if payload.communications_disruption else 0
    score += 4 if payload.scada_used else 0
    score += 4 if payload.government_services else 0
    score += 3 if payload.continuous_operation else 0
    score += 3 if payload.classified_info else 0

    if payload.ownership_level == "federal":
        score += 4
    elif payload.ownership_level == "regional":
        score += 2

    if payload.service_scale == "intersectoral":
        score += 5
    elif payload.service_scale == "federal":
        score += 3
    elif payload.service_scale == "regional":
        score += 1

    return int(min(round(score), 100))


def score_to_class(score: int) -> int:
    if score < 25:
        return 0
    if score < 50:
        return 1
    if score < 75:
        return 2
    return 3


def heuristic_distribution(score_class: int) -> np.ndarray:
    distributions = {
        0: np.array([0.74, 0.16, 0.07, 0.03], dtype=np.float32),
        1: np.array([0.11, 0.63, 0.18, 0.08], dtype=np.float32),
        2: np.array([0.05, 0.17, 0.61, 0.17], dtype=np.float32),
        3: np.array([0.03, 0.08, 0.20, 0.69], dtype=np.float32),
    }
    return distributions[score_class]


def model_distribution(features: np.ndarray) -> np.ndarray:
    if MODEL is None or not MODEL_USABLE:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    try:
        prediction = MODEL.predict(features, verbose=0)[0]
        prediction = np.array(prediction, dtype=np.float32)
        if prediction.shape[0] != 4:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        total = float(prediction.sum())
        if total <= 0:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        return prediction / total
    except Exception:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)


def key_factors(payload: KIIObjectPayload) -> list[str]:
    factors: list[str] = []

    if payload.life_safety_impact:
        factors.append("влияние на жизнь и здоровье людей")
    if payload.defense_impact:
        factors.append("влияние на оборону и безопасность государства")
    if payload.public_order_impact:
        factors.append("влияние на общественный порядок")
    if payload.scada_used:
        factors.append("наличие АСУ ТП или технологического контура")
    if payload.service_scale in {"federal", "intersectoral"}:
        factors.append("масштаб оказания услуг выше локального уровня")
    if payload.supported_users >= 1_000_000:
        factors.append("большое число пользователей или потребителей")
    if payload.annual_financial_loss_million >= 1_000:
        factors.append("существенный потенциальный финансовый ущерб")
    if payload.recovery_time_hours >= 24:
        factors.append("длительное восстановление после отказа")

    return factors[:5]


def build_summary(payload: KIIObjectPayload, category_name: str, score: int) -> str:
    summary = [
        f"Объект '{payload.object_name}' отнесен к категории '{category_name}'.",
        f"Интегральная оценка значимости: {score}/100.",
    ]

    factors = key_factors(payload)
    if factors:
        summary.append("Ключевые факторы: " + ", ".join(factors) + ".")

    return " ".join(summary)


@app.on_event("startup")
def startup_event() -> None:
    load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_usable": MODEL_USABLE,
        "model_source": MODEL_SOURCE,
        "model_diagnostic": MODEL_DIAGNOSTIC,
        "model_load_errors": MODEL_LOAD_ERRORS,
        "active_model_choice": ACTIVE_MODEL_CHOICE,
        "available_models": ["keras", "h5"],
    }


@app.post("/model/select")
def set_model(payload: ModelSelectionPayload):
    return select_model(payload.model_choice)


@app.post("/predict")
def predict(payload: KIIObjectPayload):
    features = feature_vector(payload)
    score = heuristic_score(payload)
    score_class = score_to_class(score)
    heuristics = heuristic_distribution(score_class)
    model_probs = model_distribution(features)

    combined = (heuristics * 0.55) + (model_probs * 0.45)
    final_class = max(int(np.argmax(combined)), score_class)
    category_name = CATEGORY_MAP[final_class]
    confidence = float(round(combined[final_class], 4))

    return {
        "predicted_category": category_name,
        "significance_score": score,
        "category_level": final_class,
        "heuristic_class": score_class,
        "confidence": confidence,
        "model_loaded": MODEL is not None,
        "model_usable": MODEL_USABLE,
        "model_source": MODEL_SOURCE,
        "model_diagnostic": MODEL_DIAGNOSTIC,
        "model_role": "supporting_classifier" if MODEL_USABLE else "heuristic_only",
        "heuristic_probabilities": named_probabilities(heuristics),
        "model_probabilities": named_probabilities(model_probs),
        "combined_probabilities": named_probabilities(combined),
        "key_factors": key_factors(payload),
        "summary": build_summary(payload, category_name, score),
        "recommendations": RECOMMENDATIONS[final_class],
    }
