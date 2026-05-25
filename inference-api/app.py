from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kii_methodology import (  # noqa: E402
    CATEGORY_MAP,
    CRITERIA,
    FEATURE_INPUT_DIM,
    OBJECT_TYPES,
    OWNERSHIP_LEVELS,
    RECOMMENDATIONS,
    SECTORS,
    SERVICE_SCALES,
    build_summary,
    criterion_applicable_field,
    criterion_level_field,
    criterion_results,
    default_record,
    derive_category_level,
    feature_vector,
    heuristic_distribution,
    key_factors,
    methodology_score,
    normalize_record,
)

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
    version="3.0.0",
    description=(
        "AI service for determining the significance category of a KII object "
        "based on methodology indicators 8.3.1-8.3.14."
    ),
)

MODEL = None
MODEL_SOURCE = "methodology_only"
MODEL_LOAD_ERRORS: list[str] = []
ACTIVE_MODEL_CHOICE = os.getenv("DEFAULT_MODEL_CHOICE", "keras")
MODEL_USABLE = False
MODEL_DIAGNOSTIC = "not_loaded"


class KIIObjectPayload(BaseModel):
    object_name: str = Field(..., min_length=3, max_length=200)
    sectors: list[str] = Field(default_factory=list)
    sector: str | None = None
    ownership_level: str
    service_scale: str
    object_type: str = "information_system"
    process_criticality: int = Field(5, ge=1, le=10)
    critical_process_count: int = Field(0, ge=0)
    interaction_count: int = Field(0, ge=0)
    continuous_operation: bool = False
    scada_used: bool = False
    classified_info: bool = False

    life_health_applicable: bool = False
    life_health_level: int = Field(0, ge=0, le=3)
    life_support_applicable: bool = False
    life_support_level: int = Field(0, ge=0, le=3)
    transport_applicable: bool = False
    transport_level: int = Field(0, ge=0, le=3)
    communications_applicable: bool = False
    communications_level: int = Field(0, ge=0, le=3)
    government_service_applicable: bool = False
    government_service_level: int = Field(0, ge=0, le=3)
    government_function_applicable: bool = False
    government_function_level: int = Field(0, ge=0, le=3)
    international_treaty_applicable: bool = False
    international_treaty_level: int = Field(0, ge=0, le=3)
    entity_income_loss_applicable: bool = False
    entity_income_loss_level: int = Field(0, ge=0, le=3)
    federal_budget_loss_applicable: bool = False
    federal_budget_loss_level: int = Field(0, ge=0, le=3)
    financial_market_applicable: bool = False
    financial_market_level: int = Field(0, ge=0, le=3)
    environment_applicable: bool = False
    environment_level: int = Field(0, ge=0, le=3)
    control_center_applicable: bool = False
    control_center_level: int = Field(0, ge=0, le=3)
    defense_order_applicable: bool = False
    defense_order_level: int = Field(0, ge=0, le=3)
    defense_security_system_applicable: bool = False
    defense_security_system_level: int = Field(0, ge=0, le=3)


class ModelSelectionPayload(BaseModel):
    model_choice: str


def payload_to_record(payload: KIIObjectPayload) -> dict:
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    else:
        data = payload.dict()
    return normalize_record(data)


def named_probabilities(values: np.ndarray) -> dict[str, float]:
    return {
        CATEGORY_MAP[index]: float(round(values[index], 4))
        for index in range(min(len(values), len(CATEGORY_MAP)))
    }


def configured_model_paths() -> list[Path]:
    explicit_model = os.getenv("MODEL_PATH")
    fallback_model = os.getenv("FALLBACK_MODEL_PATH")
    allow_fallback = os.getenv("ALLOW_MODEL_FALLBACK", "false").lower() == "true"
    base_dir = Path(__file__).resolve().parent

    paths: list[Path] = []
    predefined_models = {
        "keras": [
            base_dir / "models" / "model.keras",
            Path("/app/models/model.keras"),
            base_dir / "models" / "model.h5",
            Path("/app/models/model.h5"),
        ],
        "h5": [
            base_dir / "models" / "model.h5",
            Path("/app/models/model.h5"),
            base_dir / "models" / "model.keras",
            Path("/app/models/model.keras"),
        ],
    }

    if explicit_model:
        paths.append(Path(explicit_model))

    if ACTIVE_MODEL_CHOICE in predefined_models:
        paths.extend(predefined_models[ACTIVE_MODEL_CHOICE])
        alternate_choice = "h5" if ACTIVE_MODEL_CHOICE == "keras" else "keras"
        paths.extend(predefined_models[alternate_choice])
    else:
        paths.extend(predefined_models["keras"])
        paths.extend(predefined_models["h5"])

    if allow_fallback and fallback_model:
        fallback_path = Path(fallback_model)
        if fallback_path not in paths:
            paths.append(fallback_path)

    deduplicated_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        deduplicated_paths.append(path)
    return deduplicated_paths


def create_inference_model():
    if keras is None:
        raise ValueError("tensorflow/keras is unavailable.")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(FEATURE_INPUT_DIM,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(4, activation="softmax"),
        ]
    )
    model(np.zeros((1, FEATURE_INPUT_DIM), dtype=np.float32))
    return model


def load_weights_only_from_h5(path: Path):
    model = create_inference_model()
    model.load_weights(path)
    return model


def load_weights_only_from_keras_archive(path: Path):
    if not zipfile.is_zipfile(path):
        raise ValueError("File is not a valid .keras archive.")

    with zipfile.ZipFile(path, "r") as archive:
        if "model.weights.h5" not in set(archive.namelist()):
            raise ValueError("The .keras archive does not contain model.weights.h5.")

        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
            tmp.write(archive.read("model.weights.h5"))
            tmp_path = tmp.name

    try:
        model = create_inference_model()
        model.load_weights(tmp_path)
        return model
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


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
                try:
                    MODEL = load_weights_only_from_keras_archive(path)
                    MODEL_SOURCE = f"{path} (weights-only fallback)"
                    MODEL_USABLE, MODEL_DIAGNOSTIC = evaluate_model_health(MODEL)
                    return
                except Exception as weights_exc:
                    MODEL_LOAD_ERRORS.append(f"{path} weights-only fallback: {weights_exc}")
            elif path.suffix == ".h5":
                try:
                    MODEL = load_weights_only_from_h5(path)
                    MODEL_SOURCE = f"{path} (weights-only fallback)"
                    MODEL_USABLE, MODEL_DIAGNOSTIC = evaluate_model_health(MODEL)
                    return
                except Exception as weights_exc:
                    MODEL_LOAD_ERRORS.append(f"{path} weights-only fallback: {weights_exc}")

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

    source_text = MODEL_SOURCE.lower()
    if "model.h5" in source_text:
        ACTIVE_MODEL_CHOICE = "h5"
    elif "model.keras" in source_text:
        ACTIVE_MODEL_CHOICE = "keras"

    return {
        "status": "ok",
        "requested_model_choice": normalized,
        "active_model_choice": ACTIVE_MODEL_CHOICE,
        "model_loaded": True,
        "model_source": MODEL_SOURCE,
        "model_usable": MODEL_USABLE,
        "model_diagnostic": MODEL_DIAGNOSTIC,
    }


def evaluate_model_health(model) -> tuple[bool, str]:
    baseline = default_record()
    varied_records = []
    for category_level in range(4):
        record = dict(baseline)
        for criterion in CRITERIA:
            record[criterion_applicable_field(criterion["id"])] = category_level > 0
            record[criterion_level_field(criterion["id"])] = category_level
        varied_records.append(feature_vector(record))

    varied_records.append(np.ones(FEATURE_INPUT_DIM, dtype=np.float32))
    varied_records.append(np.zeros(FEATURE_INPUT_DIM, dtype=np.float32))
    test_vectors = np.stack(varied_records).astype(np.float32)

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


def model_distribution(features: np.ndarray) -> np.ndarray:
    if MODEL is None or not MODEL_USABLE:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    try:
        prediction = MODEL.predict(np.array([features], dtype=np.float32), verbose=0)[0]
        prediction = np.array(prediction, dtype=np.float32)
        if prediction.shape[0] != 4:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        total = float(prediction.sum())
        if total <= 0:
            return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        return prediction / total
    except Exception:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)


@app.on_event("startup")
def startup_event() -> None:
    load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "schema": "methodology_8_3_indicators",
        "feature_input_dim": FEATURE_INPUT_DIM,
        "criteria_count": len(CRITERIA),
        "sectors_supported": len(SECTORS),
        "ownership_levels": OWNERSHIP_LEVELS,
        "service_scales": SERVICE_SCALES,
        "object_types": OBJECT_TYPES,
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
    record = payload_to_record(payload)
    features = feature_vector(record)
    rule_class = derive_category_level(record)
    score = methodology_score(record)
    heuristics = heuristic_distribution(record)
    model_probs = model_distribution(features)
    combined = (heuristics * 0.35) + (model_probs * 0.65)
    confidence = float(round(max(float(combined[rule_class]), float(heuristics[rule_class])), 4))
    category_name = CATEGORY_MAP[rule_class]

    return {
        "predicted_category": category_name,
        "category_level": rule_class,
        "significance_score": score,
        "confidence": confidence,
        "methodology_rule": "max_applicable_indicator_level",
        "criterion_assessments": criterion_results(record),
        "key_factors": key_factors(record),
        "summary": build_summary(record),
        "recommendations": RECOMMENDATIONS[rule_class],
        "model_loaded": MODEL is not None,
        "model_usable": MODEL_USABLE,
        "model_source": MODEL_SOURCE,
        "model_diagnostic": MODEL_DIAGNOSTIC,
        "model_role": "supporting_classifier" if MODEL_USABLE else "methodology_only",
        "heuristic_probabilities": named_probabilities(heuristics),
        "model_probabilities": named_probabilities(model_probs),
        "combined_probabilities": named_probabilities(combined),
    }
