import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = PROJECT_ROOT / "inference-api"
APP_PATH = INFERENCE_DIR / "app.py"


@contextmanager
def temporary_cwd(target: Path):
    previous = Path.cwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(previous)


def load_app_module():
    spec = importlib.util.spec_from_file_location("kii_inference_app", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить модуль API из {APP_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_payload() -> dict:
    return {
        "object_name": "Тестовый объект КИИ",
        "sector": "energy",
        "ownership_level": "federal",
        "service_scale": "regional",
        "process_criticality": 6,
        "supported_users": 500000,
        "territories_count": 12,
        "annual_financial_loss_million": 850.0,
        "recovery_time_hours": 12,
        "critical_processes": 5,
        "interactions_count": 18,
        "personal_data_subjects": 120000,
        "employees_affected": 2500,
        "continuous_operation": True,
        "scada_used": True,
        "government_services": False,
        "life_safety_impact": False,
        "ecological_impact": False,
        "defense_impact": False,
        "public_order_impact": False,
        "transport_disruption": False,
        "communications_disruption": False,
        "classified_info": False,
    }


def main():
    with temporary_cwd(INFERENCE_DIR):
        module = load_app_module()

        with TestClient(module.app) as client:
            health_response = client.get("/health")
            health_data = health_response.json()
            print("=== HEALTH ===")
            print(json.dumps(health_data, ensure_ascii=False, indent=2))

            predict_response = client.post("/predict", json=build_payload())
            predict_data = predict_response.json()
            print("\n=== PREDICT ===")
            print(json.dumps(predict_data, ensure_ascii=False, indent=2))

    if health_response.status_code != 200:
        print(f"\nПРОВЕРКА НЕ ПРОЙДЕНА: /health вернул {health_response.status_code}.")
        raise SystemExit(1)

    if predict_response.status_code != 200:
        print(f"\nПРОВЕРКА НЕ ПРОЙДЕНА: /predict вернул {predict_response.status_code}.")
        raise SystemExit(1)

    if not health_data.get("model_loaded") or not health_data.get("model_usable"):
        print("\nПРОВЕРКА НЕ ПРОЙДЕНА: API не считает модель пригодной для использования.")
        raise SystemExit(1)

    if predict_data.get("model_role") != "supporting_classifier":
        print("\nПРОВЕРКА НЕ ПРОЙДЕНА: API работает без ML-модели.")
        raise SystemExit(1)

    print("\nПРОВЕРКА ПРОЙДЕНА: API загрузил и использует обученную модель.")


if __name__ == "__main__":
    main()
