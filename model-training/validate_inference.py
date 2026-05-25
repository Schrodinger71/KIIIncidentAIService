from __future__ import annotations

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
        "sectors": ["communications", "real_estate"],
        "ownership_level": "regional",
        "service_scale": "regional",
        "object_type": "information_system",
        "process_criticality": 7,
        "critical_process_count": 4,
        "interaction_count": 18,
        "continuous_operation": True,
        "scada_used": False,
        "classified_info": True,
        "life_health_applicable": False,
        "life_health_level": 0,
        "life_support_applicable": True,
        "life_support_level": 1,
        "transport_applicable": False,
        "transport_level": 0,
        "communications_applicable": True,
        "communications_level": 2,
        "government_service_applicable": True,
        "government_service_level": 2,
        "government_function_applicable": True,
        "government_function_level": 1,
        "international_treaty_applicable": False,
        "international_treaty_level": 0,
        "entity_income_loss_applicable": False,
        "entity_income_loss_level": 0,
        "federal_budget_loss_applicable": False,
        "federal_budget_loss_level": 0,
        "financial_market_applicable": False,
        "financial_market_level": 0,
        "environment_applicable": False,
        "environment_level": 0,
        "control_center_applicable": True,
        "control_center_level": 1,
        "defense_order_applicable": False,
        "defense_order_level": 0,
        "defense_security_system_applicable": False,
        "defense_security_system_level": 0,
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

    if predict_data.get("category_level") != 2:
        print("\nПРОВЕРКА НЕ ПРОЙДЕНА: методическое правило должно вернуть вторую категорию.")
        raise SystemExit(1)

    if predict_data.get("model_role") != "supporting_classifier":
        print("\nПРОВЕРКА НЕ ПРОЙДЕНА: API работает без ML-модели.")
        raise SystemExit(1)

    print("\nПРОВЕРКА ПРОЙДЕНА: API загрузил и использует обученную модель.")


if __name__ == "__main__":
    main()
