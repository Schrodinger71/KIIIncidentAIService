from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kii_methodology import (  # noqa: E402
    CRITERIA,
    OBJECT_TYPES,
    OWNERSHIP_LEVELS,
    SECTOR_LABELS,
    SERVICE_SCALES,
    criterion_applicable_field,
    criterion_level_field,
    derive_category_level,
    methodology_score,
)

NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "6000"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", Path(__file__).with_name("synthetic_kii_data.csv")))

rng = np.random.default_rng(RANDOM_SEED)

SECTOR_WEIGHTS = np.array([0.11, 0.08, 0.09, 0.11, 0.09, 0.04, 0.09, 0.08, 0.04, 0.08, 0.04, 0.05, 0.05, 0.05])

CRITERION_PROFILES = {
    "life_health": {"base_app": 0.03, "base_severity": 0.20, "sector_boosts": {"health": 0.75, "transport": 0.45, "energy": 0.30, "fuel_energy": 0.35, "nuclear": 0.75, "chemistry": 0.45}},
    "life_support": {"base_app": 0.10, "base_severity": 0.18, "sector_boosts": {"energy": 0.45, "communications": 0.35, "real_estate": 0.28, "transport": 0.30, "fuel_energy": 0.35}},
    "transport": {"base_app": 0.01, "base_severity": 0.20, "sector_boosts": {"transport": 0.90}},
    "communications": {"base_app": 0.02, "base_severity": 0.22, "sector_boosts": {"communications": 0.92}},
    "government_service": {"base_app": 0.08, "base_severity": 0.18, "sector_boosts": {"real_estate": 0.60, "communications": 0.18, "science": 0.10}},
    "government_function": {"base_app": 0.08, "base_severity": 0.22, "sector_boosts": {"real_estate": 0.55, "communications": 0.16, "science": 0.10}},
    "international_treaty": {"base_app": 0.02, "base_severity": 0.12, "sector_boosts": {"real_estate": 0.12, "defense": 0.28, "space": 0.18}},
    "entity_income_loss": {"base_app": 0.05, "base_severity": 0.18, "sector_boosts": {"finance": 0.32, "energy": 0.32, "fuel_energy": 0.38, "nuclear": 0.28, "defense": 0.26, "space": 0.24, "mining": 0.35, "metallurgy": 0.35, "chemistry": 0.35}},
    "federal_budget_loss": {"base_app": 0.03, "base_severity": 0.18, "sector_boosts": {"finance": 0.25, "energy": 0.25, "fuel_energy": 0.25, "mining": 0.22, "metallurgy": 0.22, "chemistry": 0.22}},
    "financial_market": {"base_app": 0.01, "base_severity": 0.22, "sector_boosts": {"finance": 0.94}},
    "environment": {"base_app": 0.02, "base_severity": 0.22, "sector_boosts": {"energy": 0.35, "fuel_energy": 0.55, "nuclear": 0.82, "mining": 0.52, "metallurgy": 0.42, "chemistry": 0.68}},
    "control_center": {"base_app": 0.04, "base_severity": 0.20, "sector_boosts": {"real_estate": 0.16, "transport": 0.35, "communications": 0.28, "energy": 0.34, "defense": 0.45, "space": 0.40}},
    "defense_order": {"base_app": 0.01, "base_severity": 0.20, "sector_boosts": {"defense": 0.92, "space": 0.35}},
    "defense_security_system": {"base_app": 0.02, "base_severity": 0.22, "sector_boosts": {"defense": 0.94, "real_estate": 0.10, "communications": 0.28, "space": 0.32}},
}


def generate_object_name(index: int, sectors: list[str]) -> str:
    prefixes = ["ИС", "АСУ", "Контур", "Платформа", "Сервис", "Подсистема"]
    sector_label = SECTOR_LABELS[sectors[0]]
    return f"{rng.choice(prefixes)} {sector_label} #{index + 1}"


def choose_sectors() -> list[str]:
    count = int(rng.choice([1, 2, 3], p=[0.72, 0.22, 0.06]))
    choices = rng.choice(list(SECTOR_LABELS.keys()), size=count, replace=False, p=SECTOR_WEIGHTS / SECTOR_WEIGHTS.sum())
    return list(choices)


def severity_seed(sectors: list[str], process_criticality: int, service_scale: str, ownership_level: str) -> float:
    scale_factor = {"local": 0.05, "regional": 0.12, "federal": 0.22, "intersectoral": 0.28}[service_scale]
    ownership_factor = {"private": 0.02, "municipal": 0.06, "regional": 0.10, "federal": 0.16}[ownership_level]
    sector_factor = 0.03 * len(sectors)
    return min(0.15 + process_criticality / 20.0 + scale_factor + ownership_factor + sector_factor, 0.95)


def sample_level(applicable: bool, severity: float) -> int:
    if not applicable:
        return 0

    severity = min(max(severity, 0.0), 1.0)
    p0 = max(0.05, 0.52 - severity * 0.34)
    p1 = max(0.12, 0.24 - severity * 0.02)
    p2 = max(0.08, 0.16 + severity * 0.10)
    p3 = max(0.02, 1.0 - (p0 + p1 + p2))
    probs = np.array([p0, p1, p2, p3], dtype=np.float64)
    probs /= probs.sum()
    return int(rng.choice([0, 1, 2, 3], p=probs))


def assess_criterion(criterion_id: str, sectors: list[str], base_severity: float, object_type: str, scada_used: bool) -> tuple[bool, int]:
    profile = CRITERION_PROFILES[criterion_id]
    sector_boost = max((profile["sector_boosts"].get(sector, 0.0) for sector in sectors), default=0.0)
    app_chance = profile["base_app"] + sector_boost

    if object_type == "process_control_system" and criterion_id in {"life_health", "life_support", "environment", "control_center"}:
        app_chance += 0.08
    if object_type == "telecom_network" and criterion_id in {"communications", "life_support"}:
        app_chance += 0.12
    if scada_used and criterion_id in {"life_health", "environment", "control_center"}:
        app_chance += 0.08
    if criterion_id in {"government_service", "government_function", "international_treaty"} and "real_estate" in sectors:
        app_chance += 0.10

    applicable = bool(rng.random() < min(app_chance, 0.98))
    level = sample_level(applicable, base_severity + profile["base_severity"] + sector_boost)
    return applicable, level


def generate_record(index: int) -> dict:
    sectors = choose_sectors()
    ownership_level = str(rng.choice(OWNERSHIP_LEVELS, p=[0.20, 0.32, 0.24, 0.24]))
    service_scale = str(rng.choice(SERVICE_SCALES, p=[0.34, 0.30, 0.24, 0.12]))
    object_type = str(rng.choice(OBJECT_TYPES, p=[0.58, 0.20, 0.22]))
    process_criticality = int(rng.choice(np.arange(1, 11), p=np.array([0.03, 0.05, 0.08, 0.10, 0.15, 0.16, 0.15, 0.12, 0.10, 0.06])))
    critical_process_count = int(rng.integers(1, 4 + process_criticality))
    interaction_count = int(rng.integers(2, 18 + process_criticality * 5))
    continuous_operation = bool(rng.random() < (0.25 + process_criticality / 15.0))
    scada_used = object_type == "process_control_system" or bool(rng.random() < 0.14)
    classified_info = "defense" in sectors or bool(rng.random() < 0.18)
    base = severity_seed(sectors, process_criticality, service_scale, ownership_level)

    row = {
        "object_name": generate_object_name(index, sectors),
        "sectors": "|".join(sectors),
        "ownership_level": ownership_level,
        "service_scale": service_scale,
        "object_type": object_type,
        "process_criticality": process_criticality,
        "critical_process_count": critical_process_count,
        "interaction_count": interaction_count,
        "continuous_operation": continuous_operation,
        "scada_used": scada_used,
        "classified_info": classified_info,
    }

    for criterion in CRITERIA:
        applicable, level = assess_criterion(criterion["id"], sectors, base, object_type, scada_used)
        row[criterion_applicable_field(criterion["id"])] = applicable
        row[criterion_level_field(criterion["id"])] = level

    row["category_level"] = derive_category_level(row)
    row["significance_score"] = methodology_score(row)
    return row


def main() -> None:
    data = [generate_record(index) for index in range(NUM_SAMPLES)]
    df = pd.DataFrame(data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Сгенерирован датасет: {OUTPUT_PATH}")
    print(f"Количество записей: {len(df)}")
    print("Распределение по категориям:")
    print(df["category_level"].value_counts().sort_index())


if __name__ == "__main__":
    main()
