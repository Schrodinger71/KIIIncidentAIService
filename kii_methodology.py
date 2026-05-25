from __future__ import annotations

from typing import Any, Mapping

import numpy as np


CATEGORY_MAP = {
    0: "Без категории",
    1: "Третья категория",
    2: "Вторая категория",
    3: "Первая категория",
}
CATEGORY_NAMES = [CATEGORY_MAP[index] for index in range(4)]

SECTORS = [
    "health",
    "science",
    "transport",
    "communications",
    "energy",
    "real_estate",
    "finance",
    "fuel_energy",
    "nuclear",
    "defense",
    "space",
    "mining",
    "metallurgy",
    "chemistry",
]
SECTOR_LABELS = {
    "health": "Здравоохранение",
    "science": "Наука",
    "transport": "Транспорт",
    "communications": "Связь",
    "energy": "Энергетика",
    "real_estate": "Госрегистрация прав",
    "finance": "Финансовый рынок",
    "fuel_energy": "ТЭК",
    "nuclear": "Атомная энергия",
    "defense": "Оборонная промышленность",
    "space": "Ракетно-космическая промышленность",
    "mining": "Горнодобывающая промышленность",
    "metallurgy": "Металлургическая промышленность",
    "chemistry": "Химическая промышленность",
}

OWNERSHIP_LEVELS = ["federal", "regional", "municipal", "private"]
OWNERSHIP_LABELS = {
    "federal": "Федеральный",
    "regional": "Региональный",
    "municipal": "Муниципальный",
    "private": "Частный",
}

SERVICE_SCALES = ["local", "regional", "federal", "intersectoral"]
SERVICE_SCALE_LABELS = {
    "local": "Локальный",
    "regional": "Региональный",
    "federal": "Федеральный",
    "intersectoral": "Межотраслевой",
}

OBJECT_TYPES = ["information_system", "telecom_network", "process_control_system"]
OBJECT_TYPE_LABELS = {
    "information_system": "Информационная система",
    "telecom_network": "ИТКС / сеть",
    "process_control_system": "АСУ / технологический контур",
}

CRITERIA = [
    {
        "id": "life_health",
        "code": "8.3.1",
        "title": "Причинение ущерба жизни и здоровью людей",
        "hint": "Оценка по показателю 8.3.1 после сопоставления с ПП-127.",
    },
    {
        "id": "life_support",
        "code": "8.3.2",
        "title": "Нарушение функционирования объектов жизнеобеспечения населения",
        "hint": "Используйте категорию по показателю 8.3.2 либо 0, если порог не достигнут.",
    },
    {
        "id": "transport",
        "code": "8.3.3",
        "title": "Нарушение функционирования объектов транспортной инфраструктуры",
        "hint": "Категория по показателю 8.3.3 для данного объекта.",
    },
    {
        "id": "communications",
        "code": "8.3.4",
        "title": "Нарушение функционирования сети связи",
        "hint": "Категория по показателю 8.3.4, если объект влияет на услуги связи.",
    },
    {
        "id": "government_service",
        "code": "8.3.5",
        "title": "Отсутствие доступа к государственной услуге",
        "hint": "Категория по показателю 8.3.5 после оценки времени/доли недоступности услуги.",
    },
    {
        "id": "government_function",
        "code": "8.3.6",
        "title": "Нарушение функционирования государственного органа",
        "hint": "Категория по показателю 8.3.6, если объект поддерживает функции госоргана.",
    },
    {
        "id": "international_treaty",
        "code": "8.3.7",
        "title": "Нарушение условий международного договора РФ",
        "hint": "Категория по показателю 8.3.7 либо 0, если показатель неприменим.",
    },
    {
        "id": "entity_income_loss",
        "code": "8.3.8",
        "title": "Ущерб субъекту КИИ в виде снижения дохода",
        "hint": "Категория по показателю 8.3.8 для организаций, к которым он применим.",
    },
    {
        "id": "federal_budget_loss",
        "code": "8.3.9",
        "title": "Ущерб бюджету Российской Федерации",
        "hint": "Категория по показателю 8.3.9, рассчитанная по ущербу бюджету.",
    },
    {
        "id": "financial_market",
        "code": "8.3.10",
        "title": "Нарушение клиентских операций по переводу денежных средств",
        "hint": "Категория по показателю 8.3.10 для системно значимых финансовых процессов.",
    },
    {
        "id": "environment",
        "code": "8.3.11",
        "title": "Вредные воздействия на окружающую среду",
        "hint": "Категория по показателю 8.3.11 для экологических последствий.",
    },
    {
        "id": "control_center",
        "code": "8.3.12",
        "title": "Нарушение функционирования пункта управления",
        "hint": "Категория по показателю 8.3.12 для ситуационных центров и пунктов управления.",
    },
    {
        "id": "defense_order",
        "code": "8.3.13",
        "title": "Снижение показателей государственного оборонного заказа",
        "hint": "Категория по показателю 8.3.13, если объект связан с ГОЗ.",
    },
    {
        "id": "defense_security_system",
        "code": "8.3.14",
        "title": "Нарушение функционирования ИС в области обороны, безопасности и правопорядка",
        "hint": "Категория по показателю 8.3.14 для профильных систем.",
    },
]

CRITERION_IDS = [item["id"] for item in CRITERIA]

RECOMMENDATIONS = {
    0: [
        "Зафиксируйте, какие показатели 8.3.* были признаны неприменимыми или ниже порогов ПП-127.",
        "Сохраните расчет по каждому показателю для последующей переоценки при изменении объекта.",
        "Периодически пересматривайте категорирование при изменении функций, интеграций и сфер деятельности.",
    ],
    1: [
        "Оформите расчет по каждому применимому показателю и сохраните обоснования в акте категорирования.",
        "Уточните сценарии отказа и документируйте зависимости объекта от внешних систем и сетей связи.",
        "Пересматривайте оценку после изменения масштаба последствий или состава критичных процессов.",
    ],
    2: [
        "Подготовьте усиленные меры обеспечения безопасности и план восстановления по критичным сценариям.",
        "Проверьте полноту обоснований по показателям 8.3.*, которые дали вторую категорию и выше.",
        "Зафиксируйте ответственных за пересмотр категорирования при расширении масштаба объекта.",
    ],
    3: [
        "Рассматривайте объект как приоритетный для непрерывности функционирования и реагирования на инциденты.",
        "Проверьте, что обоснования по наивысшему показателю оформлены и готовы для включения в сведения по приказу ФСТЭК.",
        "Регулярно подтверждайте актуальность категорирования при изменении архитектуры, функций и внешних связей объекта.",
    ],
}


def criterion_level_field(criterion_id: str) -> str:
    return f"{criterion_id}_level"


def criterion_applicable_field(criterion_id: str) -> str:
    return f"{criterion_id}_applicable"


CRITERION_LEVEL_FIELDS = [criterion_level_field(item["id"]) for item in CRITERIA]
CRITERION_APPLICABILITY_FIELDS = [criterion_applicable_field(item["id"]) for item in CRITERIA]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "да", "истина", "+"}


def clamp_level(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(3, numeric))


def normalize_choice(value: Any, labels: Mapping[str, str], label: str) -> str:
    if value is None:
        raise ValueError(f"Не указано значение для {label}.")
    normalized = str(value).strip()
    if normalized in labels:
        return normalized

    lower_map = {text.lower(): key for key, text in labels.items()}
    if normalized.lower() in lower_map:
        return lower_map[normalized.lower()]

    raise ValueError(f"Не удалось распознать значение '{value}' для {label}.")


def normalize_sectors(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    elif value is None:
        raw_items = []
    else:
        normalized = str(value).strip()
        if not normalized:
            raw_items = []
        elif normalized.startswith("[") and normalized.endswith("]"):
            stripped = normalized.strip("[]")
            raw_items = [item.strip().strip("'\"") for item in stripped.split(",") if item.strip()]
        else:
            raw_items = [item.strip() for item in normalized.replace(";", "|").replace(",", "|").split("|") if item.strip()]

    normalized_items: list[str] = []
    lower_map = {label.lower(): key for key, label in SECTOR_LABELS.items()}
    for item in raw_items:
        if item in SECTOR_LABELS:
            normalized = item
        else:
            normalized = lower_map.get(str(item).strip().lower())
        if not normalized:
            raise ValueError(f"Не удалось распознать сферу деятельности: {item}")
        if normalized not in normalized_items:
            normalized_items.append(normalized)
    return normalized_items


def sectors_to_text(sectors: Any) -> str:
    normalized = normalize_sectors(sectors)
    if not normalized:
        return "-"
    return ", ".join(SECTOR_LABELS[item] for item in normalized)


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if not normalized.get("sectors") and normalized.get("sector"):
        normalized["sectors"] = [normalized["sector"]]

    normalized["sectors"] = normalize_sectors(normalized.get("sectors", []))
    normalized["ownership_level"] = normalize_choice(normalized.get("ownership_level"), OWNERSHIP_LABELS, "ownership_level")
    normalized["service_scale"] = normalize_choice(normalized.get("service_scale"), SERVICE_SCALE_LABELS, "service_scale")
    normalized["object_type"] = normalize_choice(normalized.get("object_type", "information_system"), OBJECT_TYPE_LABELS, "object_type")

    for key in ("process_criticality", "critical_process_count", "interaction_count"):
        try:
            normalized[key] = int(float(normalized.get(key, 0) or 0))
        except (TypeError, ValueError):
            normalized[key] = 0

    normalized["process_criticality"] = max(1, min(10, normalized["process_criticality"] or 1))
    normalized["critical_process_count"] = max(0, normalized["critical_process_count"])
    normalized["interaction_count"] = max(0, normalized["interaction_count"])

    for key in ("continuous_operation", "scada_used", "classified_info"):
        normalized[key] = as_bool(normalized.get(key))

    for criterion in CRITERIA:
        applicable_field = criterion_applicable_field(criterion["id"])
        level_field = criterion_level_field(criterion["id"])
        normalized[applicable_field] = as_bool(normalized.get(applicable_field))
        normalized[level_field] = clamp_level(normalized.get(level_field))

    return normalized


def criterion_results(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized = normalize_record(record)
    results: list[dict[str, Any]] = []
    for criterion in CRITERIA:
        applicable_field = criterion_applicable_field(criterion["id"])
        level_field = criterion_level_field(criterion["id"])
        applicable = normalized[applicable_field]
        level = normalized[level_field] if applicable else 0
        results.append(
            {
                "id": criterion["id"],
                "code": criterion["code"],
                "title": criterion["title"],
                "hint": criterion["hint"],
                "applicable": applicable,
                "level": level,
                "category_name": CATEGORY_MAP[level],
            }
        )
    return results


def derive_category_level(record: Mapping[str, Any]) -> int:
    results = criterion_results(record)
    return max((item["level"] for item in results if item["applicable"]), default=0)


def positive_criteria_count(record: Mapping[str, Any]) -> int:
    return sum(1 for item in criterion_results(record) if item["applicable"] and item["level"] > 0)


def methodology_score(record: Mapping[str, Any]) -> int:
    results = criterion_results(record)
    positive_levels = [item["level"] for item in results if item["applicable"] and item["level"] > 0]
    if not positive_levels:
        return 0

    max_level = max(positive_levels)
    score = max_level * 25
    score += min(sum(positive_levels) * 4, 28)
    score += min(len(positive_levels) * 3, 12)
    return int(min(score, 100))


def key_factors(record: Mapping[str, Any], limit: int = 5) -> list[str]:
    ranked = sorted(
        (
            item
            for item in criterion_results(record)
            if item["applicable"] and item["level"] > 0
        ),
        key=lambda item: (item["level"], item["code"]),
        reverse=True,
    )
    return [f"{item['code']} {item['title']}" for item in ranked[:limit]]


def build_summary(record: Mapping[str, Any]) -> str:
    normalized = normalize_record(record)
    category_level = derive_category_level(normalized)
    category_name = CATEGORY_MAP[category_level]
    factors = key_factors(normalized, limit=3)

    base = (
        f"Объект '{normalized.get('object_name', 'Без названия')}' отнесен к категории "
        f"'{category_name}' по правилу методики: берется наивысшая категория из применимых показателей 8.3.*."
    )
    if not factors:
        return base + " Для всех отмеченных показателей категория не превышает пороги присвоения."
    return base + " Наибольшее влияние оказали: " + ", ".join(factors) + "."


def heuristic_distribution(record: Mapping[str, Any]) -> np.ndarray:
    category_level = derive_category_level(record)
    positive_count = positive_criteria_count(record)
    peak = min(0.62 + positive_count * 0.04, 0.9)

    if category_level == 0:
        distribution = np.array([0.82, 0.11, 0.05, 0.02], dtype=np.float32)
    elif category_level == 1:
        distribution = np.array([0.12, peak, 0.16, 0.04], dtype=np.float32)
    elif category_level == 2:
        distribution = np.array([0.05, 0.15, peak, 0.10], dtype=np.float32)
    else:
        distribution = np.array([0.03, 0.07, 0.16, peak], dtype=np.float32)

    distribution /= float(distribution.sum())
    return distribution


def feature_vector(record: Mapping[str, Any]) -> np.ndarray:
    normalized = normalize_record(record)
    levels = [
        normalized[criterion_level_field(criterion["id"])] / 3.0
        for criterion in CRITERIA
    ]
    applicability = [
        float(normalized[criterion_applicable_field(criterion["id"])])
        for criterion in CRITERIA
    ]
    context = [
        normalized["process_criticality"] / 10.0,
        min(normalized["critical_process_count"] / 25.0, 1.0),
        min(normalized["interaction_count"] / 100.0, 1.0),
        float(normalized["continuous_operation"]),
        float(normalized["scada_used"]),
        float(normalized["classified_info"]),
    ]
    sector_flags = [1.0 if sector in normalized["sectors"] else 0.0 for sector in SECTORS]
    ownership_flags = [1.0 if normalized["ownership_level"] == item else 0.0 for item in OWNERSHIP_LEVELS]
    scale_flags = [1.0 if normalized["service_scale"] == item else 0.0 for item in SERVICE_SCALES]
    object_type_flags = [1.0 if normalized["object_type"] == item else 0.0 for item in OBJECT_TYPES]

    return np.array(levels + applicability + context + sector_flags + ownership_flags + scale_flags + object_type_flags, dtype=np.float32)


def default_record() -> dict[str, Any]:
    record = {
        "object_name": "Тестовый объект КИИ",
        "sectors": ["communications"],
        "ownership_level": "regional",
        "service_scale": "regional",
        "object_type": "information_system",
        "process_criticality": 5,
        "critical_process_count": 3,
        "interaction_count": 12,
        "continuous_operation": True,
        "scada_used": False,
        "classified_info": False,
    }
    for criterion in CRITERIA:
        record[criterion_applicable_field(criterion["id"])] = False
        record[criterion_level_field(criterion["id"])] = 0
    return record


FEATURE_INPUT_DIM = int(feature_vector(default_record()).shape[0])
