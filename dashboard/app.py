import json
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from kii_methodology import (  # noqa: E402
    CATEGORY_MAP,
    CRITERIA,
    OBJECT_TYPE_LABELS,
    OWNERSHIP_LABELS,
    RECOMMENDATIONS,
    SECTOR_LABELS,
    SERVICE_SCALE_LABELS,
    criterion_applicable_field,
    criterion_level_field,
    sectors_to_text,
)


API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_FILE = Path(os.getenv("DATA_FILE", "data/objects.json"))

CATEGORY_ORDER = [CATEGORY_MAP[index] for index in range(4)]


def ensure_data_file() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        DATA_FILE.write_text("[]", encoding="utf-8")


def load_objects() -> list[dict]:
    ensure_data_file()
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_objects(items: list[dict]) -> None:
    ensure_data_file()
    DATA_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def ping_api() -> tuple[bool, dict]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.ok:
            return True, response.json()
        return False, {"status": f"http {response.status_code}"}
    except requests.RequestException:
        return False, {"status": "unavailable"}


def predict_object(payload: dict) -> dict | None:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def select_model(model_choice: str) -> tuple[bool, dict]:
    try:
        response = requests.post(
            f"{API_URL}/model/select",
            json={"model_choice": model_choice},
            timeout=10,
        )
        if response.ok:
            return True, response.json()
        try:
            return False, response.json()
        except Exception:
            return False, {"detail": f"http {response.status_code}"}
    except requests.RequestException:
        return False, {"detail": "unavailable"}


def objects_frame(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(
            columns=[
                "id",
                "object_name",
                "sectors",
                "ownership_level",
                "service_scale",
                "predicted_category",
                "significance_score",
                "created_at",
            ]
        )

    frame = pd.DataFrame(items).copy()
    if "sectors" not in frame.columns and "sector" in frame.columns:
        frame["sectors"] = frame["sector"]
    if "sectors" not in frame.columns:
        frame["sectors"] = "-"
    if "ownership_level" not in frame.columns:
        frame["ownership_level"] = "-"
    if "service_scale" not in frame.columns:
        frame["service_scale"] = "-"
    if "predicted_category" not in frame.columns:
        frame["predicted_category"] = "-"
    if "significance_score" not in frame.columns:
        frame["significance_score"] = 0
    if "confidence" not in frame.columns:
        frame["confidence"] = 0.0
    if "created_at" not in frame.columns:
        frame["created_at"] = pd.NaT
    frame["sectors"] = frame["sectors"].apply(sectors_to_text)
    frame["ownership_level"] = frame["ownership_level"].map(OWNERSHIP_LABELS).fillna(frame["ownership_level"])
    frame["service_scale"] = frame["service_scale"].map(SERVICE_SCALE_LABELS).fillna(frame["service_scale"])
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    return frame


def registry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    registry = frame.copy()
    for column, default in {
        "id": "-",
        "object_name": "-",
        "sectors": "-",
        "ownership_level": "-",
        "service_scale": "-",
        "process_criticality": 0,
        "significance_score": 0,
        "predicted_category": "-",
        "confidence": 0.0,
        "created_at": pd.NaT,
    }.items():
        if column not in registry.columns:
            registry[column] = default
    registry["created_at"] = registry["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    registry = registry.rename(
        columns={
            "id": "ID",
            "object_name": "Объект КИИ",
            "sectors": "Сферы",
            "ownership_level": "Уровень",
            "service_scale": "Масштаб",
            "process_criticality": "Критичность",
            "significance_score": "Score",
            "predicted_category": "Категория",
            "confidence": "Уверенность",
            "created_at": "Создан",
        }
    )
    columns = [
        "ID",
        "Объект КИИ",
        "Сферы",
        "Уровень",
        "Масштаб",
        "Критичность",
        "Score",
        "Категория",
        "Уверенность",
        "Создан",
    ]
    return registry[[column for column in columns if column in registry.columns]].sort_values(
        "Создан", ascending=False
    )


def probability_frame(probabilities: dict) -> pd.DataFrame:
    if not probabilities:
        return pd.DataFrame(columns=["Категория", "Вероятность"])

    frame = pd.DataFrame(
        [{"Категория": key, "Вероятность": float(value)} for key, value in probabilities.items()]
    )
    return frame.sort_values("Вероятность", ascending=False)


def field_hint(container, text: str) -> None:
    container.caption(text)


def install_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(39, 110, 241, 0.10), transparent 34%),
                radial-gradient(circle at top right, rgba(15, 137, 95, 0.10), transparent 30%),
                linear-gradient(180deg, #f6f9fc 0%, #edf3f8 100%);
            color: #12263a;
            font-family: "Segoe UI", "Trebuchet MS", sans-serif;
        }
        .block-container {
            max-width: 1520px;
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        p, li, label, span, .stMarkdown, .stCaption, .stAlert {
            color: #12263a;
        }
        h1, h2, h3 {
            color: #102033;
            letter-spacing: -0.01em;
        }
        .hero-card {
            display: flex;
            justify-content: space-between;
            gap: 24px;
            padding: 28px 30px;
            margin-bottom: 22px;
            border-radius: 24px;
            background: linear-gradient(135deg, #11304a 0%, #184867 55%, #156a67 100%);
            color: #f9fcff;
            box-shadow: 0 22px 50px rgba(15, 47, 74, 0.15);
        }
        .hero-card h1, .hero-card p, .eyebrow, .hero-note, .status-chip {
            color: #f9fcff !important;
        }
        .hero-card h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.12;
        }
        .hero-side {
            min-width: 300px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 10px;
        }
        .eyebrow {
            margin-bottom: 10px;
            font-size: 0.78rem;
            letter-spacing: 0.16em;
            font-weight: 700;
            color: #a9d8ff !important;
        }
        .hero-note, .status-chip {
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            background: rgba(255, 255, 255, 0.10);
        }
        .status-chip.ok {
            background: rgba(95, 213, 153, 0.18);
            border-color: rgba(95, 213, 153, 0.30);
        }
        .status-chip.fail {
            background: rgba(255, 120, 120, 0.18);
            border-color: rgba(255, 120, 120, 0.30);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #12314a 0%, #1d4663 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        section[data-testid="stSidebar"] * {
            color: #eef6ff !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] div[data-baseweb="base-input"],
        section[data-testid="stSidebar"] input {
            background: rgba(255, 255, 255, 0.12) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.20) !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="tag"] {
            background: rgba(255, 255, 255, 0.18) !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.96);
            border-radius: 18px;
            padding: 16px;
            border: 1px solid rgba(16, 32, 51, 0.08);
            box-shadow: 0 8px 24px rgba(34, 62, 94, 0.08);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            padding-bottom: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background: #ffffff !important;
            color: #17324d !important;
            border-radius: 12px;
            border: 1px solid rgba(16, 32, 51, 0.10);
            padding: 10px 18px;
            font-weight: 700;
            box-shadow: 0 2px 8px rgba(34, 62, 94, 0.04);
        }
        .stTabs [data-baseweb="tab"] p {
            color: #17324d !important;
            font-weight: 700 !important;
        }
        .stTabs [aria-selected="true"] {
            background: #edf5ff !important;
            color: #0e4f8a !important;
            border: 2px solid #0e68b0 !important;
            box-shadow: 0 8px 18px rgba(14, 104, 176, 0.12);
        }
        .stTabs [aria-selected="true"] p {
            color: #0e4f8a !important;
        }
        .panel-card {
            padding: 18px 18px 10px 18px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(16, 32, 51, 0.08);
            box-shadow: 0 8px 24px rgba(34, 62, 94, 0.06);
            margin-bottom: 16px;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(16, 32, 51, 0.08);
        }
        details[data-testid="stExpander"] {
            border: 1px solid rgba(16, 32, 51, 0.10);
            border-radius: 16px;
            background: #f8fbfe;
        }
        details[data-testid="stExpander"] summary {
            background: #edf5ff;
            border-radius: 16px;
            color: #0f4f89 !important;
            font-weight: 800 !important;
        }
        .ai-note {
            margin: 10px 0 16px 0;
            padding: 12px 14px;
            background: #eef5fb;
            border: 1px solid #d5e4f1;
            border-radius: 14px;
            color: #16304a;
            font-size: 0.95rem;
        }
        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            background: #ffffff !important;
            color: #102033 !important;
            border-radius: 12px !important;
            border: 1px solid #9fb1c4 !important;
            font-weight: 600 !important;
        }
        .stTextInput input:focus,
        .stNumberInput input:focus,
        .stDateInput input:focus,
        .stTextArea textarea:focus,
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="base-input"]:focus-within {
            color: #102033 !important;
            border: 1px solid #0f68b0 !important;
            box-shadow: 0 0 0 3px rgba(15, 104, 176, 0.14) !important;
            outline: none !important;
        }
        input, textarea {
            color: #102033 !important;
            caret-color: #102033 !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input {
            color: #102033 !important;
        }
        div[data-testid="stNumberInput"] button,
        div[data-testid="stNumberInput"] button[kind],
        .stNumberInput button,
        .stDateInput button,
        button[aria-label="Increment"],
        button[aria-label="Decrement"] {
            background: #edf5ff !important;
            color: #0f5fa8 !important;
            border: 1px solid #c8d9ea !important;
            border-radius: 10px !important;
            min-width: 32px !important;
            min-height: 32px !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        div[data-testid="stNumberInput"] button:hover,
        div[data-testid="stNumberInput"] button[kind]:hover,
        .stNumberInput button:hover,
        .stDateInput button:hover,
        button[aria-label="Increment"]:hover,
        button[aria-label="Decrement"]:hover {
            background: #dfefff !important;
            color: #0a4c86 !important;
            border-color: #a9c6e3 !important;
        }
        div[data-testid="stNumberInput"] button:focus,
        div[data-testid="stNumberInput"] button[kind]:focus,
        .stNumberInput button:focus,
        .stDateInput button:focus,
        button[aria-label="Increment"]:focus,
        button[aria-label="Decrement"]:focus,
        div[data-testid="stNumberInput"] button:active,
        div[data-testid="stNumberInput"] button[kind]:active,
        .stNumberInput button:active,
        .stDateInput button:active,
        button[aria-label="Increment"]:active,
        button[aria-label="Decrement"]:active {
            background: #d3e8ff !important;
            color: #0a4c86 !important;
            border-color: #8db8df !important;
            box-shadow: 0 0 0 2px rgba(15, 95, 168, 0.16) !important;
            outline: none !important;
        }
        div[data-testid="stNumberInput"] button svg,
        .stNumberInput button svg,
        button[aria-label="Increment"] svg,
        button[aria-label="Decrement"] svg {
            fill: #0f5fa8 !important;
            stroke: #0f5fa8 !important;
            width: 14px !important;
            height: 14px !important;
        }
        div[data-testid="stNumberInput"] button:hover svg,
        .stNumberInput button:hover svg,
        button[aria-label="Increment"]:hover svg,
        button[aria-label="Decrement"]:hover svg,
        div[data-testid="stNumberInput"] button:focus svg,
        .stNumberInput button:focus svg,
        button[aria-label="Increment"]:focus svg,
        button[aria-label="Decrement"]:focus svg {
            fill: #0a4c86 !important;
            stroke: #0a4c86 !important;
        }
        div[data-testid="stNumberInput"] [role="button"] {
            border-radius: 10px !important;
        }
        label[data-testid="stWidgetLabel"],
        .stSlider label,
        .stSelectbox label,
        .stNumberInput label,
        .stTextInput label,
        .stCheckbox label {
            color: #0f2840 !important;
            font-weight: 700 !important;
        }
        .stButton > button,
        .stForm button,
        button[kind="primary"],
        button[kind="secondary"],
        button[kind="formSubmit"],
        button[kind="primaryFormSubmit"],
        button[kind="secondaryFormSubmit"],
        div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stFormSubmitButton"] button,
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-secondary"] {
            background: #0f5fa8 !important;
            color: #ffffff !important;
            border: 1px solid #0f5fa8 !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
            font-size: 0.98rem !important;
            min-height: 46px !important;
            box-shadow: 0 10px 24px rgba(15, 95, 168, 0.18);
            opacity: 1 !important;
        }
        .stButton > button:hover,
        .stForm button:hover,
        button[kind="primary"]:hover,
        button[kind="secondary"]:hover,
        button[kind="formSubmit"]:hover,
        button[kind="primaryFormSubmit"]:hover,
        button[kind="secondaryFormSubmit"]:hover,
        div[data-testid="stFormSubmitButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover,
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stBaseButton-secondary"]:hover {
            background: #0a4c86 !important;
            border-color: #0a4c86 !important;
            color: #ffffff !important;
        }
        .stButton > button:focus,
        .stButton > button:active,
        .stForm button:focus,
        .stForm button:active,
        button[kind="primary"]:focus,
        button[kind="primary"]:active,
        button[kind="secondary"]:focus,
        button[kind="secondary"]:active,
        button[kind="formSubmit"]:focus,
        button[kind="formSubmit"]:active,
        button[kind="primaryFormSubmit"]:focus,
        button[kind="primaryFormSubmit"]:active,
        button[kind="secondaryFormSubmit"]:focus,
        button[kind="secondaryFormSubmit"]:active,
        div[data-testid="stFormSubmitButton"] > button:focus,
        div[data-testid="stFormSubmitButton"] > button:active,
        div[data-testid="stFormSubmitButton"] button:focus,
        div[data-testid="stFormSubmitButton"] button:active,
        [data-testid="stBaseButton-primary"]:focus,
        [data-testid="stBaseButton-primary"]:active,
        [data-testid="stBaseButton-secondary"]:focus,
        [data-testid="stBaseButton-secondary"]:active {
            background: #0f5fa8 !important;
            color: #ffffff !important;
            border-color: #0f5fa8 !important;
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(15, 95, 168, 0.22) !important;
        }
        [data-testid="stWidgetLabelHelp"] {
            margin-left: 6px !important;
        }
        [data-testid="stWidgetLabelHelp"] button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            color: #0f5fa8 !important;
        }
        [data-testid="stWidgetLabelHelp"] button:hover,
        [data-testid="stWidgetLabelHelp"] button:focus {
            color: #0a4c86 !important;
            outline: none !important;
            box-shadow: none !important;
        }
        [role="tooltip"] {
            background: #17324d !important;
            color: #f8fbff !important;
            border: 1px solid rgba(255, 255, 255, 0.10) !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 24px rgba(15, 47, 74, 0.18) !important;
            font-size: 0.88rem !important;
            line-height: 1.35 !important;
            padding: 10px 12px !important;
            max-width: 320px !important;
        }
        [role="tooltip"] * {
            color: #f8fbff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(api_ok: bool, api_status: dict, total_objects: int) -> None:
    model_loaded = api_status.get("model_loaded", False)
    model_source = api_status.get("model_source", "unknown")
    status_class = "ok" if api_ok else "fail"
    status_label = "AI online" if api_ok else "AI offline"
    model_label = "Модель загружена" if model_loaded else "Fallback-режим"

    st.markdown(
        f"""
        <div class="hero-card">
            <div>
                <div class="eyebrow">DEVOPS + DOCKER PERSONAL + STREAMLIT</div>
                <h1>Определение категории значимости объекта КИИ</h1>
                <p>
                    Приложение определяет категорию значимости объекта КИИ по показателям
                    методики 8.3.1-8.3.14 и показывает методический результат вместе с вкладом ИИ-модели.
                </p>
            </div>
            <div class="hero-side">
                <div class="status-chip {status_class}">{status_label}</div>
                <div class="hero-note">Объектов в реестре: {total_objects}</div>
                <div class="hero-note">{model_label}: {model_source}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="KII Significance Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    install_styles()

    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None
    if "model_switch_feedback" not in st.session_state:
        st.session_state["model_switch_feedback"] = None

    objects = load_objects()
    ui_frame = objects_frame(objects)
    api_ok, api_status = ping_api()

    total_objects = len(objects)
    first_category = sum(1 for item in objects if item.get("predicted_category") == "Первая категория")
    significant_objects = sum(
        1 for item in objects if item.get("predicted_category") in {"Первая категория", "Вторая категория"}
    )
    avg_score = round(ui_frame["significance_score"].mean(), 1) if not ui_frame.empty else 0

    render_header(api_ok, api_status, total_objects)

    with st.sidebar:
        st.subheader("Состояние сервиса")
        st.write(f"AI API: {'online' if api_ok else 'offline'}")
        st.write(f"Модель: {api_status.get('model_source', api_status.get('status', '-'))}")
        st.write(f"Загружена: {'да' if api_status.get('model_loaded') else 'нет'}")
        st.write(f"Хранилище: `{DATA_FILE}`")

        st.divider()
        st.subheader("Выбор модели")
        active_model_choice = api_status.get("active_model_choice", "keras")
        selected_model_choice = st.selectbox(
            "Активная модель",
            options=["keras", "h5"],
            index=0 if active_model_choice == "keras" else 1,
            format_func=lambda item: "model.keras" if item == "keras" else "model.h5",
        )
        if st.button("Применить модель", width="stretch"):
            ok, payload = select_model(selected_model_choice)
            st.session_state["model_switch_feedback"] = {"ok": ok, "payload": payload}
            st.rerun()

        feedback = st.session_state.get("model_switch_feedback")
        if feedback:
            if feedback["ok"]:
                st.success(
                    f"Активна модель: {'model.keras' if feedback['payload'].get('active_model_choice') == 'keras' else 'model.h5'}"
                )
            else:
                detail = feedback["payload"].get("detail", feedback["payload"])
                st.error(f"Не удалось переключить модель: {detail}")

        if api_status.get("model_load_errors"):
            st.warning("Последняя ошибка загрузки модели:")
            for item in api_status.get("model_load_errors", [])[:2]:
                st.caption(str(item))

        st.divider()
        st.subheader("Фильтры реестра")
        category_options = sorted(ui_frame["predicted_category"].dropna().unique().tolist()) if not ui_frame.empty else []
        sector_options = sorted(ui_frame["sectors"].dropna().unique().tolist()) if not ui_frame.empty else []
        owner_options = sorted(ui_frame["ownership_level"].dropna().unique().tolist()) if not ui_frame.empty else []
        filter_category = st.multiselect("Категория", category_options)
        filter_sector = st.multiselect("Сфера", sector_options)
        filter_owner = st.multiselect("Уровень", owner_options)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Всего объектов", total_objects)
    metric_cols[1].metric("Первая категория", first_category)
    metric_cols[2].metric("Значимые объекты", significant_objects)
    metric_cols[3].metric("Средний score", avg_score)

    tab_dashboard, tab_new, tab_registry = st.tabs(["Обзор", "Новый объект", "Реестр"])

    with tab_dashboard:
        left, right = st.columns([1.3, 1], gap="large")

        with left:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Последние объекты КИИ")
            latest_frame = registry_frame(ui_frame)
            if latest_frame.empty:
                st.info("Пока нет записей в реестре.")
            else:
                st.dataframe(
                    latest_frame[["ID", "Объект КИИ", "Сферы", "Категория", "Score", "Создан"]].head(8),
                    width="stretch",
                    hide_index=True,
                    height=360,
                    column_config={
                        "Объект КИИ": st.column_config.TextColumn(width="large"),
                        "Категория": st.column_config.TextColumn(width="medium"),
                    },
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Распределение по категориям")
            if not ui_frame.empty and "predicted_category" in ui_frame:
                category_counts = (
                    ui_frame["predicted_category"]
                    .value_counts()
                    .rename_axis("Категория")
                    .reset_index(name="Количество")
                    .set_index("Категория")
                )
                st.bar_chart(category_counts, height=320)
            else:
                st.info("Недостаточно данных для графика.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Распределение по сферам")
            if not ui_frame.empty and "sectors" in ui_frame:
                sector_counts = (
                    ui_frame["sectors"]
                    .value_counts()
                    .rename_axis("Сферы")
                    .reset_index(name="Количество")
                    .set_index("Сферы")
                )
                st.bar_chart(sector_counts, height=260)
            else:
                st.info("Недостаточно данных для графика.")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_new:
        form_col, result_col = st.columns([1.45, 0.9], gap="large")

        with form_col:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Характеристики объекта КИИ")

            with st.form("kii_object_form", clear_on_submit=False):
                c1, c2, c3 = st.columns(3)
                object_name = c1.text_input("Наименование объекта", "Региональный диспетчерский центр")
                field_hint(c1, "Полное наименование объекта КИИ для расчёта категории.")
                sectors = c2.multiselect(
                    "Сферы деятельности",
                    list(SECTOR_LABELS.keys()),
                    default=["communications"],
                    format_func=SECTOR_LABELS.get,
                )
                field_hint(c2, "Выберите одну или несколько сфер из 14 сфер КИИ по 187-ФЗ.")
                ownership_level = c3.selectbox(
                    "Уровень объекта",
                    list(OWNERSHIP_LABELS.keys()),
                    format_func=OWNERSHIP_LABELS.get,
                )
                field_hint(c3, "Федеральный, региональный, муниципальный или частный уровень.")

                c4, c5, c6 = st.columns(3)
                service_scale = c4.selectbox(
                    "Масштаб услуг",
                    list(SERVICE_SCALE_LABELS.keys()),
                    format_func=SERVICE_SCALE_LABELS.get,
                )
                field_hint(c4, "Уровень оказания услуг: локальный, региональный, федеральный или межотраслевой.")
                process_criticality = c5.slider(
                    "Критичность процесса",
                    1,
                    10,
                    7,
                )
                field_hint(c5, "Экспертная оценка критичности поддерживаемого процесса.")
                object_type = c6.selectbox(
                    "Тип объекта",
                    list(OBJECT_TYPE_LABELS.keys()),
                    format_func=OBJECT_TYPE_LABELS.get,
                    index=0,
                )
                field_hint(c6, "Информационная система, сеть или АСУ/технологический контур.")

                c7, c8, c9 = st.columns(3)
                critical_process_count = c7.number_input(
                    "Критичных процессов",
                    min_value=0,
                    value=4,
                    step=1,
                )
                field_hint(c7, "Сколько критичных процессов зависит от объекта.")
                interaction_count = c8.number_input(
                    "Интеграций и связей",
                    min_value=0,
                    value=15,
                    step=1,
                )
                field_hint(c8, "Количество внешних и внутренних зависимостей объекта.")
                continuous_operation = c9.checkbox(
                    "Непрерывный режим",
                    value=True,
                )
                field_hint(c9, "Объект должен работать без длительных остановок.")

                b1, b2 = st.columns(2)
                scada_used = b1.checkbox("Используется АСУ ТП", value=False)
                field_hint(b1, "Отметьте, если объект связан с технологическим контуром или АСУ.")
                classified_info = b2.checkbox("Есть чувствительная информация", value=False)
                field_hint(b2, "Отметьте, если объект обрабатывает чувствительные или ограниченные данные.")

                st.markdown("**Показатели методики 8.3.1-8.3.14**")
                st.caption(
                    "Для каждого показателя отметьте применимость и укажите уже рассчитанный уровень: "
                    "0 - ниже порога, 1 - третья категория, 2 - вторая, 3 - первая."
                )

                criterion_inputs: dict[str, tuple[bool, int]] = {}
                level_labels = {
                    0: "0 - ниже порога / без категории",
                    1: "1 - третья категория",
                    2: "2 - вторая категория",
                    3: "3 - первая категория",
                }
                for criterion in CRITERIA:
                    st.markdown(f"**{criterion['code']} {criterion['title']}**")
                    col_a, col_b = st.columns([1, 2])
                    applicable = col_a.checkbox(
                        "Применимо",
                        value=False,
                        key=f"{criterion['id']}_applicable_ui",
                    )
                    field_hint(col_a, criterion["hint"])
                    level = col_b.selectbox(
                        "Уровень показателя",
                        options=[0, 1, 2, 3],
                        index=0,
                        format_func=lambda item, mapping=level_labels: mapping[item],
                        key=f"{criterion['id']}_level_ui",
                    )
                    criterion_inputs[criterion["id"]] = (applicable, int(level))

                submitted = st.form_submit_button(
                    "Отправить ИИ-агенту",
                    width="stretch",
                )

                if submitted:
                    if not sectors:
                        st.error("Выберите хотя бы одну сферу деятельности объекта.")
                        st.stop()

                    payload = {
                        "object_name": object_name,
                        "sectors": sectors,
                        "ownership_level": ownership_level,
                        "service_scale": service_scale,
                        "object_type": object_type,
                        "process_criticality": int(process_criticality),
                        "critical_process_count": int(critical_process_count),
                        "interaction_count": int(interaction_count),
                        "continuous_operation": continuous_operation,
                        "scada_used": scada_used,
                        "classified_info": classified_info,
                    }
                    for criterion in CRITERIA:
                        applicable, level = criterion_inputs[criterion["id"]]
                        payload[criterion_applicable_field(criterion["id"])] = applicable
                        payload[criterion_level_field(criterion["id"])] = level

                    prediction = predict_object(payload)
                    if prediction is None:
                        st.error("AI API недоступен. Проверьте контейнер `inference-api`.")
                    else:
                        record = {
                            "id": f"OBJ-{uuid4().hex[:6].upper()}",
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            **payload,
                            **prediction,
                        }
                        objects.append(record)
                        save_objects(objects)
                        st.session_state["last_prediction"] = record
                        st.success(
                            f"Категория определена: {prediction['predicted_category']}, "
                            f"score: {prediction['significance_score']}"
                        )

            st.markdown("</div>", unsafe_allow_html=True)

        with result_col:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Последняя AI-оценка")
            last_prediction = st.session_state.get("last_prediction")
            if not last_prediction:
                st.info("После расчета здесь появятся категория, методическое обоснование и вклад модели.")
            else:
                top1, top2 = st.columns(2)
                top1.metric("Score", int(last_prediction.get("significance_score", 0)))
                top2.metric("Уверенность", f"{float(last_prediction.get('confidence', 0)):.2f}")
                st.write(f"**Категория:** {last_prediction.get('predicted_category', '-')}")
                st.write(f"**Роль модели:** {last_prediction.get('model_role', '-')}")
                st.write(f"**Дата создания записи:** {last_prediction.get('created_at', '-')}")
                st.write("**Резюме**")
                st.write(last_prediction.get("summary", "-"))
                st.write("**Ключевые факторы**")
                for item in last_prediction.get("key_factors", []):
                    st.markdown(f"- {item}")
                st.write("**Показатели 8.3.*:**")
                for item in last_prediction.get("criterion_assessments", []):
                    mark = "применим" if item.get("applicable") else "неприменим"
                    st.markdown(
                        f"- {item.get('code')} {item.get('title')}: {item.get('category_name')} ({mark})"
                    )
                st.write("**Рекомендации**")
                for item in last_prediction.get("recommendations", []):
                    st.markdown(f"- {item}")

                with st.expander("Показать вклад ИИ", expanded=False):
                    st.markdown(
                        """
                        <div class="ai-note">
                            Итоговая категория определяется по методике как максимум из применимых
                            показателей 8.3.1-8.3.14. Модель используется как дополнительный
                            классификатор и источник confidence, но не подменяет правило методики.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write(f"**Источник модели:** {last_prediction.get('model_source', '-')}")
                    st.write(f"**Модель загружена:** {'да' if last_prediction.get('model_loaded') else 'нет'}")

                    ai_col1, ai_col2 = st.columns(2, gap="medium")
                    with ai_col1:
                        st.caption("Эвристика")
                        st.dataframe(
                            probability_frame(last_prediction.get("heuristic_probabilities", {})),
                            width="stretch",
                            hide_index=True,
                            height=180,
                            column_config={
                                "Категория": st.column_config.TextColumn(width="large"),
                                "Вероятность": st.column_config.NumberColumn(format="%.2f"),
                            },
                        )
                    with ai_col2:
                        st.caption("Модель")
                        st.dataframe(
                            probability_frame(last_prediction.get("model_probabilities", {})),
                            width="stretch",
                            hide_index=True,
                            height=180,
                            column_config={
                                "Категория": st.column_config.TextColumn(width="large"),
                                "Вероятность": st.column_config.NumberColumn(format="%.2f"),
                            },
                        )

                    st.caption("Итоговое объединение")
                    st.dataframe(
                        probability_frame(last_prediction.get("combined_probabilities", {})),
                        width="stretch",
                        hide_index=True,
                        height=210,
                        column_config={
                            "Категория": st.column_config.TextColumn(width="large"),
                            "Вероятность": st.column_config.NumberColumn(format="%.2f"),
                        },
                    )

            st.markdown("</div>", unsafe_allow_html=True)

    with tab_registry:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Реестр объектов КИИ")

        filtered = ui_frame.copy()
        if filter_category:
            filtered = filtered[filtered["predicted_category"].isin(filter_category)]
        if filter_sector:
            filtered = filtered[filtered["sectors"].isin(filter_sector)]
        if filter_owner:
            filtered = filtered[filtered["ownership_level"].isin(filter_owner)]

        registry = registry_frame(filtered)
        if registry.empty:
            st.warning("По текущим фильтрам записи не найдены.")
        else:
            st.dataframe(
                registry,
                width="stretch",
                hide_index=True,
                height=540,
                column_config={
                    "Объект КИИ": st.column_config.TextColumn(width="large"),
                    "Категория": st.column_config.TextColumn(width="medium"),
                    "Сферы": st.column_config.TextColumn(width="large"),
                    "Уровень": st.column_config.TextColumn(width="medium"),
                    "Масштаб": st.column_config.TextColumn(width="medium"),
                    "Создан": st.column_config.TextColumn(width="medium"),
                    "Уверенность": st.column_config.NumberColumn(format="%.2f"),
                },
            )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
