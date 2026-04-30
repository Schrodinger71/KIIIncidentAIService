import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_FILE = Path(os.getenv("DATA_FILE", "data/objects.json"))

SECTOR_LABELS = {
    "energy": "Энергетика",
    "health": "Здравоохранение",
    "finance": "Финансы",
    "transport": "Транспорт",
    "government": "Госорганы",
    "communications": "Связь",
    "industry": "Промышленность",
}

OWNERSHIP_LABELS = {
    "federal": "Федеральный",
    "regional": "Региональный",
    "municipal": "Муниципальный",
    "private": "Частный",
}

SCALE_LABELS = {
    "local": "Локальный",
    "regional": "Региональный",
    "federal": "Федеральный",
    "intersectoral": "Межотраслевой",
}

CATEGORY_ORDER = [
    "Без категории",
    "Третья категория",
    "Вторая категория",
    "Первая категория",
]


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
                "sector",
                "ownership_level",
                "service_scale",
                "predicted_category",
                "significance_score",
                "created_at",
            ]
        )

    frame = pd.DataFrame(items).copy()
    frame["sector"] = frame["sector"].map(SECTOR_LABELS).fillna(frame["sector"])
    frame["ownership_level"] = frame["ownership_level"].map(OWNERSHIP_LABELS).fillna(frame["ownership_level"])
    frame["service_scale"] = frame["service_scale"].map(SCALE_LABELS).fillna(frame["service_scale"])
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    return frame


def registry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    registry = frame.copy()
    registry["created_at"] = registry["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    registry = registry.rename(
        columns={
            "id": "ID",
            "object_name": "Объект КИИ",
            "sector": "Сектор",
            "ownership_level": "Уровень",
            "service_scale": "Масштаб",
            "process_criticality": "Критичность",
            "supported_users": "Пользователи",
            "significance_score": "Score",
            "predicted_category": "Категория",
            "confidence": "Уверенность",
            "created_at": "Создан",
        }
    )
    columns = [
        "ID",
        "Объект КИИ",
        "Сектор",
        "Уровень",
        "Масштаб",
        "Критичность",
        "Пользователи",
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
                    Приложение определяет категорию значимости объекта КИИ по набору из 15+
                    характеристик и показывает вклад эвристики и ИИ-модели в итоговый результат.
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
        sector_options = sorted(ui_frame["sector"].dropna().unique().tolist()) if not ui_frame.empty else []
        owner_options = sorted(ui_frame["ownership_level"].dropna().unique().tolist()) if not ui_frame.empty else []
        filter_category = st.multiselect("Категория", category_options)
        filter_sector = st.multiselect("Сектор", sector_options)
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
                    latest_frame[["ID", "Объект КИИ", "Сектор", "Категория", "Score", "Создан"]].head(8),
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
            st.subheader("Распределение по секторам")
            if not ui_frame.empty and "sector" in ui_frame:
                sector_counts = (
                    ui_frame["sector"]
                    .value_counts()
                    .rename_axis("Сектор")
                    .reset_index(name="Количество")
                    .set_index("Сектор")
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
                sector = c2.selectbox(
                    "Сектор КИИ",
                    list(SECTOR_LABELS.keys()),
                    format_func=SECTOR_LABELS.get,
                )
                field_hint(c2, "Отрасль, к которой относится объект КИИ.")
                ownership_level = c3.selectbox(
                    "Уровень объекта",
                    list(OWNERSHIP_LABELS.keys()),
                    format_func=OWNERSHIP_LABELS.get,
                )
                field_hint(c3, "Федеральный, региональный, муниципальный или частный уровень.")

                c4, c5, c6 = st.columns(3)
                service_scale = c4.selectbox(
                    "Масштаб услуг",
                    list(SCALE_LABELS.keys()),
                    format_func=SCALE_LABELS.get,
                )
                field_hint(c4, "Уровень оказания услуг: локальный, региональный, федеральный или межотраслевой.")
                process_criticality = c5.slider(
                    "Критичность процесса",
                    1,
                    10,
                    7,
                )
                field_hint(c5, "Экспертная оценка критичности поддерживаемого процесса.")
                territories_count = c6.number_input(
                    "Количество территорий",
                    min_value=1,
                    value=3,
                    step=1,
                )
                field_hint(c6, "Сколько территорий или площадок затронет отказ объекта.")

                c7, c8, c9 = st.columns(3)
                supported_users = c7.number_input(
                    "Количество пользователей",
                    min_value=0,
                    value=250000,
                    step=1000,
                )
                field_hint(c7, "Сколько пользователей, клиентов или потребителей зависит от объекта.")
                annual_financial_loss_million = c8.number_input(
                    "Потери, млн руб.",
                    min_value=0.0,
                    value=150.0,
                    step=10.0,
                )
                field_hint(c8, "Оценка финансового ущерба при отказе объекта.")
                recovery_time_hours = c9.number_input(
                    "Восстановление, ч",
                    min_value=0,
                    value=12,
                    step=1,
                )
                field_hint(c9, "Время восстановления нормальной работы после отказа.")

                c10, c11, c12 = st.columns(3)
                critical_processes = c10.number_input(
                    "Критичных процессов",
                    min_value=0,
                    value=4,
                    step=1,
                )
                field_hint(c10, "Сколько критичных процессов зависит от объекта.")
                interactions_count = c11.number_input(
                    "Интеграций и связей",
                    min_value=0,
                    value=15,
                    step=1,
                )
                field_hint(c11, "Количество связанных внутренних и внешних систем.")
                personal_data_subjects = c12.number_input(
                    "Субъектов ПДн",
                    min_value=0,
                    value=50000,
                    step=1000,
                )
                field_hint(c12, "Сколько субъектов персональных данных затрагивает объект.")

                employees_affected = st.number_input(
                    "Персонал под влиянием отказа",
                    min_value=0,
                    value=1500,
                    step=10,
                )
                st.caption("Сколько сотрудников потеряют доступ к критичным функциям при отказе объекта.")

                st.markdown("**Признаки значимости объекта**")
                b1, b2, b3, b4 = st.columns(4)
                continuous_operation = b1.checkbox(
                    "Непрерывный режим",
                    value=True,
                )
                field_hint(b1, "Объект должен работать в режиме 24/7 без длительных остановок.")
                scada_used = b2.checkbox("Используется АСУ ТП")
                field_hint(b2, "Объект связан с АСУ ТП или технологическим контуром.")
                government_services = b3.checkbox("Оказывает госуслуги")
                field_hint(b3, "Объект участвует в оказании государственных или муниципальных услуг.")
                life_safety_impact = b4.checkbox("Влияние на жизнь и здоровье")
                field_hint(b4, "Отказ объекта влияет на жизнь и здоровье людей.")

                b5, b6, b7, b8 = st.columns(4)
                ecological_impact = b5.checkbox("Экологические последствия")
                field_hint(b5, "Нарушение работы объекта может вызвать экологический ущерб.")
                defense_impact = b6.checkbox("Влияние на оборону")
                field_hint(b6, "Объект влияет на оборону или безопасность государства.")
                public_order_impact = b7.checkbox("Влияние на общественный порядок")
                field_hint(b7, "Отказ объекта может нарушить общественный порядок.")
                transport_disruption = b8.checkbox("Нарушение транспорта")
                field_hint(b8, "Отказ объекта способен нарушить транспортную инфраструктуру.")

                b9, b10 = st.columns(2)
                communications_disruption = b9.checkbox("Нарушение связи")
                field_hint(b9, "Нарушение работы объекта влияет на связь или обмен данными.")
                classified_info = b10.checkbox("Обрабатывается чувствительная информация")
                field_hint(b10, "Объект обрабатывает чувствительные, ограниченные или критичные данные.")

                submitted = st.form_submit_button(
                    "Отправить ИИ-агенту",
                    width="stretch",
                )

                if submitted:
                    payload = {
                        "object_name": object_name,
                        "sector": sector,
                        "ownership_level": ownership_level,
                        "service_scale": service_scale,
                        "process_criticality": int(process_criticality),
                        "supported_users": int(supported_users),
                        "territories_count": int(territories_count),
                        "annual_financial_loss_million": float(annual_financial_loss_million),
                        "recovery_time_hours": int(recovery_time_hours),
                        "critical_processes": int(critical_processes),
                        "interactions_count": int(interactions_count),
                        "personal_data_subjects": int(personal_data_subjects),
                        "employees_affected": int(employees_affected),
                        "continuous_operation": continuous_operation,
                        "scada_used": scada_used,
                        "government_services": government_services,
                        "life_safety_impact": life_safety_impact,
                        "ecological_impact": ecological_impact,
                        "defense_impact": defense_impact,
                        "public_order_impact": public_order_impact,
                        "transport_disruption": transport_disruption,
                        "communications_disruption": communications_disruption,
                        "classified_info": classified_info,
                    }

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
                st.info("После расчета здесь появятся категория, score, причины и вклад модели.")
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
                st.write("**Рекомендации**")
                for item in last_prediction.get("recommendations", []):
                    st.markdown(f"- {item}")

                with st.expander("Показать вклад ИИ", expanded=False):
                    st.markdown(
                        """
                        <div class="ai-note">
                            Эвристика определяет базовую категорию значимости по последствиям,
                            а нейросеть добавляет второй сигнал. Итоговая категория берется из
                            объединенного результата.
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
            filtered = filtered[filtered["sector"].isin(filter_sector)]
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
                    "Сектор": st.column_config.TextColumn(width="medium"),
                    "Уровень": st.column_config.TextColumn(width="medium"),
                    "Масштаб": st.column_config.TextColumn(width="medium"),
                    "Создан": st.column_config.TextColumn(width="medium"),
                    "Уверенность": st.column_config.NumberColumn(format="%.2f"),
                },
            )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
