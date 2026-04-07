from __future__ import annotations

from pathlib import Path
import time

import altair as alt
import pandas as pd
import streamlit as st

from components.shared_components import (
    api_call_with_retry,
    build_headers,
    get_setting,
    show_api_status,
    show_footer,
)
from components.shared_theme import THEME, risk_color, risk_emoji

API_URL = get_setting("API_URL", "https://medicaidguard-api-5tphgb6fsa-as.a.run.app").rstrip("/")
API_KEY = get_setting("API_KEY", "")
HEADERS = build_headers(API_KEY)
SAMPLE_BATCH_PATH = Path(__file__).resolve().parent / "sample_data" / "sample_batch_100.csv"

EXAMPLE_SCENARIOS = {
    "🟢 Normal Claim": {
        "claim_amount": 250.0,
        "procedure_code": "99213",
        "diagnosis_code": "J06.9",
        "provider_type": "organization",
        "patient_age": 35,
        "claim_frequency_30d": 3,
        "avg_claim_amount_90d": 200.0,
        "unique_patients_30d": 150,
        "billing_pattern_score": 0.15,
    },
    "🟡 Suspicious": {
        "claim_amount": 1760.95,
        "procedure_code": "99213",
        "diagnosis_code": "E11.9",
        "provider_type": "individual",
        "patient_age": 51,
        "claim_frequency_30d": 8,
        "avg_claim_amount_90d": 4895.52,
        "unique_patients_30d": 86,
        "billing_pattern_score": 0.512,
    },
    "🔴 Likely Fraud": {
        "claim_amount": 15000.0,
        "procedure_code": "99215",
        "diagnosis_code": "Z00.00",
        "provider_type": "individual",
        "patient_age": 45,
        "claim_frequency_30d": 35,
        "avg_claim_amount_90d": 12000.0,
        "unique_patients_30d": 8,
        "billing_pattern_score": 0.92,
    },
}

REQUIRED_COLUMNS = [
    "transaction_id",
    "provider_id",
    "claim_amount",
    "procedure_code",
    "diagnosis_code",
    "provider_type",
    "patient_age",
    "claim_frequency_30d",
    "avg_claim_amount_90d",
    "unique_patients_30d",
    "billing_pattern_score",
]

RISK_ACTIONS = {
    "LOW": "APPROVE",
    "MEDIUM": "REVIEW",
    "HIGH": "ESCALATE",
    "CRITICAL": "BLOCK",
}

FEATURE_LIST = [
    "claim_amount",
    "procedure_code",
    "diagnosis_code",
    "provider_type",
    "patient_age",
    "claim_frequency_30d",
    "avg_claim_amount_90d",
    "unique_patients_30d",
    "billing_pattern_score",
]

RUN_STATES = ["Idle", "Queued", "Processing", "Success", "Error"]


st.set_page_config(page_title="MedicaidGuard Demo", page_icon="🛡️", layout="wide")

st.markdown(
    f"""
    <style>
      .stApp {{
        background: linear-gradient(180deg, {THEME['background']} 0%, #111827 100%);
      }}
      .main-header {{
        font-size: 2rem;
        font-weight: 700;
      }}
      .sub-header {{
        color: #94A3B8;
        margin-bottom: 1rem;
      }}
      .risk-badge {{
        display: inline-block;
        font-weight: 700;
        color: white;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        margin-bottom: 0.35rem;
      }}
      .run-state-pill {{
        display: inline-block;
        border-radius: 999px;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
      }}
      .state-idle {{ background:#334155; color:#E2E8F0; }}
      .state-queued {{ background:#A16207; color:#FEF3C7; }}
      .state-processing {{ background:#1D4ED8; color:#DBEAFE; }}
      .state-success {{ background:#166534; color:#DCFCE7; }}
      .state-error {{ background:#991B1B; color:#FEE2E2; }}
      .sticky-summary {{
        position: sticky;
        top: 0.35rem;
        z-index: 20;
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.9rem;
      }}
      .stTabs [data-baseweb="tab-list"] {{
        gap: 0.35rem;
        flex-wrap: wrap;
      }}
      .stTabs [data-baseweb="tab"] {{
        background: #1F2937;
        border: 1px solid #334155;
        border-radius: 999px;
        height: auto;
        padding: 0.35rem 0.8rem;
      }}
      .stTabs [aria-selected="true"] {{
        background: #0C4A6E !important;
        border-color: #0EA5E9 !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _response_error(response: dict) -> str | None:
    error = response.get("error")
    if error:
        return str(response.get("detail") or error)
    return None


def _init_state() -> None:
    normal = EXAMPLE_SCENARIOS["🟢 Normal Claim"]
    defaults = {
        "selected_scenario": "🟢 Normal Claim",
        "single_transaction_id": "TXN-DEMO-0001",
        "single_provider_id": "PRV-DEMO-1001",
        "single_claim_amount": normal["claim_amount"],
        "single_procedure_code": normal["procedure_code"],
        "single_diagnosis_code": normal["diagnosis_code"],
        "single_provider_type": normal["provider_type"],
        "single_patient_age": normal["patient_age"],
        "single_claim_frequency_30d": normal["claim_frequency_30d"],
        "single_avg_claim_amount_90d": normal["avg_claim_amount_90d"],
        "single_unique_patients_30d": normal["unique_patients_30d"],
        "single_billing_pattern_score": normal["billing_pattern_score"],
        "single_run_state": "Idle",
        "single_run_detail": "Ready to run claim analysis",
        "last_single_error": "",
        "single_result": None,
        "batch_df": None,
        "batch_result": None,
        "triage_selected_id": None,
        "kpi_snapshot": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _set_single_run_state(state: str, detail: str) -> None:
    if state not in RUN_STATES:
        state = "Error"
    st.session_state.single_run_state = state
    st.session_state.single_run_detail = detail


def _render_single_run_state() -> None:
    state = st.session_state.single_run_state
    st.markdown(
        f"<span class='run-state-pill state-{state.lower()}'>{state}</span> "
        f"<span style='color:#94A3B8'>{st.session_state.single_run_detail}</span>",
        unsafe_allow_html=True,
    )


def _scenario_to_state(name: str) -> None:
    scenario = EXAMPLE_SCENARIOS[name]
    st.session_state.single_claim_amount = scenario["claim_amount"]
    st.session_state.single_procedure_code = scenario["procedure_code"]
    st.session_state.single_diagnosis_code = scenario["diagnosis_code"]
    st.session_state.single_provider_type = scenario["provider_type"]
    st.session_state.single_patient_age = scenario["patient_age"]
    st.session_state.single_claim_frequency_30d = scenario["claim_frequency_30d"]
    st.session_state.single_avg_claim_amount_90d = scenario["avg_claim_amount_90d"]
    st.session_state.single_unique_patients_30d = scenario["unique_patients_30d"]
    st.session_state.single_billing_pattern_score = scenario["billing_pattern_score"]
    st.session_state.single_transaction_id = f"TXN-DEMO-{int(time.time())}"
    st.session_state.single_provider_id = "PRV-DEMO-1001"
    st.session_state.single_result = None
    _set_single_run_state("Idle", "Scenario loaded. Adjust fields or run analysis.")


def _load_selected_scenario() -> None:
    _scenario_to_state(st.session_state.selected_scenario)


def _single_payload() -> dict:
    return {
        "transaction_id": st.session_state.single_transaction_id,
        "provider_id": st.session_state.single_provider_id,
        "claim_amount": float(st.session_state.single_claim_amount),
        "procedure_code": st.session_state.single_procedure_code,
        "diagnosis_code": st.session_state.single_diagnosis_code,
        "provider_type": st.session_state.single_provider_type,
        "patient_age": int(st.session_state.single_patient_age),
        "claim_frequency_30d": int(st.session_state.single_claim_frequency_30d),
        "avg_claim_amount_90d": float(st.session_state.single_avg_claim_amount_90d),
        "unique_patients_30d": int(st.session_state.single_unique_patients_30d),
        "billing_pattern_score": float(st.session_state.single_billing_pattern_score),
    }


def _format_risk_label(risk_level: str) -> str:
    return f"{risk_emoji(risk_level)} {risk_level}"


def _result_summary(result: dict) -> tuple[str, str, str]:
    risk_level = str(result.get("risk_level", "LOW"))
    action = RISK_ACTIONS.get(risk_level, "REVIEW")
    factors = result.get("top_risk_factors") or []
    top_factor = "n/a"
    if factors:
        top_factor = str(factors[0].get("feature", "n/a"))
    return risk_level, action, top_factor


def _render_sticky_summary(result: dict) -> None:
    risk_level, action, top_factor = _result_summary(result)
    st.markdown(
        "<div class='sticky-summary'>"
        "<strong>Latest Result</strong><br/>"
        f"Risk: <strong>{risk_emoji(risk_level)} {risk_level}</strong>"
        f" &nbsp;|&nbsp; Action: <strong>{action}</strong>"
        f" &nbsp;|&nbsp; Top Factor: <strong>{top_factor}</strong>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_single_result(result: dict) -> None:
    probability = float(result.get("fraud_probability", 0.0))
    risk_level = str(result.get("risk_level", "LOW"))
    action = RISK_ACTIONS.get(risk_level, "REVIEW")

    st.markdown("### Result Details")
    st.progress(min(max(probability, 0.0), 1.0))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fraud Score", f"{probability * 100:.1f}%")
    with col2:
        st.metric("Risk Level", risk_level)
    with col3:
        st.metric("Recommended Action", action)

    badge = (
        f"<span class='risk-badge' style='background:{risk_color(risk_level)}'>"
        f"{_format_risk_label(risk_level)}</span>"
    )
    st.markdown(badge, unsafe_allow_html=True)
    st.caption(
        f"Inference: {result.get('inference_time_ms', '-')} ms | "
        f"Model: {result.get('model_version', 'unknown')}"
    )

    factors = result.get("top_risk_factors") or []
    if factors:
        st.markdown("### Top Risk Factors")
        factor_df = pd.DataFrame(
            {
                "feature": [f.get("feature", "unknown") for f in factors],
                "importance": [float(f.get("importance", 0.0)) for f in factors],
            }
        ).sort_values("importance", ascending=True)
        st.bar_chart(factor_df.set_index("feature"))


def _top_factor(factors: list[dict] | None) -> str:
    if isinstance(factors, list) and factors:
        return str(factors[0].get("feature", "n/a"))
    return "n/a"


def _run_single_analysis() -> None:
    state_box = st.empty()
    _set_single_run_state("Queued", "Claim queued for model scoring")
    state_box.info("Run State: Queued - Claim queued for model scoring")
    time.sleep(0.08)

    _set_single_run_state("Processing", "Model inference in progress")
    state_box.info("Run State: Processing - Model inference in progress")

    response = api_call_with_retry(
        f"{API_URL}/predict",
        method="POST",
        headers=HEADERS,
        json_payload=_single_payload(),
        timeout=90,
        max_retries=2,
    )
    response_error = _response_error(response)
    if response_error:
        st.session_state.last_single_error = response_error
        _set_single_run_state("Error", response_error)
        state_box.error(f"Run State: Error - {response_error}")
        st.session_state.single_result = None
        return

    st.session_state.last_single_error = ""
    _set_single_run_state("Success", "Prediction complete")
    state_box.success("Run State: Success - Prediction complete")
    st.session_state.single_result = response


def _run_batch_analysis(batch_df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in batch_df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    payload_rows = []
    for row in batch_df[REQUIRED_COLUMNS].to_dict(orient="records"):
        payload_rows.append(
            {
                "transaction_id": str(row["transaction_id"]),
                "provider_id": str(row["provider_id"]),
                "claim_amount": float(row["claim_amount"]),
                "procedure_code": str(row["procedure_code"]),
                "diagnosis_code": str(row["diagnosis_code"]),
                "provider_type": str(row["provider_type"]),
                "patient_age": int(row["patient_age"]),
                "claim_frequency_30d": int(row["claim_frequency_30d"]),
                "avg_claim_amount_90d": float(row["avg_claim_amount_90d"]),
                "unique_patients_30d": int(row["unique_patients_30d"]),
                "billing_pattern_score": float(row["billing_pattern_score"]),
            }
        )

    with st.spinner("Scoring transactions..."):
        response = api_call_with_retry(
            f"{API_URL}/predict/batch",
            method="POST",
            headers=HEADERS,
            json_payload={"transactions": payload_rows},
            timeout=180,
            max_retries=2,
        )

    response_error = _response_error(response)
    if response_error:
        st.error(response_error)
        st.session_state.batch_result = None
        return

    st.session_state.batch_result = response


def _build_triage_frame() -> pd.DataFrame:
    rows: list[dict] = []
    batch_result = st.session_state.batch_result
    if isinstance(batch_result, dict):
        predictions = batch_result.get("predictions") or []
        if predictions:
            frame = pd.DataFrame(predictions)
            if st.session_state.batch_df is not None and "transaction_id" in st.session_state.batch_df.columns:
                lookup_columns = ["transaction_id", "provider_id", "claim_amount"]
                available_cols = [col for col in lookup_columns if col in st.session_state.batch_df.columns]
                if available_cols:
                    lookup = st.session_state.batch_df[available_cols].copy()
                    lookup["transaction_id"] = lookup["transaction_id"].astype(str)
                    frame["transaction_id"] = frame["transaction_id"].astype(str)
                    frame = frame.merge(lookup, on="transaction_id", how="left", suffixes=("", "_input"))
                    for column in ("provider_id", "claim_amount"):
                        input_col = f"{column}_input"
                        if input_col in frame.columns:
                            frame[column] = frame[column].fillna(frame[input_col])
                            frame = frame.drop(columns=[input_col])

            for _, row in frame.iterrows():
                factors = row.get("top_risk_factors")
                rows.append(
                    {
                        "transaction_id": str(row.get("transaction_id", "n/a")),
                        "provider_id": str(row.get("provider_id", "n/a")),
                        "claim_amount": float(row.get("claim_amount", 0.0) or 0.0),
                        "fraud_probability": float(row.get("fraud_probability", 0.0) or 0.0),
                        "risk_level": str(row.get("risk_level", "LOW")),
                        "prediction": str(row.get("prediction", "LEGITIMATE")),
                        "recommended_action": RISK_ACTIONS.get(str(row.get("risk_level", "LOW")), "REVIEW"),
                        "top_factor": _top_factor(factors),
                        "top_risk_factors": factors if isinstance(factors, list) else [],
                        "inference_time_ms": float(row.get("inference_time_ms", 0.0) or 0.0),
                    }
                )

    single = st.session_state.single_result
    if isinstance(single, dict) and not single.get("error"):
        rows.append(
            {
                "transaction_id": str(single.get("transaction_id", st.session_state.single_transaction_id)),
                "provider_id": st.session_state.single_provider_id,
                "claim_amount": float(st.session_state.single_claim_amount),
                "fraud_probability": float(single.get("fraud_probability", 0.0)),
                "risk_level": str(single.get("risk_level", "LOW")),
                "prediction": str(single.get("prediction", "LEGITIMATE")),
                "recommended_action": RISK_ACTIONS.get(str(single.get("risk_level", "LOW")), "REVIEW"),
                "top_factor": _top_factor(single.get("top_risk_factors") or []),
                "top_risk_factors": single.get("top_risk_factors") or [],
                "inference_time_ms": float(single.get("inference_time_ms", 0.0) or 0.0),
            }
        )

    if not rows:
        return pd.DataFrame()

    triage = pd.DataFrame(rows).drop_duplicates(subset=["transaction_id"], keep="last")
    triage = triage.sort_values(["fraud_probability", "claim_amount"], ascending=[False, False])
    return triage.reset_index(drop=True)


def _local_kpi_fallback(triage_df: pd.DataFrame) -> dict:
    total_predictions = int(len(triage_df))
    flagged = triage_df[triage_df["prediction"] == "FRAUD"] if not triage_df.empty else pd.DataFrame()
    total_fraud_detected = int(len(flagged))
    flagged_amount = float(flagged["claim_amount"].fillna(0.0).sum()) if not flagged.empty else 0.0

    if triage_df.empty:
        avg_latency = 0.0
        p99_latency = 0.0
    else:
        avg_latency = float(triage_df["inference_time_ms"].mean())
        p99_latency = float(triage_df["inference_time_ms"].quantile(0.99))

    return {
        "total_predictions": total_predictions,
        "total_fraud_detected": total_fraud_detected,
        "avg_inference_time_ms": avg_latency,
        "p99_inference_time_ms": p99_latency,
        "flagged_amount": flagged_amount,
    }


def _refresh_kpis() -> dict:
    response = api_call_with_retry(
        f"{API_URL}/metrics",
        method="GET",
        headers=HEADERS,
        timeout=20,
        max_retries=1,
    )
    if _response_error(response):
        return st.session_state.kpi_snapshot or {}
    st.session_state.kpi_snapshot = response
    return response


def _render_kpi_header(kpi_data: dict, fallback_kpis: dict) -> None:
    total_scanned = int(kpi_data.get("total_predictions", fallback_kpis["total_predictions"]))
    total_fraud = int(kpi_data.get("total_fraud_detected", fallback_kpis["total_fraud_detected"]))
    avg_latency = float(kpi_data.get("avg_inference_time_ms", fallback_kpis["avg_inference_time_ms"]))
    p99_latency = float(kpi_data.get("p99_inference_time_ms", fallback_kpis["p99_inference_time_ms"]))
    flagged_amount = float(fallback_kpis["flagged_amount"])
    fraud_rate = (total_fraud / max(total_scanned, 1)) * 100 if total_scanned else 0.0

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Total Claims Scanned", f"{total_scanned:,}")
    with kpi_cols[1]:
        st.metric("Fraud Flagged ($)", f"${flagged_amount:,.0f}", delta=f"{fraud_rate:.1f}% flagged")
    with kpi_cols[2]:
        st.metric("Avg Latency", f"{avg_latency:.1f} ms")
    with kpi_cols[3]:
        st.metric("P99 Latency", f"{p99_latency:.1f} ms")


def _render_distribution(triage_df: pd.DataFrame) -> None:
    if triage_df.empty:
        st.info("Run single or batch analysis to populate mission-control charts.")
        return

    distribution = (
        triage_df["risk_level"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
        .rename_axis("risk_level")
        .reset_index(name="count")
    )
    chart = (
        alt.Chart(distribution)
        .mark_arc(innerRadius=55)
        .encode(
            theta=alt.Theta(field="count", type="quantitative"),
            color=alt.Color(
                field="risk_level",
                type="nominal",
                scale=alt.Scale(
                    domain=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    range=["#2A9D8F", "#F4A261", "#E76F51", "#E63946"],
                ),
            ),
            tooltip=["risk_level", "count"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def _risk_row_style(row: pd.Series) -> list[str]:
    level = str(row.get("Risk Level", "LOW")).upper()
    if level in {"HIGH", "CRITICAL"}:
        return ["background-color: rgba(230, 57, 70, 0.16)"] * len(row)
    if level == "MEDIUM":
        return ["background-color: rgba(244, 162, 97, 0.14)"] * len(row)
    return [""] * len(row)


def _factor_waterfall(factors: list[dict]) -> None:
    if not factors:
        st.info("No factor payload available for this claim.")
        return

    chart_rows = []
    for factor in factors:
        name = str(factor.get("feature", "unknown"))
        importance = float(factor.get("importance", 0.0))
        direction = str(factor.get("direction", "high")).lower()
        signed = importance if direction == "high" else -importance
        chart_rows.append(
            {
                "factor": name,
                "impact": signed,
                "direction": "Increase Risk" if signed >= 0 else "Decrease Risk",
                "abs_impact": abs(signed),
            }
        )

    factor_df = pd.DataFrame(chart_rows).sort_values("impact", ascending=True)
    waterfall = (
        alt.Chart(factor_df)
        .mark_bar(size=20)
        .encode(
            x=alt.X("impact:Q", title="Signed Feature Impact"),
            y=alt.Y("factor:N", title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Increase Risk", "Decrease Risk"], range=["#E63946", "#2A9D8F"]),
            ),
            tooltip=["factor", "impact", "direction"],
        )
    )
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#94A3B8", strokeDash=[5, 5]).encode(x="x:Q")
    st.altair_chart(waterfall + rule, use_container_width=True)


def _render_claim_drawer(selected: pd.Series) -> None:
    tx_id = str(selected.get("transaction_id", "n/a"))
    risk_level = str(selected.get("risk_level", "LOW"))
    risk_score = float(selected.get("fraud_probability", 0.0))
    claim_amount = float(selected.get("claim_amount", 0.0))
    action = str(selected.get("recommended_action", RISK_ACTIONS.get(risk_level, "REVIEW")))
    provider_id = str(selected.get("provider_id", "n/a"))
    top_factor = str(selected.get("top_factor", "n/a"))

    st.markdown("#### Triage Drawer")
    st.caption("Single-claim context panel for decisioning.")
    st.markdown(
        f"<span class='risk-badge' style='background:{risk_color(risk_level)}'>{risk_emoji(risk_level)} {risk_level}</span>",
        unsafe_allow_html=True,
    )
    st.metric("Claim ID", tx_id)
    st.metric("Provider", provider_id)
    st.metric("Claim Amount", f"${claim_amount:,.2f}")
    st.metric("Recommended Action", action)
    st.metric("Top Driver", top_factor)
    st.metric("Fraud Gauge", f"{risk_score * 100:.1f}%")
    st.progress(min(max(risk_score, 0.0), 1.0))

    st.markdown("##### Waterfall-Style Impact")
    _factor_waterfall(selected.get("top_risk_factors") or [])


_init_state()

st.markdown("<div class='main-header'>🛡️ MedicaidGuard — Mission Control Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>High-volume claim triage with live KPI signals, interactive risk grid, and drill-down panel.</div>",
    unsafe_allow_html=True,
)

recovery_cols = st.columns([4, 1])
with recovery_cols[0]:
    st.caption(
        "If UI loading appears incomplete, click Reload App. "
        "If analysis fails, use Retry Last Analysis."
    )
with recovery_cols[1]:
    if st.button("Reload App", key="reload_medicaidguard_app", use_container_width=True):
        st.rerun()

status_tab, guide_tab = st.sidebar.tabs(["📊 Status", "ℹ️ Guide"])

with status_tab:
    st.header("📊 Status")
    api_connected, health = show_api_status(API_URL, headers=HEADERS)
    if health:
        st.caption(f"Model version: {health.get('model_version', 'unknown')}")
        st.caption(f"Model loaded: {health.get('model_loaded', False)}")

    if st.button("Refresh KPI Snapshot", key="refresh_kpis_btn", use_container_width=True):
        _refresh_kpis()

    if st.button("Reset Results", key="reset_results_btn", use_container_width=True):
        st.session_state.single_result = None
        st.session_state.batch_result = None
        st.session_state.last_single_error = ""
        st.session_state.triage_selected_id = None
        _set_single_run_state("Idle", "Ready to run claim analysis")
        st.rerun()

with guide_tab:
    st.markdown("### What This Project Is")
    st.markdown(
        "MedicaidGuard scores Medicaid claims for fraud risk and supports high-throughput analyst triage."
    )
    st.markdown("### Process")
    st.markdown(
        "1. Ingest single claim or batch claim records.\n"
        "2. Submit to model inference endpoints.\n"
        "3. Rank by fraud probability and risk level.\n"
        "4. Open triage drawer to inspect why each claim was scored.\n"
        "5. Export decision-ready results."
    )

st.markdown("### Live KPI Header")
triage_df = _build_triage_frame()
kpi_fallback = _local_kpi_fallback(triage_df)
if st.session_state.kpi_snapshot is None:
    _refresh_kpis()
kpi_snapshot = st.session_state.kpi_snapshot or {}
_render_kpi_header(kpi_snapshot, kpi_fallback)

left_col, right_col = st.columns([1.6, 1.0])
with left_col:
    st.markdown("### Claim Intake (Single Analysis)")
    st.selectbox(
        "Scenario",
        options=list(EXAMPLE_SCENARIOS.keys()),
        key="selected_scenario",
        on_change=_load_selected_scenario,
        help="Select a baseline scenario to prefill claim features.",
    )

    key_left, key_right = st.columns(2)
    with key_left:
        st.number_input(
            "Claim Amount",
            min_value=0.0,
            max_value=500000.0,
            step=50.0,
            key="single_claim_amount",
        )
        st.selectbox("Provider Type", options=["individual", "organization"], key="single_provider_type")
        st.slider("Claims (30d)", min_value=0, max_value=50, key="single_claim_frequency_30d")
    with key_right:
        st.slider("Billing Pattern Score", min_value=0.0, max_value=1.0, step=0.01, key="single_billing_pattern_score")
        st.number_input(
            "Avg Claim Amount (90d)",
            min_value=0.0,
            max_value=200000.0,
            step=50.0,
            key="single_avg_claim_amount_90d",
        )
        st.slider("Unique Patients (30d)", min_value=1, max_value=200, key="single_unique_patients_30d")

    with st.expander("Advanced Single-Claim Fields", expanded=False):
        adv1, adv2 = st.columns(2)
        with adv1:
            st.text_input("Transaction ID", key="single_transaction_id")
            st.text_input("Provider ID", key="single_provider_id")
            st.text_input("Procedure Code", key="single_procedure_code")
        with adv2:
            st.text_input("Diagnosis Code", key="single_diagnosis_code")
            st.slider("Patient Age", min_value=1, max_value=100, key="single_patient_age")

    _render_single_run_state()
    if st.session_state.single_run_state == "Error" and st.session_state.last_single_error:
        st.warning(f"Last run failed: {st.session_state.last_single_error}")

    run_cols = st.columns([1, 1, 2])
    with run_cols[0]:
        if st.button("Analyze Claim", key="run_single", use_container_width=True):
            _run_single_analysis()
    with run_cols[1]:
        if st.session_state.last_single_error and st.button(
            "Retry Last Analysis", key="retry_last_single", use_container_width=True
        ):
            _run_single_analysis()

with right_col:
    st.markdown("### Batch Intake")
    load_cols = st.columns(2)
    with load_cols[0]:
        if st.button("Use Sample Batch (100)", use_container_width=True):
            st.session_state.batch_df = pd.read_csv(SAMPLE_BATCH_PATH)
            st.session_state.batch_result = None
    with load_cols[1]:
        upload = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
        if upload is not None:
            st.session_state.batch_df = pd.read_csv(upload)
            st.session_state.batch_result = None

    if st.session_state.batch_df is not None:
        st.caption(f"Loaded rows: {len(st.session_state.batch_df)}")
        st.dataframe(st.session_state.batch_df.head(8), use_container_width=True, hide_index=True)
        if st.button("Analyze Batch", key="analyze_batch", use_container_width=True):
            _run_batch_analysis(st.session_state.batch_df)
    else:
        st.info("Load sample data or upload CSV to run bulk scoring.")

    if st.session_state.batch_result:
        with st.expander("Batch Distribution", expanded=True):
            _render_distribution(triage_df)

st.markdown("### Real-Time Triage Grid")
if triage_df.empty:
    st.info("No scored claims yet. Run single or batch analysis to populate the mission-control grid.")
else:
    triage_df = triage_df.copy()
    view_df = triage_df[
        [
            "transaction_id",
            "provider_id",
            "claim_amount",
            "fraud_probability",
            "risk_level",
            "prediction",
            "recommended_action",
            "top_factor",
        ]
    ].rename(
        columns={
            "transaction_id": "Claim ID",
            "provider_id": "Provider",
            "claim_amount": "Amount",
            "fraud_probability": "Fraud %",
            "risk_level": "Risk Level",
            "prediction": "Prediction",
            "recommended_action": "Action",
            "top_factor": "Top Factor",
        }
    )
    view_df["Fraud %"] = view_df["Fraud %"].apply(lambda x: f"{float(x) * 100:.1f}%")
    view_df["Amount"] = view_df["Amount"].apply(lambda x: f"${float(x):,.2f}")
    styled = view_df.style.apply(_risk_row_style, axis=1)

    grid_col, drawer_col = st.columns([2.2, 1.0])
    with grid_col:
        st.caption("Rows with HIGH/CRITICAL risk are highlighted for fast triage.")
        st.dataframe(styled, use_container_width=True, hide_index=True, height=390)
        options = triage_df["transaction_id"].astype(str).tolist()
        if st.session_state.triage_selected_id not in options:
            st.session_state.triage_selected_id = options[0]
        st.selectbox(
            "Open Claim In Drawer",
            options=options,
            key="triage_selected_id",
            help="Select a claim ID to inspect factors and actions in the drawer.",
        )

    with drawer_col:
        selected = triage_df[triage_df["transaction_id"].astype(str) == str(st.session_state.triage_selected_id)]
        if selected.empty:
            st.info("Select a claim from the triage grid.")
        else:
            _render_claim_drawer(selected.iloc[0])

    csv_data = triage_df.drop(columns=["top_risk_factors"], errors="ignore").to_csv(index=False)
    st.download_button(
        "📥 Download Triage CSV",
        data=csv_data,
        file_name="medicaidguard_triage_results.csv",
        mime="text/csv",
    )

with st.expander("Model + Workflow Reference", expanded=False):
    st.markdown("### Model Overview")
    st.write("- Model: XGBoost")
    st.write("- Primary metric: AUPRC 0.8379")
    st.write("- Dataset: 227M HHS Medicaid claims")
    st.write(f"- Feature count: {len(FEATURE_LIST)}")
    st.write("- Features: " + ", ".join(FEATURE_LIST))

    st.markdown("### Architecture (Mermaid)")
    st.code(
        """
flowchart LR
    A[Claim Input] --> B[Preprocessor]
    B --> C[XGBoost Model]
    C --> D[Fraud Probability]
    C --> E[Risk Factors]
    D --> F[Action Engine]
""".strip(),
        language="mermaid",
    )

show_footer()
