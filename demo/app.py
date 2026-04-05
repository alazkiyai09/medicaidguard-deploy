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
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {
        "single_transaction_id": "TXN-DEMO-0001",
        "single_provider_id": "PRV-DEMO-1001",
        "single_claim_amount": 15000.0,
        "single_procedure_code": "99213",
        "single_diagnosis_code": "J06.9",
        "single_provider_type": "individual",
        "single_patient_age": 45,
        "single_claim_frequency_30d": 12,
        "single_avg_claim_amount_90d": 8500.0,
        "single_unique_patients_30d": 45,
        "single_billing_pattern_score": 0.73,
        "auto_run_single": False,
        "single_result": None,
        "batch_df": None,
        "batch_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


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
    st.session_state.auto_run_single = True


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


def _render_single_result(result: dict) -> None:
    probability = float(result.get("fraud_probability", 0.0))
    risk_level = str(result.get("risk_level", "LOW"))
    action = RISK_ACTIONS.get(risk_level, "REVIEW")

    st.markdown("### FRAUD PROBABILITY")
    st.progress(min(max(probability, 0.0), 1.0))
    st.metric("Score", f"{probability * 100:.1f}%")

    badge = (
        f"<span class='risk-badge' style='background:{risk_color(risk_level)}'>"
        f"{_format_risk_label(risk_level)}</span>"
    )
    st.markdown(badge, unsafe_allow_html=True)
    st.metric("Recommended Action", action)
    st.caption(
        f"Inference: {result.get('inference_time_ms', '-')} ms | "
        f"Model: {result.get('model_version', 'unknown')}"
    )

    factors = result.get("top_risk_factors") or []
    if factors:
        st.markdown("### TOP RISK FACTORS")
        factor_df = pd.DataFrame(
            {
                "feature": [f.get("feature", "unknown") for f in factors],
                "importance": [float(f.get("importance", 0.0)) for f in factors],
            }
        ).sort_values("importance", ascending=True)
        st.bar_chart(factor_df.set_index("feature"))


st.markdown("<div class='main-header'>🛡️ MedicaidGuard — AI Fraud Detection System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Real-time healthcare fraud detection powered by ML.</div>",
    unsafe_allow_html=True,
)

st.sidebar.header("📊 Status")
api_connected, health = show_api_status(API_URL, headers=HEADERS)
if health:
    st.sidebar.caption(f"Model version: {health.get('model_version', 'unknown')}")
    st.sidebar.caption(f"Model loaded: {health.get('model_loaded', False)}")

if st.sidebar.button("Reset Results", use_container_width=True):
    st.session_state.single_result = None
    st.session_state.batch_result = None
    st.rerun()

single_tab, batch_tab, model_tab = st.tabs(["🔍 Single Analysis", "📊 Batch Upload", "📈 Model Info"])

with single_tab:
    st.markdown("#### Quick Scenarios")
    scenario_cols = st.columns(3)
    for idx, scenario_name in enumerate(EXAMPLE_SCENARIOS):
        with scenario_cols[idx]:
            if st.button(scenario_name, key=f"scenario_{idx}", use_container_width=True):
                _scenario_to_state(scenario_name)
                st.rerun()

    left, right = st.columns([1, 1])

    with left:
        st.text_input("Transaction ID", key="single_transaction_id")
        st.text_input("Provider ID", key="single_provider_id")
        st.number_input("Claim Amount", min_value=0.0, max_value=500000.0, step=50.0, key="single_claim_amount")
        st.text_input("Procedure Code", key="single_procedure_code")
        st.text_input("Diagnosis Code", key="single_diagnosis_code")
        st.selectbox(
            "Provider Type",
            options=["individual", "organization"],
            key="single_provider_type",
        )
        st.slider("Patient Age", min_value=1, max_value=100, key="single_patient_age")
        st.slider("Claims (30d)", min_value=0, max_value=50, key="single_claim_frequency_30d")
        st.number_input(
            "Avg Claim Amount (90d)",
            min_value=0.0,
            max_value=200000.0,
            step=50.0,
            key="single_avg_claim_amount_90d",
        )
        st.slider("Unique Patients (30d)", min_value=1, max_value=200, key="single_unique_patients_30d")
        st.slider(
            "Billing Pattern Score",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="single_billing_pattern_score",
        )

        clicked = st.button("🔍 Analyze", key="run_single", use_container_width=True)
        auto_run = bool(st.session_state.auto_run_single)
        st.session_state.auto_run_single = False

        if clicked or auto_run:
            with st.spinner("Running model inference..."):
                response = api_call_with_retry(
                    f"{API_URL}/predict",
                    method="POST",
                    headers=HEADERS,
                    json_payload=_single_payload(),
                    timeout=90,
                    max_retries=2,
                )
            if "error" in response:
                st.error(response.get("detail", response["error"]))
                st.session_state.single_result = None
            else:
                st.session_state.single_result = response

    with right:
        if st.session_state.single_result:
            _render_single_result(st.session_state.single_result)
        else:
            st.info("Run a scenario or click Analyze to view prediction results.")

with batch_tab:
    st.markdown("#### Batch Analyzer")
    top_left, top_right = st.columns([1, 2])

    with top_left:
        if st.button("Use Sample Data (100 transactions)", use_container_width=True):
            st.session_state.batch_df = pd.read_csv(SAMPLE_BATCH_PATH)
            st.session_state.batch_result = None

    with top_right:
        upload = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
        if upload is not None:
            st.session_state.batch_df = pd.read_csv(upload)
            st.session_state.batch_result = None

    batch_df = st.session_state.batch_df
    if batch_df is not None:
        st.caption(f"Loaded rows: {len(batch_df)}")
        st.dataframe(batch_df.head(10), use_container_width=True)

        if st.button("Analyze Batch", use_container_width=True):
            missing = [column for column in REQUIRED_COLUMNS if column not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
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

                if "error" in response:
                    st.error(response.get("detail", response["error"]))
                    st.session_state.batch_result = None
                else:
                    st.session_state.batch_result = response

    if st.session_state.batch_result:
        result = st.session_state.batch_result
        predictions = pd.DataFrame(result.get("predictions", []))

        if not predictions.empty:
            distribution = (
                predictions["risk_level"]
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

            chart_col, summary_col = st.columns([1, 1])
            with chart_col:
                st.markdown("#### Risk Distribution")
                st.altair_chart(chart, use_container_width=True)
            with summary_col:
                total = int(result.get("batch_size", len(predictions)))
                fraud_count = int(result.get("fraud_count", int((predictions["prediction"] == "FRAUD").sum())))
                high_count = int(predictions["risk_level"].isin(["HIGH", "CRITICAL"]).sum())
                avg_score = float(predictions["fraud_probability"].mean())

                st.metric("Total", total)
                st.metric("Flagged", f"{fraud_count} ({(fraud_count / max(total, 1)) * 100:.1f}%)")
                st.metric("High Risk", f"{high_count} ({(high_count / max(total, 1)) * 100:.1f}%)")
                st.metric("Avg Score", f"{avg_score:.2f}")
                st.caption(f"Processing: {result.get('total_inference_time_ms', '-')} ms")

            enriched = predictions.copy()
            if st.session_state.batch_df is not None and "transaction_id" in st.session_state.batch_df.columns:
                amount_lookup = st.session_state.batch_df[["transaction_id", "claim_amount"]].copy()
                amount_lookup["transaction_id"] = amount_lookup["transaction_id"].astype(str)
                enriched = enriched.merge(amount_lookup, on="transaction_id", how="left")

            enriched["top_factor"] = enriched["top_risk_factors"].apply(
                lambda factors: factors[0]["feature"] if isinstance(factors, list) and factors else "n/a"
            )

            flagged = enriched[enriched["prediction"] == "FRAUD"].copy()
            flagged = flagged.sort_values("fraud_probability", ascending=False)

            st.markdown("#### Flagged Transactions")
            if flagged.empty:
                st.info("No transactions crossed the fraud threshold in this batch.")
            else:
                flagged_view = flagged[["transaction_id", "claim_amount", "fraud_probability", "risk_level", "top_factor"]]
                flagged_view = flagged_view.rename(
                    columns={
                        "transaction_id": "ID",
                        "claim_amount": "Amount",
                        "fraud_probability": "Risk",
                        "risk_level": "Level",
                        "top_factor": "Top Factor",
                    }
                )
                st.dataframe(flagged_view, use_container_width=True)

            csv_data = enriched.drop(columns=["top_risk_factors"], errors="ignore").to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                data=csv_data,
                file_name="medicaidguard_batch_results.csv",
                mime="text/csv",
            )

with model_tab:
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
