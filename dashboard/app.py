from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DUMMY_PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"
REAL_PREDICTIONS_PATH = BASE_DIR.parent / "data" / "ml" / "predictions.csv"
TRAINING_PATH = BASE_DIR.parent / "data" / "cleaned_data.csv"
METRICS_PATH = BASE_DIR.parent / "data" / "ml" / "model_metrics.txt"

HIGH_RISK_THRESHOLD = 0.60

COLORS = {
    "LOW": "#20C997",
    "HIGH": "#FF647C",
    "NAVY": "#0B1220",
    "PANEL": "#111C2E",
    "TEAL": "#45D6C5",
    "TEXT": "#E7EEF8",
    "MUTED": "#94A3B8",
}

ADMISSION_TYPES = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Mapped",
}

st.set_page_config(
    page_title="Readmission Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DARK EXECUTIVE THEME
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: #0B1220;
        color: #E7EEF8;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #0E1728;
        border-right: 1px solid #243249;
    }

    [data-testid="stSidebar"] * {
        color: #E7EEF8;
    }

    h1, h2, h3 {
        color: #F4F7FB !important;
        letter-spacing: -0.4px;
    }

    p, label, .stCaption {
        color: #A9B7CB;
    }

    [data-testid="stMetric"] {
        background: #111C2E;
        border: 1px solid #243249;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 125px;
    }

    [data-testid="stMetricLabel"] {
        color: #A9B7CB !important;
    }

    [data-testid="stMetricValue"] {
        color: #F4F7FB !important;
    }

    [data-testid="stMetricDelta"] {
        color: #45D6C5 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #243249;
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stPlotlyChart"] {
        background: #111C2E;
        border: 1px solid #243249;
        border-radius: 14px;
        padding: 8px;
    }

    div.stButton > button {
        background: #173C49;
        color: #E7EEF8;
        border: 1px solid #287D80;
        border-radius: 9px;
    }

    div.stButton > button:hover {
        background: #205462;
        color: white;
        border-color: #45D6C5;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #243249;
    }

    .hero {
        background: linear-gradient(115deg, #13243A 0%, #123746 100%);
        border: 1px solid #285263;
        border-radius: 18px;
        padding: 26px 28px;
        margin-bottom: 22px;
    }

    .hero-eyebrow {
        color: #45D6C5;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #F4F7FB;
        font-size: 30px;
        font-weight: 750;
        margin: 8px 0 6px 0;
    }

    .hero-subtitle {
        color: #B5C6D9;
        font-size: 14px;
        margin: 0;
    }

    .section-kicker {
        color: #45D6C5;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .info-panel {
        background: #111C2E;
        border: 1px solid #243249;
        border-radius: 14px;
        padding: 18px;
        min-height: 110px;
    }

    .info-title {
        color: #F4F7FB;
        font-size: 15px;
        font-weight: 650;
        margin-bottom: 7px;
    }

    .info-body {
        color: #A9B7CB;
        font-size: 13px;
        line-height: 1.6;
    }

    .risk-pill-high {
        color: #FF8798;
        font-weight: 700;
    }

    .risk-pill-low {
        color: #45D6C5;
        font-weight: 700;
    }

    .footer-note {
        color: #8191A8;
        font-size: 11px;
        padding-top: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING AND NORMALIZATION
# ============================================================
@st.cache_data
def load_csv(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string)


def normalize_predictions(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize dummy and real predictions to a shared schema."""
    out = raw.copy()

    if "patient_id" not in out.columns:
        raise ValueError("Prediction data must contain a patient_id column.")

    if "risk_score" not in out.columns:
        if "readmission_probability" in out.columns:
            out["risk_score"] = out["readmission_probability"]
        else:
            raise ValueError(
                "Predictions must contain risk_score or "
                "readmission_probability."
            )

    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce")
    out = out.dropna(subset=["patient_id", "risk_score"]).copy()
    out = out[out["risk_score"].between(0, 1)].copy()

    # Use one consistent risk definition for both sources.
    out["risk_level"] = np.where(
        out["risk_score"] > HIGH_RISK_THRESHOLD,
        "HIGH",
        "LOW",
    )

    # Optional columns are retained when present.
    optional_columns = [
        "actual_readmitted_30_days",
        "predicted_readmitted_30_days",
    ]
    keep_columns = [
        "patient_id",
        "risk_score",
        "risk_level",
    ] + [column for column in optional_columns if column in out.columns]

    return out[keep_columns].reset_index(drop=True)


@st.cache_data
def load_predictions(path_string: str) -> pd.DataFrame:
    return normalize_predictions(load_csv(path_string))


@st.cache_data
def load_training_data(path_string: str) -> pd.DataFrame:
    return pd.read_csv(
        path_string,
        usecols=["age", "admission_type_id", "readmitted_30_days"],
    )


def chart_layout(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["TEXT"], family="Arial"),
        margin=dict(l=18, r=18, t=45, b=18),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["TEXT"]),
        ),
        title_font=dict(size=16, color=COLORS["TEXT"]),
        xaxis=dict(
            gridcolor="#243249",
            zerolinecolor="#243249",
            linecolor="#243249",
        ),
        yaxis=dict(
            gridcolor="#243249",
            zerolinecolor="#243249",
            linecolor="#243249",
        ),
    )
    return fig


def section_heading(kicker, heading, caption=None):
    st.markdown(
        f'<div class="section-kicker">{kicker}</div><h2>{heading}</h2>',
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


# ============================================================
# SIDEBAR AND SOURCE SELECTION
# ============================================================
st.sidebar.markdown("## 🏥 Readmission AI")
st.sidebar.caption("Clinical risk intelligence")
st.sidebar.divider()

source_options = ["Real predictions (ML)", "Dummy data"]

source = st.sidebar.radio(
    "Prediction source",
    source_options,
    index=0 if REAL_PREDICTIONS_PATH.exists() else 1,
)

predictions_path = (
    REAL_PREDICTIONS_PATH
    if source == "Real predictions (ML)"
    else DUMMY_PREDICTIONS_PATH
)

if not predictions_path.exists():
    st.error(f"Prediction file not found: {predictions_path}")
    st.info("Check that the prediction CSV exists at the path shown above.")
    st.stop()

try:
    df = load_predictions(str(predictions_path))
except Exception as exc:
    st.error(f"Could not load prediction data: {exc}")
    st.stop()

if df.empty:
    st.warning("The selected prediction file contains no valid prediction rows.")
    st.stop()

training_df = None
if TRAINING_PATH.exists():
    try:
        training_df = load_training_data(str(TRAINING_PATH))
    except Exception as exc:
        st.sidebar.warning(f"Training summaries unavailable: {exc}")

st.sidebar.markdown("### Risk configuration")
st.sidebar.metric("High-risk threshold", f"{HIGH_RISK_THRESHOLD:.0%}")
st.sidebar.caption("High risk means a score strictly greater than 60%.")

st.sidebar.divider()
st.sidebar.caption("For decision support only. Not a clinical diagnosis.")


# ============================================================
# HERO HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Clinical operations · Decision support</div>
        <div class="hero-title">Readmission Intelligence</div>
        <p class="hero-subtitle">
            A clear view of predicted 30-day readmission risk, patient
            prioritization, and model evaluation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Data source: {source}  ·  "
    f"Risk threshold: > {HIGH_RISK_THRESHOLD:.0%}  ·  "
    "Scores represent model-predicted probabilities."
)

# ============================================================
# INTERACTIVE NAVIGATION
# ============================================================
page = st.radio(
    "Dashboard section",
    ["Overview", "Risk", "Filters", "Model", "Explain"],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    section_heading(
        "Executive summary",
        "Patient risk at a glance",
        "High-level indicators for the currently selected prediction source.",
    )

    total_patients = len(df)
    high_risk_count = int((df["risk_level"] == "HIGH").sum())
    low_risk_count = int((df["risk_level"] == "LOW").sum())
    average_risk = float(df["risk_score"].mean())
    high_risk_pct = high_risk_count / total_patients * 100 if total_patients else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients scored", f"{total_patients:,}")
    c2.metric("High-risk patients", f"{high_risk_count:,}")
    c3.metric("Low-risk patients", f"{low_risk_count:,}")
    c4.metric("Average risk score", f"{average_risk:.1%}")

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        section_heading("Risk profile", "Predicted risk distribution")

        dist_fig = px.histogram(
            df,
            x="risk_score",
            nbins=20,
            color="risk_level",
            color_discrete_map=COLORS,
            category_orders={"risk_level": ["LOW", "HIGH"]},
            labels={
                "risk_score": "Predicted probability",
                "count": "Patients",
                "risk_level": "Risk category",
            },
        )
        dist_fig.add_vline(
            x=HIGH_RISK_THRESHOLD,
            line_dash="dash",
            line_color=COLORS["TEAL"],
            annotation_text="60% threshold",
            annotation_position="top right",
        )
        dist_fig.update_layout(bargap=0.08, yaxis_title="Patients")
        st.plotly_chart(chart_layout(dist_fig), width="stretch")

    with right:
        section_heading("Patient mix", "Risk category split")

        risk_counts = (
            df["risk_level"]
            .value_counts()
            .reindex(["LOW", "HIGH"], fill_value=0)
            .rename_axis("Risk level")
            .reset_index(name="Patients")
        )

        pie_fig = px.pie(
            risk_counts,
            names="Risk level",
            values="Patients",
            hole=0.68,
            color="Risk level",
            color_discrete_map=COLORS,
        )
        pie_fig.update_traces(
            textinfo="percent",
            textfont_size=14,
            marker=dict(line=dict(color=COLORS["NAVY"], width=3)),
        )
        pie_fig.update_layout(showlegend=True)
        st.plotly_chart(chart_layout(pie_fig), width="stretch")

    st.write("")
    section_heading("Action queue", "Highest predicted risk")

    top_patients = (
        df.sort_values("risk_score", ascending=False)
        .head(8)
        .copy()
    )
    top_patients["Risk score"] = top_patients["risk_score"].map(
        lambda value: f"{value:.1%}"
    )
    top_patients["Risk level"] = top_patients["risk_level"]
    top_patients["Suggested workflow"] = top_patients["risk_level"].map(
        {"HIGH": "Review for follow-up", "LOW": "Routine workflow"}
    )

    st.dataframe(
        top_patients[
            ["patient_id", "Risk score", "Risk level", "Suggested workflow"]
        ].rename(columns={"patient_id": "Patient ID"}),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        f"{high_risk_pct:.1f}% of the selected patients are above the "
        "configured risk threshold. This is a prioritization signal, "
        "not a confirmed readmission outcome."
    )


# ============================================================
# RISK
# ============================================================
elif page == "Risk":
    section_heading(
        "Risk monitoring",
        "High-risk review queue",
        "Prioritize review using predicted probability. Clinical staff should "
        "confirm appropriate actions independently.",
    )

    high_df = (
        df[df["risk_level"] == "HIGH"]
        .sort_values("risk_score", ascending=False)
        .copy()
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("High-risk queue", f"{len(high_df):,}")
    c2.metric(
        "Highest score",
        f"{df['risk_score'].max():.1%}" if len(df) else "N/A",
    )
    c3.metric(
        "Threshold",
        f">{HIGH_RISK_THRESHOLD:.0%}",
    )

    if high_df.empty:
        st.success("No patients in this source exceed the configured threshold.")
    else:
        high_df["Action"] = "Review for follow-up"
        high_df["Risk score"] = high_df["risk_score"].map(
            lambda value: f"{value:.1%}"
        )

        st.dataframe(
            high_df[
                ["patient_id", "Risk score", "Action"]
            ].rename(columns={"patient_id": "Patient ID"}),
            width="stretch",
            hide_index=True,
        )

        csv_bytes = high_df[
            ["patient_id", "risk_score", "risk_level"]
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Export high-risk queue (CSV)",
            data=csv_bytes,
            file_name="high_risk_review_queue.csv",
            mime="text/csv",
        )

    st.write("")
    section_heading("Distribution", "Risk score bands")

    bins = [-0.001, 0.20, 0.40, 0.60, 0.80, 1.001]
    labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

    band_data = df.copy()
    band_data["Risk band"] = pd.cut(
        band_data["risk_score"],
        bins=bins,
        labels=labels,
        right=True,
    )

    band_counts = (
        band_data["Risk band"]
        .value_counts(sort=False)
        .rename_axis("Risk band")
        .reset_index(name="Patients")
    )

    band_fig = px.bar(
        band_counts,
        x="Risk band",
        y="Patients",
        labels={"Risk band": "Predicted risk band"},
        text="Patients",
    )
    band_fig.update_traces(marker_color=COLORS["TEAL"], textposition="outside")
    st.plotly_chart(chart_layout(band_fig), width="stretch")


# ============================================================
# FILTERS
# ============================================================
elif page == "Filters":
    section_heading(
        "Patient explorer",
        "Search and filter predictions",
        "Use these controls to explore the currently selected source.",
    )

    f1, f2 = st.columns([1.2, 1])

    with f1:
        search_text = st.text_input(
            "Search patient ID",
            placeholder="Enter a patient ID or part of an ID",
        )

    with f2:
        risk_filter = st.multiselect(
            "Risk category",
            options=["HIGH", "LOW"],
            default=["HIGH", "LOW"],
        )

    score_range = st.slider(
        "Predicted risk-score range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
        format="%.2f",
    )

    filtered_df = df[
        df["risk_level"].isin(risk_filter)
        & df["risk_score"].between(score_range[0], score_range[1])
    ].copy()

    if search_text.strip():
        filtered_df = filtered_df[
            filtered_df["patient_id"]
            .astype(str)
            .str.contains(search_text.strip(), case=False, na=False)
        ]

    filtered_df = filtered_df.sort_values("risk_score", ascending=False)

    a, b, c = st.columns(3)
    a.metric("Matching patients", f"{len(filtered_df):,}")
    b.metric(
        "High-risk matches",
        f"{int((filtered_df['risk_level'] == 'HIGH').sum()):,}",
    )
    c.metric(
        "Mean score in results",
        f"{filtered_df['risk_score'].mean():.1%}"
        if not filtered_df.empty
        else "N/A",
    )

    display_df = filtered_df.copy()
    display_df["Risk score"] = display_df["risk_score"].map(
        lambda value: f"{value:.1%}"
    )

    rename_map = {
        "patient_id": "Patient ID",
        "risk_level": "Risk level",
    }

    visible_columns = ["patient_id", "Risk score", "risk_level"]
    for optional_column in [
        "actual_readmitted_30_days",
        "predicted_readmitted_30_days",
    ]:
        if optional_column in display_df.columns:
            visible_columns.append(optional_column)

    st.dataframe(
        display_df[visible_columns].rename(columns=rename_map),
        width="stretch",
        hide_index=True,
    )

    export_df = filtered_df.copy()
    st.download_button(
        "Export filtered results (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_readmission_predictions.csv",
        mime="text/csv",
    )


# ============================================================
# MODEL
# ============================================================
elif page == "Model":
    section_heading(
        "Model transparency",
        "Logistic Regression performance",
        "Evaluation results loaded from your project's model_metrics.txt file "
        "when available.",
    )

    # Values are the actual metrics you supplied from model_metrics.txt.
    # If the file exists, parse the current saved metrics to avoid stale values.
    metric_values = {
        "Accuracy": 0.6423,
        "Precision": 0.1660,
        "Recall": 0.5482,
        "F1 Score": 0.2548,
        "ROC-AUC": 0.6441,
    }

    if METRICS_PATH.exists():
        try:
            metrics_text = METRICS_PATH.read_text(encoding="utf-8")
            for metric_name in metric_values:
                for line in metrics_text.splitlines():
                    if line.strip().startswith(metric_name):
                        value = line.split(":", 1)[1].strip()
                        metric_values[metric_name] = float(value)
                        break
        except (ValueError, IndexError, OSError):
            st.warning(
                "Could not parse the metrics file; showing the saved "
                "evaluation values from this dashboard configuration."
            )
    else:
        st.warning(
            "Metrics file not found. Showing the last confirmed evaluation "
            "results from your project."
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metric_values['Accuracy']:.2%}")
    m2.metric("Precision", f"{metric_values['Precision']:.2%}")
    m3.metric("Recall", f"{metric_values['Recall']:.2%}")

    m4, m5 = st.columns(2)
    m4.metric("F1 score", f"{metric_values['F1 Score']:.2%}")
    m5.metric("ROC-AUC", f"{metric_values['ROC-AUC']:.2%}")

    st.write("")
    left, right = st.columns([1.1, 1])

    with left:
        section_heading("Evaluation", "Metric comparison")

        metric_chart_df = pd.DataFrame(
            {
                "Metric": list(metric_values.keys()),
                "Score": list(metric_values.values()),
            }
        )

        metric_fig = px.bar(
            metric_chart_df,
            x="Metric",
            y="Score",
            text=metric_chart_df["Score"].map(lambda value: f"{value:.1%}"),
            range_y=[0, 1],
        )
        metric_fig.update_traces(
            marker_color=COLORS["TEAL"],
            textposition="outside",
        )
        st.plotly_chart(chart_layout(metric_fig), width="stretch")

    with right:
        section_heading("Classification", "Confusion matrix")

        # Actual confusion matrix from the saved evaluation output:
        # rows = actual class (0, 1); columns = predicted class (0, 1).
        confusion = np.array([[11828, 6255], [1026, 1245]])

        cm_fig = px.imshow(
            confusion,
            text_auto=True,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            color_continuous_scale=[
                [0, "#142238"],
                [0.5, "#1B6B75"],
                [1, "#45D6C5"],
            ],
            aspect="auto",
            labels={"x": "", "y": "", "color": "Patients"},
        )
        cm_fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(chart_layout(cm_fig), width="stretch")

    st.markdown(
        """
        <div class="info-panel">
            <div class="info-title">How to interpret these results</div>
            <div class="info-body">
                <b>Precision</b> indicates how many patients flagged by the
                model were positive in the evaluation set.
                <b>Recall</b> indicates how many actual positive cases the
                model identified. ROC-AUC measures ranking discrimination
                across thresholds. These metrics are not a substitute for
                external validation or clinical review.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Evaluation set: 20,354 patients. Positive class: readmitted within "
        "30 days. Metrics describe the recorded evaluation, not guaranteed "
        "future performance."
    )


# ============================================================
# EXPLAIN
# ============================================================
elif page == "Explain":
    section_heading(
        "Explainability",
        "Understand a patient's risk score",
        "This view explains what the score means without inventing individual "
        "causes that are not present in the prediction data.",
    )

    patient_ids = df["patient_id"].astype(str).tolist()

    selected_id = st.selectbox(
        "Select a patient",
        options=patient_ids,
        index=0,
    )

    selected_rows = df[df["patient_id"].astype(str) == selected_id]
    patient = selected_rows.iloc[0]
    score = float(patient["risk_score"])
    level = patient["risk_level"]

    p1, p2, p3 = st.columns(3)
    p1.metric("Patient ID", str(patient["patient_id"]))
    p2.metric("Predicted risk", f"{score:.1%}")
    p3.metric("Risk category", level)

    st.write("")
    st.markdown("### Risk score")

    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"color": COLORS["TEXT"], "size": 34}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": COLORS["TEXT"],
                    "tickfont": {"color": COLORS["TEXT"]},
                },
                "bar": {
                    "color": COLORS["HIGH"] if level == "HIGH" else COLORS["TEAL"]
                },
                "bgcolor": COLORS["PANEL"],
                "bordercolor": "#243249",
                "steps": [
                    {"range": [0, 20], "color": "#172A37"},
                    {"range": [20, 40], "color": "#1B3946"},
                    {"range": [40, 60], "color": "#244652"},
                    {"range": [60, 80], "color": "#4A3547"},
                    {"range": [80, 100], "color": "#5B2C40"},
                ],
                "threshold": {
                    "line": {"color": COLORS["TEAL"], "width": 3},
                    "thickness": 0.8,
                    "value": HIGH_RISK_THRESHOLD * 100,
                },
            },
            title={"text": "Predicted probability", "font": {"color": COLORS["TEXT"]}},
        )
    )
    gauge_fig.update_layout(
        height=290,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["TEXT"]},
        margin=dict(l=20, r=20, t=45, b=15),
    )
    st.plotly_chart(gauge_fig, width="stretch")

    if level == "HIGH":
        st.warning(
            "This score is above the configured threshold. Consider reviewing "
            "the patient through your established clinical workflow. The score "
            "alone does not establish a diagnosis or dictate care."
        )
    else:
        st.success(
            "This score is at or below the configured high-risk threshold. "
            "A lower score does not mean readmission is impossible; follow "
            "usual clinical assessment and protocols."
        )

    st.markdown("### What can be concluded?")
    st.markdown(
        f"""
        <div class="info-panel">
            <div class="info-title">Score interpretation</div>
            <div class="info-body">
                The model assigned this record a predicted probability of
                <b>{score:.1%}</b>. The configured high-risk rule is a score
                strictly greater than <b>{HIGH_RISK_THRESHOLD:.0%}</b>, so
                this record is categorized as <b>{level}</b>.<br><br>
                This is a model output, not a confirmed outcome or a causal
                explanation. The available prediction file does not contain
                enough information to identify the specific clinical factors
                responsible for this patient's score.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "actual_readmitted_30_days" in patient.index:
        actual_value = patient["actual_readmitted_30_days"]
        if pd.notna(actual_value):
            st.markdown("### Recorded evaluation outcome")
            st.caption(
                "This field is available in the real evaluation predictions. "
                "It is the recorded outcome, not something known at the time "
                "of a future prediction."
            )
            st.metric(
                "Actual 30-day outcome",
                "Readmitted" if int(actual_value) == 1 else "Not readmitted",
            )

    st.markdown("### Risk bands")
    band_cols = st.columns(5)
    bands = [
        ("0–20%", "Lower score"),
        ("20–40%", "Moderate-low"),
        ("40–60%", "Moderate"),
        ("60–80%", "Higher score"),
        ("80–100%", "Very high score"),
    ]

    for column, (band, description) in zip(band_cols, bands):
        with column:
            st.markdown(
                f"""
                <div class="info-panel">
                    <div class="info-title">{band}</div>
                    <div class="info-body">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    """
    <div class="footer-note">
        READMISSION INTELLIGENCE · Decision support only ·
        Predictions should be reviewed alongside clinical judgment.
    </div>
    """,
    unsafe_allow_html=True,
)