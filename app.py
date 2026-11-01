import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

# -----------------------------------
# LOAD ML MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Check random_forest_model.pkl")
    st.stop()

# -----------------------------------
# SESSION STATE
# -----------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------
# LIGHT THEME CSS
# -----------------------------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: #F3F8FF !important;
    color: #183153 !important;
}

/* Main content */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}

/* Headings */
h1, h2, h3, h4 {
    color: #12345A !important;
}

/* Normal text */
p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: #334B68 !important;
}

/* Header */
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #12345A !important;
}

.main-title span {
    color: #0797A5 !important;
}

.subtitle {
    font-size: 19px;
    color: #294967 !important;
    font-weight: 600;
}

.description {
    color: #526B85 !important;
    font-size: 16px;
}

/* Image cards */
.image-card {
    background: white;
    padding: 10px;
    border-radius: 16px;
    border: 1px solid #D5E5F5;
    box-shadow: 0 5px 15px rgba(30, 70, 110, 0.10);
}

.image-card img {
    width: 100%;
    height: 260px;
    object-fit: cover;
    border-radius: 12px;
}

.image-caption {
    text-align: center;
    color: #12345A !important;
    font-size: 16px;
    font-weight: 700;
    padding: 14px 5px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #D5E5F5;
    border-radius: 15px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #526B85 !important;
}

[data-testid="stMetricValue"] {
    color: #087F8C !important;
}

/* Input labels */
[data-testid="stNumberInput"] label {
    color: #12345A !important;
    font-weight: 600;
}

/* Input box */
[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: #12345A !important;
    -webkit-text-fill-color: #12345A !important;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button {
    background: #087F8C !important;
    color: #FFFFFF !important;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

/* Download button */
.stDownloadButton button {
    background: #12345A !important;
    color: #FFFFFF !important;
}

.stDownloadButton button p {
    color: #FFFFFF !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: white !important;
}

/* Plotly chart background */
[data-testid="stPlotlyChart"] {
    background: white !important;
    border-radius: 14px;
    padding: 10px;
}

/* Footer */
.footer {
    color: #526B85 !important;
    text-align: center;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# HEADER
# -----------------------------------
st.markdown("""
<div class="main-title">
    ⚙️ PREDICTIVE MAINTENANCE <span>AI</span>
</div>

<div class="subtitle">
    Intelligent Industrial Machine Monitoring System
</div>

<div class="description">
    AI-powered monitoring to identify potential machine failures
    before unexpected breakdowns.
</div>
""", unsafe_allow_html=True)


# -----------------------------------
# INDUSTRIAL IMAGES
# -----------------------------------

st.markdown('<div class="section-heading">Industrial Monitoring Overview</div>',
            unsafe_allow_html=True)

image1 = "https://images.unsplash.com/photo-1581092160562-40aa08e78837?auto=format&fit=crop&w=1000&q=85"

image2 = "https://www.tcihn.com/Imagenes/Proyecto7.jpg"

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(f"""
    <div class="image-card">
        <img src="{image1}">
        <div class="image-caption">
            ⚙️ Industrial Engineering and Machine Design
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="image-card">
        <img src="{image2}">
        <div class="image-caption">
            🔧 Industrial Machine Maintenance
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------
# MODEL OVERVIEW
# -----------------------------------

st.markdown('<div class="section-heading">Model Overview</div>',
            unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("ML Algorithm", "Random Forest")

with m2:
    st.metric("Test Accuracy", "98.15%")

with m3:
    st.metric("Sensor Inputs", "5 Features")


# -----------------------------------
# MACHINE SENSOR INPUTS
# -----------------------------------

st.markdown('<div class="section-heading">Machine Sensor Monitoring</div>',
            unsafe_allow_html=True)

st.write(
    "Enter the machine sensor readings below to check its failure condition."
)

with st.form("prediction_form"):

    left, right = st.columns(2)

    with left:
        air_temp = st.number_input(
            "Air Temperature (K)",
            min_value=250.0,
            max_value=350.0,
            value=300.0
        )

        process_temp = st.number_input(
            "Process Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=310.0
        )

        speed = st.number_input(
            "Rotational Speed (RPM)",
            min_value=0,
            max_value=5000,
            value=1500
        )

    with right:
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=150.0,
            value=40.0
        )

        tool_wear = st.number_input(
            "Tool Wear (Minutes)",
            min_value=0,
            max_value=300,
            value=100
        )

    submit = st.form_submit_button("Predict Machine Condition")


# -----------------------------------
# PREDICTION
# -----------------------------------

if submit:

    input_data = pd.DataFrame([[
        air_temp,
        process_temp,
        speed,
        torque,
        tool_wear
    ]], columns=[
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ])

    prediction = model.predict(input_data)[0]

    failure_score = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:

        st.error("⚠️ MACHINE FAILURE DETECTED")

        st.markdown("""
        <div class="result-card">
            <h3 style="color:#D97706;">
                Warning: Potential Machine Failure
            </h3>
            <p>
                The model has identified a possible machine failure.
                Maintenance inspection is recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)

        status = "Failure Detected"

    else:

        st.success("✅ MACHINE OPERATING NORMALLY")

        st.markdown("""
        <div class="result-card">
            <h3 style="color:#087F8C;">
                Machine Condition: Normal
            </h3>
            <p>
                The model has not detected a machine failure
                from the entered sensor readings.
            </p>
        </div>
        """, unsafe_allow_html=True)

        status = "No Failure"

    st.markdown("### Model-Estimated Failure Score")

    st.progress(int(failure_score))

    st.write(f"Estimated score: **{failure_score:.2f}%**")

    # Save prediction history
    st.session_state.history.append({
        "Date & Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "Air Temperature (K)": air_temp,
        "Process Temperature (K)": process_temp,
        "Rotational Speed (RPM)": speed,
        "Torque (Nm)": torque,
        "Tool Wear (min)": tool_wear,
        "Prediction": status,
        "Failure Score (%)": round(failure_score, 2)
    })


# -----------------------------------
# PREDICTION HISTORY
# -----------------------------------

st.markdown('<div class="section-heading">Prediction History</div>',
            unsafe_allow_html=True)

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # Chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_df["Date & Time"],
        y=history_df["Air Temperature (K)"],
        mode="lines+markers",
        name="Air Temperature"
    ))

    fig.add_trace(go.Scatter(
        x=history_df["Date & Time"],
        y=history_df["Rotational Speed (RPM)"],
        mode="lines+markers",
        name="Rotational Speed",
        yaxis="y2"
    ))
    fig.update_layout(
    title=dict(
        text="Machine Sensor Reading History",
        font=dict(color="#12345A", size=20)
    ),
    xaxis=dict(
        title=dict(text="Prediction Time", font=dict(color="#12345A", size=14)),
        tickfont=dict(color="#12345A", size=12),
        gridcolor="#E2E8F0"
    ),
    yaxis=dict(
        title=dict(text="Air Temperature (K)", font=dict(color="#12345A", size=14)),
        tickfont=dict(color="#12345A", size=12),
        gridcolor="#E2E8F0"
    ),
    yaxis2=dict(
        title=dict(text="Rotational Speed (RPM)", font=dict(color="#12345A", size=14)),
        tickfont=dict(color="#12345A", size=12),
        overlaying="y",
        side="right"
    ),
    legend=dict(
        font=dict(color="#12345A", size=12)
    ),
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#12345A", size=14)
)
  

    st.plotly_chart(fig, use_container_width=True)

    # Download history
    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction History",
        data=csv,
        file_name="machine_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear Prediction History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No predictions yet. Enter sensor values to start monitoring.")


# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("""
<hr>
<div class="footer">
    Predictive Maintenance AI |
    Machine Learning Based Failure Detection
    <br>
    Developed using Python, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)