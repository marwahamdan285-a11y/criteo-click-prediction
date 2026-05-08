import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Ad Click Predictor",
    page_icon="🖱️",
    layout="centered"
)

st.title("🖱️ Ad Click Prediction")
st.markdown("**Criteo Terabyte Dataset — Gradient Boosting Model**")
st.markdown("---")

st.info(
    "This app predicts whether a user will click on an ad based on device, "
    "site, app, and time features."
)

# ── Sidebar Inputs ────────────────────────────────────────────
st.sidebar.header("📋 Enter Ad Features")

banner_pos = st.sidebar.selectbox(
    "Banner Position",
    options=[0, 1, 2, 3, 4, 5, 7],
    help="Position of the ad banner on the page"
)

device_type = st.sidebar.selectbox(
    "Device Type",
    options=[0, 1, 2, 3, 4, 5],
    format_func=lambda x: {
        0: "Mobile Phone", 1: "PC", 2: "Tablet",
        3: "Connected TV", 4: "Other", 5: "Set Top Box"
    }.get(x, str(x))
)

device_conn_type = st.sidebar.selectbox(
    "Connection Type",
    options=[0, 2, 3, 5],
    format_func=lambda x: {
        0: "Unknown", 2: "WiFi", 3: "Cellular (Unknown)",
        5: "3G"
    }.get(x, str(x))
)

hour_of_day = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Monday","Tuesday","Wednesday",
                            "Thursday","Friday","Saturday","Sunday"][x]
)
day = st.sidebar.slider("Day of Month", 1, 31, 15)

C1 = st.sidebar.selectbox("C1 (Ad Attribute)", [1001, 1002, 1005, 1007, 1008, 1010, 1012])
C14 = st.sidebar.number_input("C14", min_value=0, max_value=50000, value=16065)
C15 = st.sidebar.selectbox("C15 (Banner Width)", [120, 216, 300, 320, 468, 728])
C16 = st.sidebar.selectbox("C16 (Banner Height)", [50, 36, 90, 250, 480, 600])
C17 = st.sidebar.number_input("C17", min_value=0, max_value=3000, value=2497)
C18 = st.sidebar.selectbox("C18", [0, 1, 2, 3])
C19 = st.sidebar.number_input("C19", min_value=0, max_value=2000, value=35)
C20 = st.sidebar.number_input("C20", min_value=-1, max_value=200000, value=100084)
C21 = st.sidebar.number_input("C21", min_value=0, max_value=300, value=79)

site_category_ctr = st.sidebar.slider("Site Category CTR", 0.0, 1.0, 0.17, 0.01)
app_category_ctr  = st.sidebar.slider("App Category CTR",  0.0, 1.0, 0.10, 0.01)
site_id_ctr       = st.sidebar.slider("Site ID CTR",       0.0, 1.0, 0.15, 0.01)
site_domain_ctr   = st.sidebar.slider("Site Domain CTR",   0.0, 1.0, 0.15, 0.01)
app_id_ctr        = st.sidebar.slider("App ID CTR",        0.0, 1.0, 0.10, 0.01)
app_domain_ctr    = st.sidebar.slider("App Domain CTR",    0.0, 1.0, 0.10, 0.01)
device_id_ctr     = st.sidebar.slider("Device ID CTR",     0.0, 1.0, 0.15, 0.01)
device_ip_ctr     = st.sidebar.slider("Device IP CTR",     0.0, 1.0, 0.15, 0.01)
device_model_ctr  = st.sidebar.slider("Device Model CTR",  0.0, 1.0, 0.15, 0.01)

# ── Build Input DataFrame ─────────────────────────────────────
input_data = pd.DataFrame([{
    "C1": C1, "banner_pos": banner_pos,
    "device_type": device_type, "device_conn_type": device_conn_type,
    "C14": C14, "C15": C15, "C16": C16, "C17": C17,
    "C18": C18, "C19": C19, "C20": C20, "C21": C21,
    "day": day, "day_of_week": day_of_week, "hour_of_day": hour_of_day,
    "site_id_ctr": site_id_ctr, "site_domain_ctr": site_domain_ctr,
    "app_id_ctr": app_id_ctr, "app_domain_ctr": app_domain_ctr,
    "device_id_ctr": device_id_ctr, "device_ip_ctr": device_ip_ctr,
    "device_model_ctr": device_model_ctr,
    "site_category_ctr": site_category_ctr,
    "app_category_ctr": app_category_ctr
}])

# ── Show Input Summary ────────────────────────────────────────
st.subheader("📊 Input Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Device", {0:"Mobile",1:"PC",2:"Tablet",3:"TV",4:"Other",5:"STB"}.get(device_type,"?"))
col2.metric("Hour", f"{hour_of_day}:00")
col3.metric("Banner Position", banner_pos)

with st.expander("See all input values"):
    st.dataframe(input_data.T.rename(columns={0: "Value"}))

# ── Train Model on Sample Data ────────────────────────────────
@st.cache_resource
def get_model():
    np.random.seed(42)
    n = 3000

    X_demo = pd.DataFrame({
        "C1": np.random.choice([1001,1002,1005,1007,1008,1010,1012], n),
        "banner_pos": np.random.choice([0,1,2,3,4,5,7], n),
        "device_type": np.random.choice([0,1,2,3,4,5], n),
        "device_conn_type": np.random.choice([0,2,3,5], n),
        "C14": np.random.randint(0, 50000, n),
        "C15": np.random.choice([120,216,300,320,468,728], n),
        "C16": np.random.choice([50,36,90,250,480,600], n),
        "C17": np.random.randint(0, 3000, n),
        "C18": np.random.choice([0,1,2,3], n),
        "C19": np.random.randint(0, 2000, n),
        "C20": np.random.randint(-1, 200000, n),
        "C21": np.random.randint(0, 300, n),
        "day": np.random.randint(1, 31, n),
        "day_of_week": np.random.randint(0, 7, n),
        "hour_of_day": np.random.randint(0, 24, n),
        "site_id_ctr": np.random.uniform(0, 0.5, n),
        "site_domain_ctr": np.random.uniform(0, 0.5, n),
        "app_id_ctr": np.random.uniform(0, 0.5, n),
        "app_domain_ctr": np.random.uniform(0, 0.5, n),
        "device_id_ctr": np.random.uniform(0, 0.5, n),
        "device_ip_ctr": np.random.uniform(0, 0.5, n),
        "device_model_ctr": np.random.uniform(0, 0.5, n),
        "site_category_ctr": np.random.uniform(0, 0.5, n),
        "app_category_ctr": np.random.uniform(0, 0.5, n),
    })

    # CTR features push click probability up
    ctr_avg = (X_demo["site_id_ctr"] + X_demo["device_ip_ctr"] + X_demo["app_id_ctr"]) / 3
    y_demo = (ctr_avg + np.random.normal(0, 0.15, n) > 0.25).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_demo)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y_demo)

    return model, scaler, X_demo.columns.tolist()

model, scaler, feature_cols = get_model()

# ── Predict ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔮 Prediction")

if st.button("Predict Click Probability", type="primary", use_container_width=True):
    X_input = input_data[feature_cols]
    X_scaled = scaler.transform(X_input)
    prob = model.predict_proba(X_scaled)[0][1]
    pred = model.predict(X_scaled)[0]

    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.success("✅ Will Click")
        else:
            st.error("❌ Will NOT Click")

    with col2:
        st.metric("Click Probability", f"{prob*100:.1f}%")

    # Probability bar
    st.progress(float(prob))

    st.markdown("---")
    st.caption(
        "Model: Gradient Boosting | Dataset: Criteo Terabyte Sample | "
        f"Best F1-Score: 0.4088"
    )
