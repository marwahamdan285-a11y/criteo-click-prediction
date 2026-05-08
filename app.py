import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Ad Click Predictor",
    page_icon="🖱️",
    layout="centered"
)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('voting_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    return model, scaler, selector

model, scaler, selector = load_model()

# ── Header ────────────────────────────────────────────────────
st.title("🖱️ Ad Click Prediction")
st.markdown("**Criteo Terabyte Dataset — Soft Voting Ensemble (7 Models)**")
st.markdown("---")
st.info("Predicts whether a user will click on an ad using a Soft Voting Ensemble of 7 models.")

# ── Sidebar Inputs ────────────────────────────────────────────
st.sidebar.header("📋 Enter Ad Features")

banner_pos = st.sidebar.selectbox("Banner Position", options=[0,1,2,3,4,5,7])

device_type = st.sidebar.selectbox(
    "Device Type", options=[0,1,2,3,4,5],
    format_func=lambda x: {
        0:"Mobile Phone", 1:"PC", 2:"Tablet",
        3:"Connected TV", 4:"Other", 5:"Set Top Box"
    }[x]
)

device_conn_type = st.sidebar.selectbox(
    "Connection Type", options=[0,2,3,5],
    format_func=lambda x: {0:"Unknown", 2:"WiFi", 3:"Cellular", 5:"3G"}[x]
)

hour_of_day = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_of_week = st.sidebar.selectbox(
    "Day of Week", options=list(range(7)),
    format_func=lambda x: ["Monday","Tuesday","Wednesday",
                            "Thursday","Friday","Saturday","Sunday"][x]
)
day = st.sidebar.slider("Day of Month", 1, 31, 15)

C1  = st.sidebar.selectbox("C1",  [1001,1002,1005,1007,1008,1010,1012])
C14 = st.sidebar.number_input("C14", min_value=0,  max_value=50000,  value=16065)
C15 = st.sidebar.selectbox("C15 (Banner Width)",  [120,216,300,320,468,728])
C16 = st.sidebar.selectbox("C16 (Banner Height)", [50,36,90,250,480,600])
C17 = st.sidebar.number_input("C17", min_value=0,  max_value=3000,   value=2497)
C18 = st.sidebar.selectbox("C18", [0,1,2,3])
C19 = st.sidebar.number_input("C19", min_value=0,  max_value=2000,   value=35)
C20 = st.sidebar.number_input("C20", min_value=-1, max_value=200000, value=100084)
C21 = st.sidebar.number_input("C21", min_value=0,  max_value=300,    value=79)

st.sidebar.markdown("---")
st.sidebar.markdown("**Historical Click Rates (CTR)**")
site_id_ctr       = st.sidebar.slider("Site ID CTR",       0.0, 1.0, 0.15, 0.01)
site_domain_ctr   = st.sidebar.slider("Site Domain CTR",   0.0, 1.0, 0.15, 0.01)
site_category_ctr = st.sidebar.slider("Site Category CTR", 0.0, 1.0, 0.17, 0.01)
app_id_ctr        = st.sidebar.slider("App ID CTR",        0.0, 1.0, 0.10, 0.01)
app_domain_ctr    = st.sidebar.slider("App Domain CTR",    0.0, 1.0, 0.10, 0.01)
app_category_ctr  = st.sidebar.slider("App Category CTR",  0.0, 1.0, 0.10, 0.01)
device_id_ctr     = st.sidebar.slider("Device ID CTR",     0.0, 1.0, 0.15, 0.01)
device_ip_ctr     = st.sidebar.slider("Device IP CTR",     0.0, 1.0, 0.15, 0.01)
device_model_ctr  = st.sidebar.slider("Device Model CTR",  0.0, 1.0, 0.15, 0.01)

# ── Build Input DataFrame ─────────────────────────────────────
input_df = pd.DataFrame([{
    "C1": C1, "banner_pos": banner_pos,
    "device_type": device_type, "device_conn_type": device_conn_type,
    "C14": C14, "C15": C15, "C16": C16, "C17": C17,
    "C18": C18, "C19": C19, "C20": C20, "C21": C21,
    "day": day, "day_of_week": day_of_week, "hour_of_day": hour_of_day,
    "site_id_ctr": site_id_ctr, "site_domain_ctr": site_domain_ctr,
    "site_category_ctr": site_category_ctr,
    "app_id_ctr": app_id_ctr, "app_domain_ctr": app_domain_ctr,
    "app_category_ctr": app_category_ctr,
    "device_id_ctr": device_id_ctr, "device_ip_ctr": device_ip_ctr,
    "device_model_ctr": device_model_ctr
}])

# ── Input Summary ─────────────────────────────────────────────
st.subheader("📊 Input Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Device", {0:"Mobile",1:"PC",2:"Tablet",3:"TV",4:"Other",5:"STB"}[device_type])
c2.metric("Hour",   f"{hour_of_day}:00")
c3.metric("Banner", f"Position {banner_pos}")

with st.expander("See all input values"):
    st.dataframe(input_df.T.rename(columns={0: "Value"}))

# ── Predict ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔮 Prediction")

if st.button("Predict Click Probability", type="primary", use_container_width=True):
    try:
        X_selected = selector.transform(input_df)
        X_scaled   = scaler.transform(X_selected)
        prob       = model.predict_proba(X_scaled)[0][1]
        pred       = model.predict(X_scaled)[0]

        col1, col2 = st.columns(2)
        with col1:
            if pred == 1:
                st.success("✅ Will Click")
            else:
                st.error("❌ Will NOT Click")
        with col2:
            st.metric("Click Probability", f"{prob*100:.1f}%")

        st.progress(float(prob))

        if prob >= 0.6:
            st.balloons()
            st.success("High chance of click! 🎯")
        elif prob >= 0.4:
            st.warning("Moderate chance of click.")
        else:
            st.info("Low chance of click.")

        # Show individual model contributions
        st.markdown("---")
        st.subheader("📈 Individual Model Probabilities")
        model_names = ["Logistic Regression","KNN","Random Forest",
                       "Gradient Boosting","XGBoost","LightGBM","CatBoost"]
        probs = []
        for est_name, est in model.estimators_:
            p = est.predict_proba(X_scaled)[0][1]
            probs.append(p)

        prob_df = pd.DataFrame({
            "Model": model_names,
            "Click Probability": [round(p*100,1) for p in probs]
        }).sort_values("Click Probability", ascending=False)

        st.dataframe(prob_df, use_container_width=True)

        st.markdown("---")
        st.caption(
            "Model: Soft Voting Ensemble (7 models) | "
            "Dataset: Criteo Terabyte Sample"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Make sure voting_model.pkl, scaler.pkl, and selector.pkl are in the GitHub repo.")
