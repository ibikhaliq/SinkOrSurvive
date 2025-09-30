import streamlit as st
import requests


st.set_page_config(page_title="Titanic Survival", page_icon="🚢", layout="centered")
st.title("🚢 Titanic Survival Predictor (API-backed)")

API_URL = "http://127.0.0.1:8000/predict"

pclass   = st.sidebar.selectbox("Ticket Class", [1, 2, 3], index=2)
sex      = st.sidebar.radio("Sex", ["male", "female"])
age      = st.sidebar.slider("Age", 0, 80, 25)
sibsp    = st.sidebar.slider("Siblings/Spouses", 0, 8, 0)
parch    = st.sidebar.slider("Parents/Children", 0, 6, 0)
fare     = st.sidebar.slider("Fare", 0.0, 520.0, 32.0)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])
has_cabin= st.sidebar.checkbox("Has Cabin?", value=False)

payload = {
    "Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp, "Parch": parch,
    "Fare": fare, "Embarked": embarked, "HasCabin": int(has_cabin)
}

if st.button("Predict"):
    r = requests.post(API_URL, json=payload, timeout=10)
    if r.ok:
        data = r.json()
        pred, proba = data["prediction"], data["survival_probability"]
        if pred == 1:
            st.success(f"✅ Survived (probability {proba:.1%})")
        else:
            st.error(f"❌ Did not survive (survival probability {proba:.1%})")
    else:
        st.error(f"API error {r.status_code}: {r.text}")