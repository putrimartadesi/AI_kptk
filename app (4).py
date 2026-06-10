import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
try:
    model = joblib.load("mental_health_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

st.set_page_config(
    page_title="Prediksi Kesehatan Mental Mahasiswa",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Prediksi Kesehatan Mental Mahasiswa")

# Input
st.sidebar.header("Input Data")

gpa = st.sidebar.slider("GPA", 1.0, 4.0, 3.0)

stress = st.sidebar.slider("Stress Level", 0, 10, 5)

anxiety = st.sidebar.slider("Anxiety Score", 0, 10, 5)

depression = st.sidebar.slider("Depression Score", 0, 10, 5)

mood = st.sidebar.selectbox(
    "Mood Description",
    [
        "Happy",
        "Calm",
        "Neutral",
        "Sad",
        "Anxious",
        "Stressed",
        "Motivated",
        "Frustrated"
    ]
)

daily_reflections = st.sidebar.text_area(
    "Daily Reflections",
    "I feel good today"
)

# Encoding Mood_Description
mood_map = {
    "Happy": 0,
    "Calm": 1,
    "Neutral": 2,
    "Sad": 3,
    "Anxious": 4,
    "Stressed": 5,
    "Motivated": 6,
    "Frustrated": 7
}

input_data = pd.DataFrame({
    "GPA": [gpa],
    "Stress_Level": [stress],
    "Anxiety_Score": [anxiety],
    "Depression_Score": [depression],
    "Mood_Description": [mood_map[mood]],
    "Daily_Reflections": [daily_reflections]
})

st.subheader("Data Input")
st.dataframe(input_data)

if st.button("Prediksi"):

    try:
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]

        if prediction == 0:
            st.success("Healthy 😊")

        elif prediction == 1:
            st.warning("At-Risk ⚠️")

        else:
            st.error("Struggling 🚨")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
