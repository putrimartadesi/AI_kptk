import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("mental_health_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Prediksi Kesehatan Mental Mahasiswa")

# Input
gpa = st.slider("GPA", 1.0, 4.0, 3.0)

stress = st.slider("Stress Level", 0, 10, 5)

anxiety = st.slider("Anxiety Score", 0, 10, 5)

depression = st.slider("Depression Score", 0, 10, 5)

mood = st.selectbox(
    "Mood Description",
    ["Happy", "Calm", "Neutral", "Sad", "Anxious", "Stressed"]
)

# DAILY REFLECTIONS HARUS NUMERIK
daily_reflections = st.slider(
    "Daily Reflections Score",
    0,
    100,
    50
)

# Encoding Mood
mood_map = {
    "Happy": 0,
    "Calm": 1,
    "Neutral": 2,
    "Sad": 3,
    "Anxious": 4,
    "Stressed": 5
}

input_df = pd.DataFrame({
    "GPA": [gpa],
    "Stress_Level": [stress],
    "Anxiety_Score": [anxiety],
    "Depression_Score": [depression],
    "Mood_Description": [mood_map[mood]],
    "Daily_Reflections": [daily_reflections]
})

# Samakan urutan kolom dengan model
if hasattr(model, "feature_names_in_"):
    input_df = input_df.reindex(columns=model.feature_names_in_)

st.write(input_df)

if st.button("Prediksi"):

    try:
        data_scaled = scaler.transform(input_df)
        prediction = model.predict(data_scaled)

        st.success(f"Hasil Prediksi: {prediction[0]}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
