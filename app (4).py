import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(
    page_title="Prediksi Kesehatan Mental Mahasiswa",
    layout="wide"
)

st.title("🧠 Prediksi Kesehatan Mental Mahasiswa")
st.write("Sistem Prediksi Berbasis AI dan Data Science")

st.sidebar.header("Input Data Mahasiswa")

age = st.sidebar.slider("Usia", 17, 30, 20)
gpa = st.sidebar.slider("GPA", 1.0, 4.0, 3.0)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

stress = st.sidebar.slider(
    "Stress Level",
    0,
    10,
    5
)

anxiety = st.sidebar.slider(
    "Anxiety Score",
    0,
    10,
    5
)

depression = st.sidebar.slider(
    "Depression Score",
    0,
    10,
    5
)

sleep = st.sidebar.slider(
    "Sleep Hours",
    1,
    12,
    7
)

steps = st.sidebar.number_input(
    "Steps Per Day",
    1000,
    30000,
    8000
)

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

sentiment = st.sidebar.slider(
    "Sentiment Score",
    -1.0,
    1.0,
    0.0
)

# Encoding sederhana
gender_map = {
    "Female": 0,
    "Male": 1,
    "Other": 2
}

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

data = pd.DataFrame({
    'Age':[age],
    'GPA':[gpa],
    'Gender':[gender_map[gender]],
    'Stress_Level':[stress],
    'Anxiety_Score':[anxiety],
    'Depression_Score':[depression],
    'Sleep_Hours':[sleep],
    'Steps_Per_Day':[steps],
    'Mood_Description':[mood_map[mood]],
    'Sentiment_Score':[sentiment]
})

st.subheader("Data Input")

st.dataframe(data)

if st.button("Prediksi Kesehatan Mental"):

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    if prediction == 0:
        status = "Healthy 😊"
        st.success(f"Hasil Prediksi: {status}")

    elif prediction == 1:
        status = "At-Risk ⚠️"
        st.warning(f"Hasil Prediksi: {status}")

    else:
        status = "Struggling 🚨"
        st.error(f"Hasil Prediksi: {status}")

    st.subheader("Interpretasi")

    if prediction == 0:
        st.write("""
        Mahasiswa berada dalam kondisi kesehatan mental yang baik.
        Tetap menjaga pola hidup sehat dan keseimbangan aktivitas.
        """)

    elif prediction == 1:
        st.write("""
        Mahasiswa mulai menunjukkan indikasi risiko gangguan kesehatan mental.
        Disarankan melakukan monitoring dan konseling ringan.
        """)

    else:
        st.write("""
        Mahasiswa menunjukkan tingkat risiko tinggi.
        Direkomendasikan mendapatkan pendampingan profesional.
        """)
