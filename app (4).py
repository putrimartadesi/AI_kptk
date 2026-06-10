import streamlit as st
import pandas as pd
import pickle

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open("mental_health_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Prediksi Kesehatan Mental Mahasiswa",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Prediksi Kesehatan Mental Mahasiswa")
st.markdown("### Project AI & Data Science")

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.header("Input Data Mahasiswa")

age = st.sidebar.number_input(
    "Usia",
    min_value=17,
    max_value=35,
    value=20
)

gpa = st.sidebar.slider(
    "GPA",
    1.0,
    4.0,
    3.0
)

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
    min_value=1000,
    max_value=50000,
    value=8000
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

# ======================
# ENCODING
# ======================
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

# ======================
# DATAFRAME INPUT
# ======================
input_data = pd.DataFrame({
    "Age": [age],
    "GPA": [gpa],
    "Gender": [gender_map[gender]],
    "Stress_Level": [stress],
    "Anxiety_Score": [anxiety],
    "Depression_Score": [depression],
    "Sleep_Hours": [sleep],
    "Steps_Per_Day": [steps],
    "Mood_Description": [mood_map[mood]],
    "Sentiment_Score": [sentiment]
})

st.subheader("Data Input")
st.dataframe(input_data)

# ======================
# PREDIKSI
# ======================
if st.button("Prediksi Kesehatan Mental"):

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    if prediction == 0:
        st.success("Healthy 😊")
        st.write(
            "Mahasiswa berada pada kondisi kesehatan mental yang baik."
        )

    elif prediction == 1:
        st.warning("At Risk ⚠️")
        st.write(
            "Mahasiswa memiliki risiko gangguan kesehatan mental dan perlu perhatian."
        )

    else:
        st.error("Struggling 🚨")
        st.write(
            "Mahasiswa membutuhkan pendampingan atau konseling lebih lanjut."
        )

# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown(
    "Project AI & Data Science - Prediksi Kesehatan Mental Mahasiswa"
)
