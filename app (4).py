import streamlit as st
import pandas as pd
import joblib

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Prediksi Kesehatan Mental Mahasiswa",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Prediksi Kesehatan Mental Mahasiswa")

# ==========================
# LOAD MODEL
# ==========================
try:
    model = joblib.load("mental_health_model.pkl")
    scaler = joblib.load("scaler.pkl")

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info("""
    Pastikan:
    1. File mental_health_model.pkl ada dalam folder project
    2. File scaler.pkl ada dalam folder project
    3. Model dibuat menggunakan joblib.dump()
    4. Versi scikit-learn sama dengan saat training
    """)
    st.stop()

# ==========================
# SIDEBAR INPUT
# ==========================
st.sidebar.header("Input Data Mahasiswa")

age = st.sidebar.number_input("Age", 17, 35, 20)

gpa = st.sidebar.slider(
    "GPA",
    min_value=1.0,
    max_value=4.0,
    value=3.0
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

# ==========================
# ENCODING
# ==========================
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

# ==========================
# DATA INPUT
# ==========================
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

st.subheader("Data Input Mahasiswa")
st.dataframe(input_data)

# ==========================
# PREDIKSI
# ==========================
if st.button("Prediksi"):

    try:
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]

        if prediction == 0:
            st.success("Healthy 😊")
            st.write("Mahasiswa berada pada kondisi kesehatan mental yang baik.")

        elif prediction == 1:
            st.warning("At-Risk ⚠️")
            st.write("Mahasiswa memiliki risiko gangguan kesehatan mental.")

        else:
            st.error("Struggling 🚨")
            st.write("Mahasiswa membutuhkan perhatian dan pendampingan lebih lanjut.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
# ==========================
with st.expander("Lihat File Repository"):
    st.write(os.listdir())
