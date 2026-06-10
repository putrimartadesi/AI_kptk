import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD MODEL DAN SCALER
# ===============================
try:
    model = joblib.load("mental_health_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ===============================
# JUDUL APLIKASI
# ===============================
st.set_page_config(
    page_title="Prediksi Kesehatan Mental Mahasiswa",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Prediksi Kesehatan Mental Mahasiswa")
st.markdown("### Sistem Prediksi Berbasis AI dan Data Science")

# ===============================
# INPUT USER
# ===============================
st.sidebar.header("Input Data")

gpa = st.sidebar.slider(
    "GPA",
    min_value=1.0,
    max_value=4.0,
    value=3.0,
    step=0.1
)

stress = st.sidebar.slider(
    "Stress Level",
    min_value=0,
    max_value=10,
    value=5
)

anxiety = st.sidebar.slider(
    "Anxiety Score",
    min_value=0,
    max_value=10,
    value=5
)

depression = st.sidebar.slider(
    "Depression Score",
    min_value=0,
    max_value=10,
    value=5
)

daily_reflections = st.sidebar.text_area(
    "Daily Reflections",
    "I feel good today"
)

# ===============================
# DATA INPUT
# ===============================
input_df = pd.DataFrame({
    "GPA": [gpa],
    "Stress_Level": [stress],
    "Anxiety_Score": [anxiety],
    "Depression_Score": [depression],
    "Daily_Reflections": [daily_reflections]
})

st.subheader("Data Input")
st.dataframe(input_df)

# ===============================
# PREDIKSI
# ===============================
if st.button("Prediksi"):

    try:

        # Jika scaler hanya untuk numerik
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        if prediction == 0:
            st.success("Healthy 😊")
            st.write("Mahasiswa berada dalam kondisi kesehatan mental yang baik.")

        elif prediction == 1:
            st.warning("At Risk ⚠️")
            st.write("Mahasiswa memiliki risiko gangguan kesehatan mental.")

        else:
            st.error("Struggling 🚨")
            st.write("Mahasiswa memerlukan perhatian dan pendampingan lebih lanjut.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
