import streamlit as st
import pandas as pd
import pickle
import os

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Prediksi Kesehatan Mental Siswa",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Prediksi Kesehatan Mental Siswa")
st.write("Masukkan data siswa untuk melakukan prediksi.")

# ==========================
# LOAD MODEL
# ==========================
MODEL_PATH = "mental_health_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    st.success("✅ Model berhasil dimuat")

except FileNotFoundError:
    st.error(f"❌ File '{MODEL_PATH}' tidak ditemukan.")
    st.stop()

except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ==========================
# INPUT USER
# ==========================
gpa = st.number_input(
    "GPA",
    min_value=0.0,
    max_value=4.0,
    value=3.0
)

stress = st.slider(
    "Stress Level",
    min_value=0,
    max_value=10,
    value=5
)

anxiety = st.slider(
    "Anxiety Score",
    min_value=0,
    max_value=100,
    value=50
)

depression = st.slider(
    "Depression Score",
    min_value=0,
    max_value=100,
    value=50
)

# ==========================
# PREDIKSI
# ==========================
if st.button("Prediksi"):

    data = pd.DataFrame({
        'GPA': [gpa],
        'Stress_Level': [stress],
        'Anxiety_Score': [anxiety],
        'Depression_Score': [depression]
    })

    try:
        prediction = model.predict(data)

        st.subheader("Hasil Prediksi")

        if prediction[0] == 0:
            st.warning("Kategori 0")

        elif prediction[0] == 1:
            st.info("Kategori 1")

        elif prediction[0] == 2:
            st.success("Kategori 2")

        else:
            st.write(f"Hasil Prediksi: {prediction[0]}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ==========================
# DEBUG
# ==========================
with st.expander("Lihat File Repository"):
    st.write(os.listdir())
