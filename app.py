
import streamlit as st
import numpy as np
import pickle
import joblib

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.set_page_config(
    page_title= "Diabetes Predicted Dashboard",
    page_icon= "🩺"
)

st.sidebar.title('Menu')
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ("Home", "BMI", "Cek Diabetes")
)

if menu == "Home":
    st.title("Diabetes Prediksi", anchor=None, help=None)

    st.markdown(''' Diabetes adalah penyakit mematikan yang terus meningkat di dunia. :blue-background[Pada 2024, 589 juta orang hidup dengan diabetes, 
    dan angka ini diprediksi naik jadi 853 juta pada 2050 (IDF)]. :blue-background[Di Indonesia, 19,5 juta] orang menderita diabetes pada 2021, 
    dan lebih dari 73% penderita belum terdiagnosis (Kemenkes RI). Diabetes bisa menyebabkan serangan jantung, stroke, gagal ginjal, 
    bahkan amputasi (WHO). Deteksi dini sangat penting agar kita bisa mencegah komplikasi serius. ''')

    st.markdown(''' Diabetes adalah :blue-background[pembunuh senyap] yang sering kali tidak terdeteksi hingga menimbulkan komplikasi serius. 
    Keterlambatan dalam diagnosis dan pengobatan menjadi penyebab utama tingginya angka kematian akibat penyakit ini.''')

    st.subheader('Fakta Mengerikan tentang Kematian Akibat Diabetes')

    with st.container():
        st.error("1️⃣ **Setiap 5 detik, 1 orang meninggal akibat diabetes.** Pada tahun 2021, diabetes menyebabkan sekitar 6,7 juta kematian di seluruh dunia.")

    with st.container():
        st.error("2️⃣ **Di Indonesia, pada tahun 2021, tercatat sekitar 236.711 kematian terkait diabetes.**")

    with st.container():
        st.error('3️⃣ **73,7% penderita diabetes di Indonesia tidak terdiagnosis, menyebabkan banyak kasus baru diketahui setelah komplikasi muncul.** ')

elif menu == "BMI":
    st.title('Cek BMI mu!')
    with st.form('BMI Form'):
        bb = st.number_input("Berat Badan (Kg)",value=None, placeholder="Ketikan Berat Badanmu...")
        tb = st.number_input("Tinggi Badan (Cm)",value=None, placeholder="Ketikan Tinggi Badanmu...")
        submitted = st.form_submit_button("Hitung BMI")

        if submitted:
            tb_m = tb/100
            bmi = bb/ (tb_m ** 2)
            if bmi < 18.5:
                st.info(f'BMI anda adalah {bmi: .2f} dan terkategori Berat badan kurang')
            elif 18.5 <= bmi < 24.9:
                st.success(f'BMI anda adalah {bmi: .2f} dan terkategori Berat badan Normal')
            elif 25 <= bmi < 29.9:
                st.warning(f'BMI anda adalah {bmi: .2f} dan terkategori Berat badan berlebih')
            else:
                st.error(f'BMI anda adalah {bmi: .2f} dan terkategori obesitas')

elif menu == "Cek Diabetes":
    st.title("Ayo cek diabetes sedini mungkin!")
    with st.form("Diabetes Form"):
        gender = st.selectbox("Jenis Kelamin:", ["Laki-laki", "Perempuan"])
        age = st.number_input("Usia:",value=None, placeholder="Ketikan Usiamu...")
        hypertension = st.selectbox("Hipertensi:", ["Ya", "Tidak"])
        smoking_history = st.selectbox("Riwayat Merokok:", ["No Info", "never", "former", "current", "ever"])
        heart_disease = st.selectbox("Riwayat Penyakit Jantung:", ["Ya", "Tidak"])
        bmi = st.number_input("BMI:",value=None, placeholder="Ketikan BMImu...")
        HbA1c_level = st.number_input("HbA1c Level:",value=None, placeholder="Ketikan HbA1c Levelmu...")
        glucose_level = st.number_input("Kadar Gula:",value=None, placeholder="Ketikan Kadar Gulamu...")
        submitted = st.form_submit_button("Cek Diabetes")

        if submitted:
            # Label Encoding manual (harus sama seperti saat training!)
            gender_encoded = 1 if gender == "Perempuan" else 0
            hypertension_encoded = 1 if hypertension == "Ya" else 0
            heart_disease_encoded = 1 if heart_disease == "Ya" else 0

            # Untuk smoking_history (harus urutannya sama!)
            mapping_smoking = {
                "No Info": 0,
                "never": 1,
                "former": 2,
                "current": 3,
                "ever": 4
            }
            smoking_encoded = mapping_smoking[smoking_history]

            # Susun input array sesuai urutan fitur saat training
            input_data = np.array([[gender_encoded, age, hypertension_encoded, heart_disease_encoded,
                                    smoking_encoded, bmi, HbA1c_level, glucose_level]])

            # Prediksi
            scaled_input = scaler.transform(input_data)
            prediksi = model.predict(scaled_input)

            # Tampilkan hasil prediksi
            if prediksi[0] == 1:
                st.error("⚠️ Anda berisiko terkena diabetes. Segera konsultasikan ke dokter.")
            else:
                st.success("✅ Anda tidak terdeteksi diabetes. Tetap jaga kesehatan!")
