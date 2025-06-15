import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io

# ========= SIDEBAR =========
with st.sidebar:
    st.title("⛅ Prediksi Curah Hujan (RR)")
    st.markdown("Dashboard ini digunakan untuk menampilkan hasil prediksi yang menggunakan model Machine Learning.")
    st.markdown("---")
    st.info("Upload file CSV atau input data cuaca manual di bawah ini untuk memprediksi curah hujan (RR).")

# ========= JUDUL UTAMA =========
st.markdown("<h1 style='text-align: center;'>🌦️ Dashboard Prediksi Curah Hujan (RR) - Yogyakarta</h1>", unsafe_allow_html=True)

# ========= PILIH MODEL =========
model_option = st.selectbox("🔧 Pilih Model Random Forest", ["Awal", "Tuning"])
model_path = "random_forest_awal.pkl" if model_option == "Awal" else "random_forest_tuned.pkl"
model = joblib.load(model_path)

# ========= EVALUASI (Opsional Ditampilkan) =========
if model_option == "Awal":
    st.markdown("### 📌 Evaluasi Model Awal")
    st.markdown("""
    - **RMSE** : 0.3207  
    - **MAPE** : 240.66%  
    - **R² Score** : 0.2690
    """)
else:
    st.markdown("### 📌 Evaluasi Model Setelah Tuning")
    st.markdown("""
    - **RMSE** : 7.2947  
    - **MAPE** : 1460.25%  
    - **R² Score** : 0.2547
    """)

# ========= TAB MENU =========
tab1, tab2 = st.tabs(["📁 Upload File CSV", "📝 Input Manual"])

# ======= FUNGSI KATEGORI RR =======
def kategori_rr(rr):
    if rr <= 0.2:
        return "Cerah / Ringan ☀️"
    elif rr <= 0.5:
        return "Sedang 🌤️"
    else:
        return "Ekstrem 🌧️"

def badge_kategori(kat):
    if "Cerah" in kat:
        return "🟢 " + kat
    elif "Sedang" in kat:
        return "🟡 " + kat
    else:
        return "🔴 " + kat

# ========= TAB 1: UPLOAD CSV =========
with tab1:
    uploaded_file = st.file_uploader("Upload file CSV cuaca historis", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        df.columns = df.columns.str.strip()

        st.subheader("📊 Data Cuaca yang Diunggah:")
        st.dataframe(df.head())

        drop_cols = ["RR", "TANGGAL"]
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])

        prediction = model.predict(X)
        df["Prediksi RR"] = prediction
        df["Kategori"] = df["Prediksi RR"].apply(lambda x: badge_kategori(kategori_rr(x)))

        st.subheader("📈 Hasil Prediksi Curah Hujan:")
        if "TANGGAL" in df.columns:
            st.dataframe(df[["TANGGAL", "Prediksi RR", "Kategori"]])
        else:
            st.dataframe(df[["Prediksi RR", "Kategori"]])

        # Visualisasi
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["Prediksi RR"], color="skyblue")
        ax.set_title(f"Prediksi Curah Hujan (RR) oleh Random Forest - Model {model_option}", fontsize=14)
        ax.set_xlabel("Index Hari", fontsize=12)
        ax.set_ylabel("RR (Ternormalisasi)", fontsize=12)
        ax.grid(True)
        st.pyplot(fig)

        # Download tombol CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Prediksi CSV",
            data=csv,
            file_name=f'hasil_prediksi_rr_{model_option.lower()}.csv',
            mime='text/csv'
        )

# ========= TAB 2: INPUT MANUAL =========
with tab2:
    st.subheader("📥 Input Data Cuaca Manual")
    tanggal_manual = st.date_input("🗓️ Tanggal Prediksi (opsional)")

    tn = st.number_input("Temperatur Minimum (TN)", 0.0, 50.0, help="Dalam °C")
    tx = st.number_input("Temperatur Maksimum (TX)", 0.0, 50.0, help="Dalam °C")
    tavg = st.number_input("Temperatur Rata-rata (TAVG)", 0.0, 50.0, help="Dalam °C")
    rh = st.number_input("Kelembaban Rata-rata (RH_AVG)", 0.0, 100.0, help="Dalam %")
    ffx = st.number_input("Kecepatan Angin Maksimum (FF_X)", 0.0, 20.0, help="Dalam m/s")
    ffavg = st.number_input("Kecepatan Angin Rata-rata (FF_AVG)", 0.0, 20.0, help="Dalam m/s")

    if st.button("🚀 Prediksi Curah Hujan"):
        input_data = pd.DataFrame([[tn, tx, tavg, rh, ffx, ffavg]],
                                  columns=["TN", "TX", "TAVG", "RH_AVG", "FF_X", "FF_AVG"])
        pred = model.predict(input_data)[0]
        label = kategori_rr(pred)
        badge = badge_kategori(label)
        st.success(f"🌧️ Prediksi Curah Hujan (RR): {pred:.4f}")
        st.info(f"📌 Kategori: **{badge}**")

# ========= FOOTER =========
st.markdown("---")
st.markdown("<center><small>© 2025 - Dela Okta | Skripsi Informatika</small></center>", unsafe_allow_html=True)