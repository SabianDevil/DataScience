import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Prediksi Kemacetan", layout="wide")
plt.style.use('dark_background')

# --- SIDEBAR: HEADER ---
st.sidebar.title("🚦 Sistem Prediksi Kemacetan")
st.sidebar.markdown("Dashboard AI untuk analisis & prediksi lalu lintas.")

# --- 1. UPLOAD DATA ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- PRE-PROCESSING ---
        # Membuat index interval (0, 1, 2... dst) sebagai fitur waktu
        df['interval_index'] = np.arange(len(df))
        max_val = len(df)
        
        # Konfigurasi Kolom
        st.sidebar.subheader("⚙️ Konfigurasi Data")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        col_flow = st.sidebar.selectbox("Kolom Flow:", numeric_cols, index=0)
        idx_occ = 1 if len(numeric_cols) > 1 else 0
        col_occ = st.sidebar.selectbox("Kolom Occupancy:", numeric_cols, index=idx_occ)

        # --- 2. PENGATURAN MODEL AI ---
        st.sidebar.divider()
        st.sidebar.subheader("🧠 Model AI")
        model_option = st.sidebar.selectbox(
            "Pilih Algoritma:",
            ("Polynomial Regression (Rekomen)", "Decision Tree", "Linear Regression")
        )

        # --- 3. METODE INPUT (FITUR BARU) ---
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Metode Input")
        
        # Pilihan Radio Button
        input_method = st.sidebar.radio("Pilih Cara Input:", ["Input Detik (Interval)", "Input Jam (Waktu)"])

        final_input_value = 0 # Variabel untuk menyimpan hasil inputan user
        display_time_str = "" # String untuk label grafik

        if input_method == "Input Detik (Interval)":
            # Cara Lama: Input Angka Langsung
            input_detik = st.sidebar.number_input(f"Masukkan Detik (0 - {max_val})", min_value=0, max_value=max_val, value=int(max_val/2))
            final_input_value = input_detik
            display_time_str = f"Interval {input_detik}"
            
        else:
            # Cara Baru: Input Jam -> Konversi ke Interval
            input_time = st.sidebar.time_input("Pilih Jam:", value=datetime.time(12, 0))
            
            # Logika Konversi: Jam -> Total Detik -> Persentase -> Index Data
            seconds_in_day = 24 * 3600
            user_seconds = (input_time.hour * 3600) + (input_time.minute * 60) + input_time.second
            
            # Hitung rasio (misal jam 12 siang = 0.5 atau 50%)
            ratio = user_seconds / seconds_in_day
            
            # Terapkan rasio ke total baris data
            final_input_value = int(ratio * max_val)
            
            # Pastikan tidak melebihi batas data
            final_input_value = max(0, min(final_input_value, max_val - 1))
            display_time_str = f"Pukul {input_time.strftime('%H:%M:%S')}"
            
            st.sidebar.info(f"💡 Info: Pukul {input_time} dikonversi sistem menjadi Interval ke-{final_input_value}")


        # --- TOMBOL PREDIKSI ---
        if st.sidebar.button("🚀 Prediksi Sekarang", type="primary"):
            
            # --- A. TRAINING MODEL ---
            X = df[['interval_index']]
            y_flow = df[col_flow]
            y_occ = df[col_occ]
            
            # Inisialisasi Model
            if "Decision Tree" in model_option:
                model_flow = DecisionTreeRegressor(max_depth=10)
                model_occ = DecisionTreeRegressor(max_depth=10)
            elif "Polynomial" in model_option:
                model_flow = make_pipeline(PolynomialFeatures(4), LinearRegression())
                model_occ = make_pipeline(PolynomialFeatures(4), LinearRegression())
            else:
                model_flow = LinearRegression()
                model_occ = LinearRegression()
            
            # Latih Model
            model_flow.fit(X, y_flow)
            model_occ.fit(X, y_occ)
            
            # --- B. PREDIKSI ---
            input_df = pd.DataFrame({'interval_index': [final_input_value]})
            pred_flow = model_flow.predict(input_df)[0]
            pred_occ = model_occ.predict(input_df)[0]
            
            # Threshold Otomatis
            thresh_flow = df[col_flow].max() * 0.7
            thresh_occ = df[col_occ].max() * 0.3
            
            # --- C. TAMPILAN DASHBOARD ---
            st.title(f"Sistem Prediksi Kemacetan")
            
            # Kartu Status Utama
            c_stat, c_flow, c_occ, c_int, c_mod = st.columns(5)
            
            # Logika Warna Status
            status_text = "LANCAR"
            status_color = "green"
            if pred_occ > thresh_occ:
                status_text = "MACET TOTAL"
                status_color = "red"
            elif pred_flow > thresh_flow:
                status_text = "PADAT MERAYAP"
                status_color = "orange"

            c_stat.markdown(f"### Status:\n## :{status_color}[{status_text}]")
            c_flow.metric("Prediksi Flow", f"{pred_flow:.1f}")
            c_occ.metric("Prediksi Occupancy", f"{pred_occ:.2f}%")
            c_int.metric("Interval Input", final_input_value)
            c_mod.markdown(f"**Model:**\n{model_option}")
            
            st.divider()

            # --- D. GRAFIK VISUALISASI ---
            st.subheader("📈 Visualisasi Prediksi")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            # Background transparan
            fig.patch.set_facecolor('black')
            fig.patch.set_alpha(0.0)

            # --- Grafik 1: Flow ---
            # Data Asli
            ax1.plot(df['interval_index'], df[col_flow], color='white', alpha=0.1, label='Data Asli')
            # Garis Model AI
            y_flow_line = model_flow.predict(X)
            ax1.plot(df['interval_index'], y_flow_line, color='cyan', alpha=0.5, linewidth=1, label=f'Pola {model_option}')
            # Titik Prediksi
            ax1.scatter(final_input_value, pred_flow, color='red', s=200, zorder=10, label=display_time_str, edgecolors='white')
            # Threshold
            ax1.axhline(thresh_flow, color='#ffcc00', linestyle='--', label='Batas Padat')
            
            ax1.set_title("Prediksi Flow", color='white')
            ax1.set_facecolor('black')
            ax1.grid(False)
            leg1 = ax1.legend(facecolor='#262730', edgecolor='white')
            for t in leg1.get_texts(): t.set_color("white")
            ax1.tick_params(colors='white')

            # --- Grafik 2: Occupancy ---
            ax2.plot(df['interval_index'], df[col_occ], color='white', alpha=0.1, label='Data Asli')
            y_occ_line = model_occ.predict(X)
            ax2.plot(df['interval_index'], y_occ_line, color='cyan', alpha=0.5, linewidth=1, label=f'Pola {model_option}')
            
            ax2.scatter(final_input_value, pred_occ, color='red', s=200, zorder=10, label=display_time_str, edgecolors='white')
            ax2.axhline(thresh_occ, color='#ffcc00', linestyle='--', label='Batas Macet')
            
            ax2.set_title("Prediksi Occupancy", color='white')
            ax2.set_facecolor('black')
            ax2.grid(False)
            leg2 = ax2.legend(facecolor='#262730', edgecolor='white')
            for t in leg2.get_texts(): t.set_color("white")
            ax2.tick_params(colors='white')

            st.pyplot(fig)

        else:
            st.info("👈 Pilih Metode Input, lalu klik tombol 'Prediksi Sekarang'.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

else:
    # Halaman Awal
    c1, c2 = st.columns([1, 4])
    with c1:
        # Placeholder Logo (Bisa diganti URL logo universitas Anda)
        st.image("https://cdn-icons-png.flaticon.com/512/2382/2382461.png", width=100)
    with c2:
        st.title("Sistem Prediksi Kemacetan")
        st.markdown("### Aplikasi Dashboard AI")
    
    st.info("👋 Silakan upload file CSV Traffic di sidebar sebelah kiri.")
