import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# --- FUNGSI FUZZY LOGIC MANUAL (TANPA LIBRARY) ---
def triangular_membership(x, a, b, c):
    """Fungsi keanggotaan segitiga manual"""
    return max(0, min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def calculate_fuzzy_score(flow_val, occ_val, max_flow):
    # 1. FUZZIFICATION (Merubah angka ke Bahasa Fuzzy)
    
    # --- Variabel Flow (Rendah, Sedang, Tinggi) ---
    # Batas-batas flow (disesuaikan dinamis)
    f_low_peak = 0
    f_mid_peak = max_flow * 0.5
    f_high_peak = max_flow
    
    flow_rendah = triangular_membership(flow_val, -1, 0, f_mid_peak)
    flow_sedang = triangular_membership(flow_val, 0, f_mid_peak, f_high_peak)
    flow_tinggi = triangular_membership(flow_val, f_mid_peak, f_high_peak, f_high_peak * 1.5)
    
    # --- Variabel Occupancy (Rendah, Sedang, Tinggi) ---
    # Batas occupancy 0 - 100%
    occ_rendah = triangular_membership(occ_val, -1, 0, 40)
    occ_sedang = triangular_membership(occ_val, 20, 50, 80)
    occ_tinggi = triangular_membership(occ_val, 60, 100, 150)
    
    # 2. INFERENCE (Aturan/Rule Base)
    # Rule 1: Jika Occ Rendah -> Lancar
    r1 = occ_rendah 
    z1 = 20 # Titik pusat Lancar (0-40)
    
    # Rule 2: Jika Occ Sedang DAN Flow Sedang -> Padat
    r2 = min(occ_sedang, flow_sedang)
    z2 = 60 # Titik pusat Padat (40-80)
    
    # Rule 3: Jika Occ Tinggi ATAU Flow Tinggi -> Macet
    r3 = max(occ_tinggi, flow_tinggi)
    z3 = 90 # Titik pusat Macet (80-100)
    
    # Rule 4: Jika Occ Sedang tapi Flow Rendah -> Lancar (Biasanya lampu merah sebentar)
    r4 = min(occ_sedang, flow_rendah)
    z4 = 30 # Agak lancar
    
    # 3. DEFUZZIFICATION (Metode Sugeno Sederhana / Weighted Average)
    # Menghitung rata-rata terbobot
    pembilang = (r1 * z1) + (r2 * z2) + (r3 * z3) + (r4 * z4)
    penyebut = r1 + r2 + r3 + r4
    
    if penyebut == 0:
        return 0
    else:
        return pembilang / penyebut

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Prediksi Kemacetan", layout="wide")
plt.style.use('dark_background')

# --- SIDEBAR ---
st.sidebar.title("🚦 Sistem Prediksi Kemacetan")
st.sidebar.markdown("Dashboard AI untuk analisis & prediksi lalu lintas.")

# --- 1. UPLOAD DATA ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Pre-processing
        df['interval_index'] = np.arange(len(df))
        max_val = len(df)
        
        # Konfigurasi Kolom
        st.sidebar.subheader("⚙️ Konfigurasi Data")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        col_flow = st.sidebar.selectbox("Kolom Flow:", numeric_cols, index=0)
        idx_occ = 1 if len(numeric_cols) > 1 else 0
        col_occ = st.sidebar.selectbox("Kolom Occupancy:", numeric_cols, index=idx_occ)

        # --- 2. MODEL AI ---
        st.sidebar.divider()
        st.sidebar.subheader("🧠 Model AI")
        model_option = st.sidebar.selectbox("Pilih Algoritma:", 
            ("Polynomial Regression (Rekomen)", "Decision Tree", "Linear Regression"))

        # --- 3. METODE INPUT ---
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Metode Input")
        input_method = st.sidebar.radio("Pilih Cara Input:", ["Input Detik (Interval)", "Input Jam (Waktu)"])

        final_input_value = 0
        display_time_str = ""

        if input_method == "Input Detik (Interval)":
            input_detik = st.sidebar.number_input(f"Masukkan Detik (0 - {max_val})", min_value=0, max_value=max_val, value=int(max_val/2))
            final_input_value = input_detik
            display_time_str = f"Interval {input_detik}"
        else:
            input_time = st.sidebar.time_input("Pilih Jam:", value=datetime.time(12, 0))
            seconds_in_day = 24 * 3600
            user_seconds = (input_time.hour * 3600) + (input_time.minute * 60) + input_time.second
            ratio = user_seconds / seconds_in_day
            final_input_value = int(ratio * max_val)
            final_input_value = max(0, min(final_input_value, max_val - 1))
            display_time_str = f"Pukul {input_time.strftime('%H:%M:%S')}"
            st.sidebar.info(f"💡 Info: Pukul {input_time} dikonversi menjadi Interval ke-{final_input_value}")

        # --- TOMBOL PREDIKSI ---
        if st.sidebar.button("🚀 Prediksi Sekarang", type="primary"):
            
            # Training
            X = df[['interval_index']]
            y_flow = df[col_flow]
            y_occ = df[col_occ]
            
            if "Decision Tree" in model_option:
                model_flow = DecisionTreeRegressor(max_depth=10)
                model_occ = DecisionTreeRegressor(max_depth=10)
            elif "Polynomial" in model_option:
                model_flow = make_pipeline(PolynomialFeatures(4), LinearRegression())
                model_occ = make_pipeline(PolynomialFeatures(4), LinearRegression())
            else:
                model_flow = LinearRegression()
                model_occ = LinearRegression()
            
            model_flow.fit(X, y_flow)
            model_occ.fit(X, y_occ)
            
            # Prediksi
            input_df = pd.DataFrame({'interval_index': [final_input_value]})
            pred_flow = model_flow.predict(input_df)[0]
            pred_occ = model_occ.predict(input_df)[0]
            
            thresh_flow = df[col_flow].max() * 0.7
            thresh_occ = df[col_occ].max() * 0.3
            
            # Tampilan Dashboard
            st.title(f"Sistem Prediksi Kemacetan")
            
            c_stat, c_flow, c_occ, c_int, c_mod = st.columns(5)
            
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

            # Grafik Visualisasi
            st.subheader("📈 Visualisasi Prediksi")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.patch.set_facecolor('black')
            fig.patch.set_alpha(0.0)

            # Grafik Flow
            ax1.plot(df['interval_index'], df[col_flow], color='white', alpha=0.1, label='Data Asli')
            ax1.plot(df['interval_index'], model_flow.predict(X), color='cyan', alpha=0.5, linewidth=1, label=f'Pola AI')
            ax1.scatter(final_input_value, pred_flow, color='red', s=200, zorder=10, label=display_time_str, edgecolors='white')
            ax1.axhline(thresh_flow, color='#ffcc00', linestyle='--', label='Batas Padat')
            ax1.set_title("Prediksi Flow", color='white')
            ax1.set_facecolor('black')
            ax1.legend(facecolor='#262730', edgecolor='white').get_texts()[0].set_color("white")
            ax1.tick_params(colors='white')

            # Grafik Occupancy
            ax2.plot(df['interval_index'], df[col_occ], color='white', alpha=0.1, label='Data Asli')
            ax2.plot(df['interval_index'], model_occ.predict(X), color='cyan', alpha=0.5, linewidth=1, label=f'Pola AI')
            ax2.scatter(final_input_value, pred_occ, color='red', s=200, zorder=10, label=display_time_str, edgecolors='white')
            ax2.axhline(thresh_occ, color='#ffcc00', linestyle='--', label='Batas Macet')
            ax2.set_title("Prediksi Occupancy", color='white')
            ax2.set_facecolor('black')
            ax2.legend(facecolor='#262730', edgecolor='white').get_texts()[0].set_color("white")
            ax2.tick_params(colors='white')

            st.pyplot(fig)

            # ==========================================
            #   BAGIAN 1: FUZZY LOGIC (MANUAL - NO LIBRARY)
            # ==========================================
            st.divider()
            st.subheader("🤖 Analisis Fuzzy Logic (Manual Calculation)")
            
            # Panggil fungsi manual di atas
            max_flow_data = df[col_flow].max()
            fuzzy_score = calculate_fuzzy_score(pred_flow, pred_occ, max_flow_data)
            
            fc1, fc2 = st.columns([1, 2])
            with fc1:
                st.markdown("#### Hasil Fuzzy Score")
                st.markdown(f"<h1 style='text-align: center; color: yellow;'>{fuzzy_score:.2f}/100</h1>", unsafe_allow_html=True)
                
                f_status = "Tidak Diketahui"
                if fuzzy_score < 45: f_status = "LANCAR"
                elif fuzzy_score < 75: f_status = "PADAT"
                else: f_status = "MACET"
                
                st.markdown(f"<p style='text-align: center;'>Status: <b>{f_status} (Validasi Fuzzy)</b></p>", unsafe_allow_html=True)

            with fc2:
                st.info("ℹ️ **Metode:** Menggunakan perhitungan Fuzzy Logic Manual (Membership Function & Rules). Tanpa dependency library eksternal.")
                # Visualisasi Bar sederhana pengganti grafik library
                st.write("Tingkat Kemacetan (0 = Lancar, 100 = Macet Total)")
                st.progress(int(min(fuzzy_score, 100)))

            # ==========================================
            #   BAGIAN 2: STATISTIK DATA MENTAH
            # ==========================================
            st.divider()
            with st.expander("📊 Lihat Statistik Data Mentah", expanded=False):
                st.markdown("### Ringkasan Statistik")
                st.dataframe(df[[col_flow, col_occ]].describe().T, use_container_width=True)

                st.markdown("### Distribusi Data")
                c1, c2 = st.columns(2)
                with c1:
                    fig_h1, ax_h1 = plt.subplots()
                    ax_h1.hist(df[col_flow], bins=30, color='skyblue', edgecolor='black')
                    ax_h1.set_title(f"Histogram {col_flow}", color='white')
                    ax_h1.set_facecolor('black')
                    fig_h1.patch.set_facecolor('black')
                    ax_h1.tick_params(colors='white')
                    st.pyplot(fig_h1)
                with c2:
                    fig_h2, ax_h2 = plt.subplots()
                    ax_h2.hist(df[col_occ], bins=30, color='salmon', edgecolor='black')
                    ax_h2.set_title(f"Histogram {col_occ}", color='white')
                    ax_h2.set_facecolor('black')
                    fig_h2.patch.set_facecolor('black')
                    ax_h2.tick_params(colors='white')
                    st.pyplot(fig_h2)
                    
                st.markdown("### Data Mentah (50 Baris Pertama)")
                st.dataframe(df.head(50), use_container_width=True)

        else:
            st.info("👈 Pilih Metode Input, lalu klik tombol 'Prediksi Sekarang'.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("👋 Silakan upload file CSV Traffic di sidebar sebelah kiri.")
