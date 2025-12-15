import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Import library Fuzzy Logic
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

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
            
            # Logika Warna Status (Sederhana untuk visual)
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
            ax1.plot(df['interval_index'], df[col_flow], color='white', alpha=0.1, label='Data Asli')
            y_flow_line = model_flow.predict(X)
            ax1.plot(df['interval_index'], y_flow_line, color='cyan', alpha=0.5, linewidth=1, label=f'Pola {model_option}')
            ax1.scatter(final_input_value, pred_flow, color='red', s=200, zorder=10, label=display_time_str, edgecolors='white')
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

            # ==========================================
            #   BAGIAN TAMBAHAN 1: FUZZY LOGIC
            # ==========================================
            st.divider()
            st.subheader("🤖 Analisis Fuzzy Logic (Validasi)")

            if HAS_FUZZY:
                # 1. Tentukan Variabel Fuzzy (Universe)
                max_flow_data = df[col_flow].max()
                max_occ_data = df[col_occ].max()
                
                # Antecedents (Input)
                # Flow: 0 sampai max flow
                f_flow = ctrl.Antecedent(np.arange(0, max_flow_data + 1, 1), 'flow')
                # Occupancy: 0 sampai 100% (atau max data)
                f_occ = ctrl.Antecedent(np.arange(0, 101, 1), 'occupancy')
                
                # Consequent (Output): Tingkat Kemacetan (0-100)
                f_condition = ctrl.Consequent(np.arange(0, 101, 1), 'condition')

                # 2. Tentukan Membership Function (Rendah, Sedang, Tinggi)
                # Flow (Otomatis membagi 3 range)
                f_flow.automf(3, names=['Rendah', 'Sedang', 'Tinggi'])
                
                # Occupancy (Manual custom biar lebih akurat)
                f_occ['Rendah'] = fuzz.trimf(f_occ.universe, [0, 0, 30])
                f_occ['Sedang'] = fuzz.trimf(f_occ.universe, [20, 50, 80])
                f_occ['Tinggi'] = fuzz.trimf(f_occ.universe, [60, 100, 100])

                # Output Condition
                f_condition['Lancar'] = fuzz.trimf(f_condition.universe, [0, 0, 50])
                f_condition['Padat'] = fuzz.trimf(f_condition.universe, [40, 60, 80])
                f_condition['Macet'] = fuzz.trimf(f_condition.universe, [70, 100, 100])

                # 3. Rules (Aturan Fuzzy)
                rule1 = ctrl.Rule(f_occ['Rendah'], f_condition['Lancar'])
                rule2 = ctrl.Rule(f_occ['Sedang'] & f_flow['Sedang'], f_condition['Padat'])
                rule3 = ctrl.Rule(f_occ['Tinggi'] | f_flow['Tinggi'], f_condition['Macet'])
                
                # Tambahan Rule untuk logika umum
                rule4 = ctrl.Rule(f_occ['Sedang'] & f_flow['Rendah'], f_condition['Lancar']) # Occ sedang tapi flow rendah (mungkin lampu merah bentar)
                rule5 = ctrl.Rule(f_occ['Sedang'] & f_flow['Tinggi'], f_condition['Padat'])  # Ramai lancar

                # 4. Control System
                traffic_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
                traffic_sim = ctrl.ControlSystemSimulation(traffic_ctrl)

                # 5. Input Nilai Prediksi ke Fuzzy
                # Kita masukkan nilai hasil prediksi AI (Regression) ke Fuzzy Logic
                # Pastikan nilai tidak negatif
                input_f_flow = max(0, pred_flow)
                input_f_occ = max(0, min(100, pred_occ)) # Cap di 100

                traffic_sim.input['flow'] = input_f_flow
                traffic_sim.input['occupancy'] = input_f_occ

                # Hitung
                try:
                    traffic_sim.compute()
                    result_fuzzy = traffic_sim.output['condition']
                    
                    # Tampilkan Hasil Fuzzy
                    fc1, fc2 = st.columns([1, 2])
                    
                    with fc1:
                        st.markdown("#### Hasil Fuzzy Score")
                        st.markdown(f"<h1 style='text-align: center; color: yellow;'>{result_fuzzy:.2f}/100</h1>", unsafe_allow_html=True)
                        
                        f_status = "Tidak Diketahui"
                        if result_fuzzy < 45: f_status = "LANCAR (Fuzzy)"
                        elif result_fuzzy < 75: f_status = "PADAT (Fuzzy)"
                        else: f_status = "MACET (Fuzzy)"
                        
                        st.markdown(f"<p style='text-align: center;'>Status: <b>{f_status}</b></p>", unsafe_allow_html=True)

                    with fc2:
                        st.info("Logika Fuzzy mempertimbangkan ketidakpastian. Jika prediksi AI bilang 'Macet' tapi Fuzzy bilang 'Padat', berarti kondisi ada di perbatasan.")

                    # Visualisasi Grafik Fuzzy (Output)
                    fig_fuz, ax_fuz = plt.subplots(figsize=(8, 3))
                    f_condition.view(sim=traffic_sim, ax=ax_fuz)
                    fig_fuz.patch.set_facecolor('#0e1117') # Match Streamlit dark theme bg
                    ax_fuz.set_facecolor('#0e1117')
                    ax_fuz.tick_params(colors='white')
                    ax_fuz.xaxis.label.set_color('white')
                    ax_fuz.yaxis.label.set_color('white')
                    # Hapus judul default yang mengganggu
                    ax_fuz.set_title("Posisi Kondisi pada Grafik Membership", color='white')
                    st.pyplot(fig_fuz)
                    
                except Exception as e:
                    st.warning(f"Tidak dapat menghitung Fuzzy Logic (Mungkin input di luar range): {e}")

            else:
                st.error("Library `scikit-fuzzy` belum terinstall. Mohon install dengan `pip install scikit-fuzzy` di terminal.")

            # ==========================================
            #   BAGIAN TAMBAHAN 2: STATISTIK DATA MENTAH
            # ==========================================
            st.divider()
            with st.expander("📊 Lihat Statistik Data Mentah (Flow & Occupancy)", expanded=False):
                st.markdown("### Ringkasan Statistik")
                # Tampilkan Describe (Count, Mean, Std, Min, Max, dll)
                st.dataframe(df[[col_flow, col_occ]].describe().T, use_container_width=True)

                st.markdown("### Distribusi Data")
                col_hist1, col_hist2 = st.columns(2)
                
                with col_hist1:
                    st.markdown(f"**Histogram {col_flow}**")
                    fig_h1, ax_h1 = plt.subplots()
                    ax_h1.hist(df[col_flow], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                    ax_h1.set_facecolor('black')
                    fig_h1.patch.set_facecolor('black')
                    ax_h1.tick_params(colors='white')
                    st.pyplot(fig_h1)

                with col_hist2:
                    st.markdown(f"**Histogram {col_occ}**")
                    fig_h2, ax_h2 = plt.subplots()
                    ax_h2.hist(df[col_occ], bins=30, color='salmon', edgecolor='black', alpha=0.7)
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
    # Halaman Awal
    c1, c2 = st.columns([1, 4])
    with c1:
        # Placeholder Logo
        st.image("https://cdn-icons-png.flaticon.com/512/2382/2382461.png", width=100)
    with c2:
        st.title("Sistem Prediksi Kemacetan")
        st.markdown("### Aplikasi Dashboard AI")
    
    st.info("👋 Silakan upload file CSV Traffic di sidebar sebelah kiri.")
