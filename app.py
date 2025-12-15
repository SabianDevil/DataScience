import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# --- FUNGSI BANTUAN ---
def get_triangular_y(x_array, a, b, c):
    """Menghitung nilai Y untuk grafik segitiga"""
    y = []
    for x in x_array:
        val = max(0, min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))
        y.append(val)
    return np.array(y)

def triangular_membership(x, a, b, c):
    """Fungsi hitung satu titik"""
    return max(0, min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def calculate_fuzzy_score(flow_val, occ_val, max_flow, max_occ):
    # --- 1. FUZZIFICATION (DINAMIS) ---
    
    # FLOW: Batas berdasarkan max data flow
    f_mid = max_flow * 0.5
    f_high_peak = max_flow
    
    flow_rendah = triangular_membership(flow_val, -1, 0, f_mid)
    flow_sedang = triangular_membership(flow_val, 0, f_mid, f_high_peak)
    flow_tinggi = triangular_membership(flow_val, f_mid, f_high_peak, f_high_peak * 1.5)
    
    # OCCUPANCY: Batas berdasarkan max data occupancy (PERBAIKAN DISINI)
    # Kita bagi range data occupancy menjadi 3 bagian proporsional
    o_low_limit = max_occ * 0.4  # 40% dari max data
    o_mid_peak = max_occ * 0.5   # 50% dari max data
    o_high_start = max_occ * 0.6 # 60% dari max data
    
    occ_rendah = triangular_membership(occ_val, -1, 0, o_low_limit)
    occ_sedang = triangular_membership(occ_val, max_occ*0.2, o_mid_peak, max_occ*0.8)
    occ_tinggi = triangular_membership(occ_val, o_high_start, max_occ, max_occ * 1.5)
    
    # --- 2. INFERENCE RULES ---
    # R1: Occ Rendah -> Lancar
    r1 = occ_rendah 
    z1 = 20 
    
    # R2: Occ Sedang & Flow Sedang -> Padat
    r2 = min(occ_sedang, flow_sedang)
    z2 = 60 
    
    # R3: Occ Tinggi ATAU Flow Tinggi -> Macet
    # Gunakan max() agar jika salah satu tinggi, langsung dianggap macet
    r3 = max(occ_tinggi, flow_tinggi)
    z3 = 90 
    
    # R4: Kasus Khusus (Occ Sedang tapi Flow Rendah) -> Lancar
    r4 = min(occ_sedang, flow_rendah) 
    z4 = 30
    
    # --- 3. DEFUZZIFICATION ---
    pembilang = (r1 * z1) + (r2 * z2) + (r3 * z3) + (r4 * z4)
    penyebut = r1 + r2 + r3 + r4
    
    score = 0 if penyebut == 0 else pembilang / penyebut
    
    # Return semua variabel untuk visualisasi
    return score, flow_rendah, flow_sedang, flow_tinggi, occ_rendah, occ_sedang, occ_tinggi

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
        
        df['interval_index'] = np.arange(len(df))
        max_val = len(df)
        
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
            c_occ.metric("Prediksi Occupancy", f"{pred_occ:.2f}")
            c_int.metric("Interval Input", final_input_value)
            c_mod.markdown(f"**Model:**\n{model_option}")
            
            st.divider()

            # Grafik Visualisasi Prediksi
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
            #   BAGIAN 1: FUZZY LOGIC MANUAL (PERBAIKAN VISUAL)
            # ==========================================
            st.divider()
            st.subheader("🤖 Analisis Fuzzy Logic")

            # Ambil nilai MAX dari data asli untuk penskalaan
            max_flow_data = df[col_flow].max()
            max_occ_data = df[col_occ].max() # <-- INI KUNCI PERBAIKANNYA

            # Hitung Score dengan batas dinamis
            f_score, f_low, f_mid, f_hi, o_low, o_mid, o_hi = calculate_fuzzy_score(pred_flow, pred_occ, max_flow_data, max_occ_data)

            col_viz, col_rule = st.columns([2, 1])

            with col_viz:
                st.markdown("**Membership Functions (Posisi Data Anda)**")
                
                # --- PLOT 1: FLOW ---
                fig_f, ax_f = plt.subplots(figsize=(8, 3))
                fig_f.patch.set_facecolor('#0e1117') 
                ax_f.set_facecolor('#0e1117')
                
                x_flow = np.linspace(0, max_flow_data * 1.5, 500)
                mid_point_f = max_flow_data * 0.5
                
                # Gambar Segitiga Flow
                y_low = get_triangular_y(x_flow, -1, 0, mid_point_f)
                y_mid = get_triangular_y(x_flow, 0, mid_point_f, max_flow_data)
                y_high = get_triangular_y(x_flow, mid_point_f, max_flow_data, max_flow_data * 1.5)
                
                ax_f.plot(x_flow, y_low, label='Rendah', color='#1f77b4')
                ax_f.plot(x_flow, y_mid, label='Sedang', color='#ff7f0e')
                ax_f.plot(x_flow, y_high, label='Tinggi', color='#2ca02c')
                ax_f.axvline(pred_flow, color='white', linewidth=3, linestyle='-', label='Input Anda')
                
                ax_f.set_title(f"Variable: Flow (Value: {pred_flow:.0f})", color='white')
                ax_f.legend(facecolor='#262730', edgecolor='white', loc='center right').get_texts()[0].set_color("white")
                ax_f.tick_params(colors='white')
                st.pyplot(fig_f)
                
                # --- PLOT 2: OCCUPANCY (PERBAIKAN) ---
                fig_o, ax_o = plt.subplots(figsize=(8, 3))
                fig_o.patch.set_facecolor('#0e1117')
                ax_o.set_facecolor('#0e1117')
                
                # Buat sumbu X menyesuaikan max data (bukan cuma 100)
                x_occ = np.linspace(0, max_occ_data * 1.5, 500)
                
                # Parameter segitiga occupancy (Sesuai fungsi calculate)
                o_low_limit = max_occ_data * 0.4
                o_mid_peak = max_occ_data * 0.5
                o_high_start = max_occ_data * 0.6
                
                y_o_low = get_triangular_y(x_occ, -1, 0, o_low_limit)
                y_o_mid = get_triangular_y(x_occ, max_occ_data*0.2, o_mid_peak, max_occ_data*0.8)
                y_o_hi = get_triangular_y(x_occ, o_high_start, max_occ_data, max_occ_data * 1.5)
                
                ax_o.plot(x_occ, y_o_low, label='Rendah', color='#1f77b4')
                ax_o.plot(x_occ, y_o_mid, label='Sedang', color='#ff7f0e')
                ax_o.plot(x_occ, y_o_hi, label='Tinggi', color='#2ca02c')
                
                # Garis Vertikal
                ax_o.axvline(pred_occ, color='white', linewidth=3, linestyle='-', label='Input Anda')
                
                ax_o.set_title(f"Variable: Occupancy (Value: {pred_occ:.1f})", color='white')
                ax_o.legend(facecolor='#262730', edgecolor='white', loc='center right').get_texts()[0].set_color("white")
                ax_o.tick_params(colors='white')
                st.pyplot(fig_o)

            with col_rule:
                st.markdown("**Rule Base**")
                rule_data = {
                    'Rule': ['R1', 'R2', 'R3', 'R4'],
                    'Cond': ['Occ Low', 'Occ&Flow Mid', 'Occ/Flow Hi', 'Occ Mid-Flow Low'],
                    'Out': ['LANCAR', 'PADAT', 'MACET', 'LANCAR']
                }
                st.table(pd.DataFrame(rule_data))
                
                st.markdown("---")
                st.markdown("**Hasil Logika:**")
                
                f_status_label = "LANCAR"
                color_score = "green"
                if f_score > 45: 
                    f_status_label = "PADAT"
                    color_score = "orange"
                if f_score > 75: 
                    f_status_label = "MACET"
                    color_score = "red"
                    
                st.markdown(f"<h1 style='color: {color_score};'>{f_score:.2f}</h1>", unsafe_allow_html=True)
                st.markdown(f"Status: **{f_status_label}**")

            # ==========================================
            #   BAGIAN 2: STATISTIK
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
                    
                st.dataframe(df.head(50), use_container_width=True)

        else:
            st.info("👈 Pilih Metode Input, lalu klik tombol 'Prediksi Sekarang'.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("👋 Silakan upload file CSV Traffic di sidebar sebelah kiri.")
