import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# --- FUNGSI LOGIKA MATEMATIKA MANUAL (NUMPY) ---
def get_triangular_y(x_array, a, b, c):
    """Membuat array Y untuk plotting bentuk segitiga"""
    y = []
    for x in x_array:
        # Rumus segitiga: max(min((x-a)/(b-a), (c-x)/(c-b)), 0)
        # Ditambah 1e-9 untuk menghindari pembagian dengan nol
        val = max(0, min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))
        y.append(val)
    return np.array(y)

def triangular_membership(x, a, b, c):
    """Menghitung derajat keanggotaan satu titik"""
    return max(0, min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def calculate_fuzzy_logic(flow_val, occ_val, max_flow, max_occ):
    # --- 1. FUZZIFICATION (Input -> Derajat 0-1) ---
    
    # FLOW (Dinamis ikut max data)
    f_mid = max_flow * 0.5
    f_hi_peak = max_flow
    
    flow_low = triangular_membership(flow_val, -1, 0, f_mid)
    flow_mid = triangular_membership(flow_val, 0, f_mid, f_hi_peak)
    flow_hi = triangular_membership(flow_val, f_mid, f_hi_peak, f_hi_peak * 1.5)
    
    # OCCUPANCY (Dinamis ikut max data)
    o_mid_peak = max_occ * 0.5
    
    occ_low = triangular_membership(occ_val, -1, 0, max_occ * 0.4)
    occ_mid = triangular_membership(occ_val, max_occ * 0.2, o_mid_peak, max_occ * 0.8)
    occ_hi = triangular_membership(occ_val, max_occ * 0.6, max_occ, max_occ * 1.5)
    
    # --- 2. INFERENCE (Menerapkan Rules) ---
    # Rule 1: Occ Low -> LANCAR
    alpha_lancar_1 = occ_low
    
    # Rule 2: Occ Mid & Flow Mid -> PADAT
    alpha_padat = min(occ_mid, flow_mid)
    
    # Rule 3: Occ Hi ATAU Flow Hi -> MACET
    alpha_macet = max(occ_hi, flow_hi)
    
    # Rule 4: Occ Mid tapi Flow Low -> LANCAR
    alpha_lancar_2 = min(occ_mid, flow_low)
    
    # Gabungkan Rule untuk Output yang sama (Max aggregation)
    # Total kekuatan untuk Lancar, Padat, Macet
    act_lancar = max(alpha_lancar_1, alpha_lancar_2)
    act_padat = alpha_padat
    act_macet = alpha_macet
    
    # --- 3. DEFUZZIFICATION (Centroid) ---
    # Titik pusat (Singleton) untuk perhitungan skor sederhana
    z_lancar = 20
    z_padat = 60
    z_macet = 90
    
    pembilang = (act_lancar * z_lancar) + (act_padat * z_padat) + (act_macet * z_macet)
    penyebut = act_lancar + act_padat + act_macet
    
    score = 0 if penyebut == 0 else pembilang / penyebut
    
    # Kembalikan semua data untuk visualisasi "Arsir"
    return {
        'score': score,
        'flow_mfs': (flow_low, flow_mid, flow_hi),
        'occ_mfs': (occ_low, occ_mid, occ_hi),
        'activations': (act_lancar, act_padat, act_macet) # Ini kunci gambar arsir
    }

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Prediksi Kemacetan", layout="wide")
plt.style.use('dark_background')

# --- SIDEBAR ---
st.sidebar.title("🚦 Sistem Prediksi Kemacetan")
st.sidebar.markdown("Dashboard AI untuk analisis & prediksi lalu lintas.")

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

        st.sidebar.divider()
        st.sidebar.subheader("🧠 Model AI")
        model_option = st.sidebar.selectbox("Pilih Algoritma:", ("Polynomial Regression (Rekomen)", "Decision Tree", "Linear Regression"))

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

        if st.sidebar.button("🚀 Prediksi Sekarang", type="primary"):
            
            # --- TRAINING ---
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
            
            input_df = pd.DataFrame({'interval_index': [final_input_value]})
            pred_flow = model_flow.predict(input_df)[0]
            pred_occ = model_occ.predict(input_df)[0]
            
            thresh_flow = df[col_flow].max() * 0.7
            thresh_occ = df[col_occ].max() * 0.3
            
            # --- DASHBOARD ---
            st.title(f"Sistem Prediksi Kemacetan")
            
            c_stat, c_flow, c_occ, c_int, c_mod = st.columns(5)
            
            # Hitung Logika Fuzzy Dulu untuk status
            max_flow_data = df[col_flow].max()
            max_occ_data = df[col_occ].max()
            
            fuzzy_res = calculate_fuzzy_logic(pred_flow, pred_occ, max_flow_data, max_occ_data)
            f_score = fuzzy_res['score']
            
            status_text = "LANCAR"
            status_color = "green"
            if f_score > 45: 
                status_text = "PADAT"
                status_color = "orange"
            if f_score > 75: 
                status_text = "MACET"
                status_color = "red"

            c_stat.markdown(f"### Status:\n## :{status_color}[{status_text}]")
            c_flow.metric("Prediksi Flow", f"{pred_flow:.1f}")
            c_occ.metric("Prediksi Occupancy", f"{pred_occ:.1f}")
            c_int.metric("Interval Input", final_input_value)
            c_mod.markdown(f"**Model:**\n{model_option}")
            
            st.divider()

            # --- GRAFIK PREDIKSI ---
            st.subheader("📈 Visualisasi Prediksi")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.patch.set_facecolor('black')
            fig.patch.set_alpha(0.0)

            ax1.plot(df['interval_index'], df[col_flow], color='white', alpha=0.1, label='Data Asli')
            ax1.plot(df['interval_index'], model_flow.predict(X), color='cyan', alpha=0.5, linewidth=1, label=f'Pola AI')
            ax1.scatter(final_input_value, pred_flow, color='red', s=200, zorder=10, label=display_time_str, edgecolors='white')
            ax1.axhline(thresh_flow, color='#ffcc00', linestyle='--', label='Batas Padat')
            ax1.set_title("Prediksi Flow", color='white')
            ax1.set_facecolor('black')
            ax1.legend(facecolor='#262730', edgecolor='white').get_texts()[0].set_color("white")
            ax1.tick_params(colors='white')

            ax2.plot(df['interval_index'], df[col_occ], color='white', alpha=0.1, label='Data Asli')
            ax2.plot(df['interval_index'], model_occ.predict(X), color='cyan', alpha=0.5, linewidth=1, label=f'Pola AI')
            ax2.scatter(final_input_value, pred_occ, color='red', s=200, zorder=10, label=display_time_str, edgecolors='white')
            ax2.axhline(thresh_occ, color='#ffcc00', linestyle='--', label='Batas Macet')
            ax2.set_title("Prediksi Occupancy", color='white')
            ax2.set_facecolor('black')
            ax2.legend(facecolor='#262730', edgecolor='white').get_texts()[0].set_color("white")
            ax2.tick_params(colors='white')

            st.pyplot(fig)

            # --- BAGIAN FUZZY LOGIC LENGKAP (DENGAN ARSIR) ---
            st.divider()
            st.subheader("🤖 Analisis Fuzzy Logic & Area Keputusan")

            col_viz, col_result = st.columns([3, 1])

            with col_viz:
                # Kita buat 3 subplot ke bawah: Flow, Occupancy, dan OUTPUT (Hasil Arsir)
                fig_fuz, (ax_f, ax_o, ax_out) = plt.subplots(3, 1, figsize=(8, 10))
                fig_fuz.subplots_adjust(hspace=0.6)
                fig_fuz.patch.set_facecolor('#0e1117')
                
                # 1. PLOT FLOW
                x_flow = np.linspace(0, max_flow_data * 1.2, 500) # Range dipersempit biar garis kelihatan gerak
                y_f_low = get_triangular_y(x_flow, -1, 0, max_flow_data*0.5)
                y_f_mid = get_triangular_y(x_flow, 0, max_flow_data*0.5, max_flow_data)
                y_f_hi = get_triangular_y(x_flow, max_flow_data*0.5, max_flow_data, max_flow_data*1.5)
                
                ax_f.plot(x_flow, y_f_low, label='Rendah', color='#1f77b4')
                ax_f.plot(x_flow, y_f_mid, label='Sedang', color='#ff7f0e')
                ax_f.plot(x_flow, y_f_hi, label='Tinggi', color='#2ca02c')
                ax_f.axvline(pred_flow, color='white', linewidth=3, linestyle='-', label='Input')
                ax_f.set_title(f"Input 1: Flow (Nilai: {pred_flow:.0f})", color='white', loc='left')
                ax_f.set_facecolor('#0e1117')
                ax_f.tick_params(colors='white')
                ax_f.spines['bottom'].set_color('white')
                ax_f.spines['left'].set_color('white')
                
                # 2. PLOT OCCUPANCY
                x_occ = np.linspace(0, max_occ_data * 1.2, 500)
                y_o_low = get_triangular_y(x_occ, -1, 0, max_occ_data*0.4)
                y_o_mid = get_triangular_y(x_occ, max_occ_data*0.2, max_occ_data*0.5, max_occ_data*0.8)
                y_o_hi = get_triangular_y(x_occ, max_occ_data*0.6, max_occ_data, max_occ_data*1.5)
                
                ax_o.plot(x_occ, y_o_low, label='Rendah', color='#1f77b4')
                ax_o.plot(x_occ, y_o_mid, label='Sedang', color='#ff7f0e')
                ax_o.plot(x_occ, y_o_hi, label='Tinggi', color='#2ca02c')
                ax_o.axvline(pred_occ, color='white', linewidth=3, linestyle='-', label='Input')
                ax_o.set_title(f"Input 2: Occupancy (Nilai: {pred_occ:.1f})", color='white', loc='left')
                ax_o.set_facecolor('#0e1117')
                ax_o.tick_params(colors='white')
                ax_o.spines['bottom'].set_color('white')
                ax_o.spines['left'].set_color('white')

                # 3. PLOT OUTPUT (AREA ARSIR AGREGASI)
                # Ambil nilai aktivasi dari fungsi fuzzy
                act_l, act_p, act_m = fuzzy_res['activations']
                
                x_out = np.linspace(0, 100, 500)
                
                # Buat bentuk dasar segitiga Output (Lancar, Padat, Macet)
                # Output Scale: 0-100
                y_out_l = get_triangular_y(x_out, -20, 0, 40)
                y_out_p = get_triangular_y(x_out, 20, 50, 80)
                y_out_m = get_triangular_y(x_out, 60, 100, 120)
                
                # Potong segitiga berdasarkan Aktivasi (Clipping)
                y_out_l_clip = np.fmin(act_l, y_out_l)
                y_out_p_clip = np.fmin(act_p, y_out_p)
                y_out_m_clip = np.fmin(act_m, y_out_m)
                
                # Gabungkan semua potongan (Union / Aggregation)
                y_aggregated = np.fmax(y_out_l_clip, np.fmax(y_out_p_clip, y_out_m_clip))
                
                # Gambar garis tipis bentuk aslinya (sbg referensi)
                ax_out.plot(x_out, y_out_l, '--', color='#1f77b4', alpha=0.3)
                ax_out.plot(x_out, y_out_p, '--', color='#ff7f0e', alpha=0.3)
                ax_out.plot(x_out, y_out_m, '--', color='#2ca02c', alpha=0.3)
                
                # GAMBAR ARSIR (FILLED AREA)
                ax_out.fill_between(x_out, 0, y_aggregated, facecolor='cyan', alpha=0.5, label='Area Keputusan')
                
                # Garis Tegas Hasil Akhir (Centroid)
                ax_out.axvline(f_score, color='red', linewidth=3, linestyle='-', label='Score Akhir')
                
                ax_out.set_title(f"Output: Tingkat Kemacetan (Score: {f_score:.2f})", color='white', loc='left')
                ax_out.set_facecolor('#0e1117')
                ax_out.tick_params(colors='white')
                ax_out.spines['bottom'].set_color('white')
                ax_out.spines['left'].set_color('white')
                ax_out.legend(facecolor='#262730', edgecolor='white', loc='upper right').get_texts()[0].set_color("white")

                st.pyplot(fig_fuz)

            with col_result:
                st.markdown("### Detail Nilai")
                st.info("Grafik paling bawah adalah 'Area Keputusan'. Daerah berwarna Cyan adalah hasil penggabungan semua aturan.")
                
                st.markdown(f"""
                **Nilai Input:**
                * Flow: `{pred_flow:.1f}`
                * Occ: `{pred_occ:.1f}`
                
                **Aktivasi Aturan:**
                * Lancar: `{act_l:.2f}` (Kekuatan)
                * Padat: `{act_p:.2f}` (Kekuatan)
                * Macet: `{act_m:.2f}` (Kekuatan)
                
                **Hasil Akhir:**
                # {f_score:.2f}
                Status: **{status_text}**
                """)

            # --- STATISTIK ---
            st.divider()
            with st.expander("📊 Lihat Statistik Data Mentah"):
                st.dataframe(df[[col_flow, col_occ]].describe().T, use_container_width=True)
                st.dataframe(df.head(50), use_container_width=True)

        else:
            st.info("👈 Pilih Metode Input, lalu klik tombol 'Prediksi Sekarang'.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("👋 Silakan upload file CSV Traffic di sidebar sebelah kiri.")
