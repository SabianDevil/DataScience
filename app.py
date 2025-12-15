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

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2382/2382461.png", width=80)
st.sidebar.title("Sistem Prediksi Kemacetan")
st.sidebar.markdown("Powered by AI & Fuzzy Logic")

# --- 1. UPLOAD DATA ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --- FUNGSI FUZZY LOGIC MANUAL (SEGITIGA) ---
def triangular_membership(x, a, b, c):
    """Fungsi untuk menghitung derajat keanggotaan (bentuk segitiga)"""
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

def fuzzy_inference_system(flow_val, occ_val, max_flow, max_occ):
    # 1. FUZZIFICATION (Mengubah Angka jadi Derajat 0-1)
    
    # Range Flow
    f_low = triangular_membership(flow_val, -1, 0, max_flow * 0.4)
    f_med = triangular_membership(flow_val, max_flow * 0.2, max_flow * 0.5, max_flow * 0.8)
    f_high = triangular_membership(flow_val, max_flow * 0.6, max_flow, max_flow * 1.5)
    
    # Range Occupancy
    o_low = triangular_membership(occ_val, -1, 0, max_occ * 0.4)
    o_med = triangular_membership(occ_val, max_occ * 0.2, max_occ * 0.5, max_occ * 0.8)
    o_high = triangular_membership(occ_val, max_occ * 0.6, max_occ, max_occ * 1.5)

    # 2. RULE BASE EVALUATION (Mamdani Style - Sederhana)
    # Rules:
    # R1: Flow Low & Occ Low -> Lancar (Score 10)
    # R2: Flow Med & Occ Med -> Padat (Score 50)
    # R3: Flow High & Occ High -> Macet (Score 90)
    # ... dst (kombinasi lain)
    
    # Kita ambil max degree untuk setiap output category
    degree_lancar = max(
        min(f_low, o_low), 
        min(f_med, o_low), 
        min(f_low, o_med)
    )
    
    degree_padat = max(
        min(f_high, o_low), 
        min(f_med, o_med), 
        min(f_low, o_high)
    )
    
    degree_macet = max(
        min(f_high, o_med), 
        min(f_med, o_high), 
        min(f_high, o_high)
    )

    # 3. DEFUZZIFICATION (Metode Weighted Average)
    # Pusat massa: Lancar=20, Padat=50, Macet=80
    score_final = (degree_lancar * 20 + degree_padat * 50 + degree_macet * 80) / (degree_lancar + degree_padat + degree_macet + 1e-9)
    
    return score_final, (f_low, f_med, f_high), (o_low, o_med, o_high)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Pre-processing
        df['interval_index'] = np.arange(len(df))
        max_val = len(df)
        
        # Konfigurasi Data
        st.sidebar.subheader("⚙️ Konfigurasi Data")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        col_flow = st.sidebar.selectbox("Kolom Flow:", numeric_cols, index=0)
        idx_occ = 1 if len(numeric_cols) > 1 else 0
        col_occ = st.sidebar.selectbox("Kolom Occupancy:", numeric_cols, index=idx_occ)

        # Model AI
        st.sidebar.divider()
        st.sidebar.subheader("🧠 Pengaturan Model AI")
        model_option = st.sidebar.selectbox(
            "Pilih Model AI (Regresi):",
            ("Polynomial Regression", "Decision Tree", "Linear Regression")
        )

        # Input Waktu
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Metode Input")
        input_method = st.sidebar.radio("Pilih Cara Input:", ["Input Detik", "Input Jam"])

        final_input_value = 0
        display_time_str = ""

        if input_method == "Input Detik":
            input_detik = st.sidebar.number_input(f"Detik (0 - {max_val})", 0, max_val, int(max_val/2))
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

        if st.sidebar.button("🚀 Prediksi Sekarang", type="primary"):
            
            # --- 1. PROSES AI (PREDIKSI ANGKA) ---
            X = df[['interval_index']]
            y_flow = df[col_flow]
            y_occ = df[col_occ]
            
            # Fitting Model
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

            # --- 2. PROSES FUZZY LOGIC (INTERPRETASI HASIL) ---
            max_flow_data = df[col_flow].max()
            max_occ_data = df[col_occ].max()
            
            fuzzy_score, flow_degrees, occ_degrees = fuzzy_inference_system(pred_flow, pred_occ, max_flow_data, max_occ_data)
            
            # Tentukan Text Status dari Skor Fuzzy
            status_text = "LANCAR"
            status_color = "green"
            if fuzzy_score > 60:
                status_text = "MACET"
                status_color = "red"
            elif fuzzy_score > 35:
                status_text = "PADAT"
                status_color = "orange"

            # --- TAMPILAN DASHBOARD ---
            st.title("Sistem Prediksi Kemacetan Munich")
            
            # Kartu Atas
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"### Status:\n## :{status_color}[{status_text}]")
            c2.metric("Prediksi Flow", f"{pred_flow:.1f}")
            c3.metric("Prediksi Occupancy", f"{pred_occ:.2f}")
            c4.metric("Skor Kemacetan (Fuzzy)", f"{fuzzy_score:.2f}")

            st.divider()

            # --- BAGIAN ANALISIS FUZZY LOGIC (SEPERTI TEMAN ANDA) ---
            st.subheader("🧠 Analisis Fuzzy Logic")
            col_fuzzy_viz, col_fuzzy_rule = st.columns([2, 1])

            with col_fuzzy_viz:
                st.write("**Visualisasi Membership Function:**")
                
                # Plot Segitiga Fuzzy Flow
                fig_fuz, ax_fuz = plt.subplots(figsize=(8, 4))
                x_axis = np.linspace(0, max_flow_data, 100)
                
                # Gambar segitiga Low, Med, High
                y_low = [triangular_membership(x, -1, 0, max_flow_data*0.4) for x in x_axis]
                y_med = [triangular_membership(x, max_flow_data*0.2, max_flow_data*0.5, max_flow_data*0.8) for x in x_axis]
                y_high = [triangular_membership(x, max_flow_data*0.6, max_flow_data, max_flow_data*1.5) for x in x_axis]
                
                ax_fuz.plot(x_axis, y_low, label='Low (Rendah)', color='green')
                ax_fuz.plot(x_axis, y_med, label='Medium (Sedang)', color='orange')
                ax_fuz.plot(x_axis, y_high, label='High (Tinggi)', color='red')
                
                # Garis posisi data saat ini
                ax_fuz.axvline(pred_flow, color='white', linestyle='--', linewidth=2, label='Prediksi Saat Ini')
                ax_fuz.fill_between(x_axis, 0, 1, where=(x_axis >= pred_flow-5) & (x_axis <= pred_flow+5), color='white', alpha=0.3)
                
                ax_fuz.set_title(f"Posisi Flow dalam Logika Fuzzy ({pred_flow:.0f})")
                ax_fuz.legend()
                ax_fuz.set_facecolor('#1e1e1e')
                fig_fuz.patch.set_facecolor('#0e1117') # Background Streamlit gelap
                
                # Ubah warna text jadi putih biar kelihatan di dark mode
                ax_fuz.tick_params(colors='white')
                ax_fuz.xaxis.label.set_color('white')
                ax_fuz.yaxis.label.set_color('white')
                ax_fuz.title.set_color('white')
                leg = ax_fuz.legend(facecolor='#262730', edgecolor='white')
                for text in leg.get_texts(): text.set_color("white")
                
                st.pyplot(fig_fuz)

            with col_fuzzy_rule:
                st.write("**Rule Base & Hasil:**")
                st.markdown(f"""
                <div style="background-color:#262730; padding:15px; border-radius:10px;">
                    <h2 style="color:white; margin:0;">{fuzzy_score:.2f}</h2>
                    <p style="color:gray;">Skor Kemacetan (0-100)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tabel Rules Sederhana
                rules_data = {
                    'Rule': ['R1', 'R2', 'R3', 'R4', 'R5'],
                    'Flow': ['Low', 'Med', 'High', 'Low', 'High'],
                    'Occ': ['Low', 'Med', 'High', 'High', 'Low'],
                    'Output': ['Lancar', 'Padat', 'Macet', 'Macet', 'Padat']
                }
                st.dataframe(pd.DataFrame(rules_data), hide_index=True)

            st.divider()

            # --- VISUALISASI DATA HISTORIS (YANG LAMA) ---
            st.subheader("📈 Posisi Prediksi pada Data Historis")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.patch.set_alpha(0.0)

            # Grafik Flow
            ax1.plot(df['interval_index'], df[col_flow], color='white', alpha=0.1)
            # Garis AI
            y_ai_flow = model_flow.predict(X)
            ax1.plot(df['interval_index'], y_ai_flow, color='cyan', alpha=0.5, label=f'Pola {model_option}')
            
            ax1.scatter(final_input_value, pred_flow, color='red', s=150, zorder=10, label=display_time_str)
            ax1.set_title("Grafik Flow", color='white')
            ax1.set_facecolor('black')
            ax1.legend(facecolor='#262730', labelcolor='white')
            ax1.grid(False)
            ax1.tick_params(colors='white')

            # Grafik Occ
            ax2.plot(df['interval_index'], df[col_occ], color='white', alpha=0.1)
            # Garis AI
            y_ai_occ = model_occ.predict(X)
            ax2.plot(df['interval_index'], y_ai_occ, color='cyan', alpha=0.5, label=f'Pola {model_option}')

            ax2.scatter(final_input_value, pred_occ, color='red', s=150, zorder=10, label=display_time_str)
            ax2.set_title("Grafik Occupancy", color='white')
            ax2.set_facecolor('black')
            ax2.legend(facecolor='#262730', labelcolor='white')
            ax2.grid(False)
            ax2.tick_params(colors='white')

            st.pyplot(fig)

        else:
            st.info("👈 Klik 'Prediksi Sekarang' di sidebar.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👋 Silakan upload file CSV.")
