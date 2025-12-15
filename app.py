import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="Posisi Prediksi Data", layout="wide")

# Mengatur tema Matplotlib menjadi Gelap
plt.style.use('dark_background')

st.title("📈 Posisi Prediksi pada Data Historis")
st.markdown("Visualisasi posisi data saat ini terhadap keseluruhan data historis.")

# --- SIDEBAR ---
st.sidebar.header("📂 Panel Kontrol")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # 1. BACA DATA
        df = pd.read_csv(uploaded_file)
        
        # 2. KONFIGURASI KOLOM (DIKEMBALIKAN + Kolom Waktu)
        st.sidebar.subheader("⚙️ Konfigurasi Kolom")
        all_cols = df.columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # A. Pilih Kolom Waktu (Wajib ada agar input Jam bisa bekerja)
        # Cari default yang namanya mirip 'date' atau 'time'
        default_time = next((i for i, c in enumerate(all_cols) if 'date' in c.lower() or 'time' in c.lower()), 0)
        col_time = st.sidebar.selectbox("Pilih Kolom Waktu/Tanggal:", all_cols, index=default_time)

        # Proses Convert Waktu di background
        df['datetime_convert'] = pd.to_datetime(df[col_time], errors='coerce')
        df = df.dropna(subset=['datetime_convert']) # Hapus jika ada format waktu error

        # B. Pilih Kolom Flow & Occupancy (Manual seperti permintaan)
        # Filter agar kolom waktu hasil convert tidak muncul di pilihan angka
        numeric_cols_clean = [c for c in numeric_cols if c != 'datetime_convert']
        
        col_flow = st.sidebar.selectbox("Pilih Kolom Flow:", numeric_cols_clean, index=0)
        # Coba cari default index untuk occupancy agar user ga repot
        idx_occ = 1 if len(numeric_cols_clean) > 1 else 0
        col_occ = st.sidebar.selectbox("Pilih Kolom Occupancy:", numeric_cols_clean, index=idx_occ)

        # 3. METODE INPUT (DIGANTI MENJADI WAKTU)
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Metode Input (Waktu)")
        
        # Ambil waktu awal dari data untuk default value
        min_time = df['datetime_convert'].iloc[0].time()
        
        # Input Jam/Menit/Detik
        waktu_input = st.sidebar.time_input("Masukkan Waktu:", value=min_time, step=1)
        
        # --- LOGIKA PENCARIAN DATA (Smart Match) ---
        # 1. Ambil tanggal referensi dari data
        tanggal_ref = df['datetime_convert'].iloc[0].date()
        # 2. Gabungkan tanggal + waktu input user
        target_ts = pd.Timestamp.combine(tanggal_ref, waktu_input)
        # 3. Cari index data yang jam-nya paling mendekati input user
        posisi_saat_ini = (df['datetime_convert'] - target_ts).abs().idxmin()
        
        # Ambil string waktu aktual yang ketemu di data
        waktu_aktual_str = df.loc[posisi_saat_ini, 'datetime_convert'].strftime("%H:%M:%S")

        # 4. THRESHOLD SLIDERS (Tetap Ada)
        st.sidebar.divider()
        st.sidebar.subheader("⚠️ Batas Ambang (Threshold)")
        
        max_flow = int(df[col_flow].max())
        max_occ = float(df[col_occ].max())
        
        threshold_flow = st.sidebar.slider("Threshold Macet (Flow)", 0, max_flow, int(max_flow * 0.7))
        threshold_occ = st.sidebar.slider("Threshold Macet (Occupancy)", 0.0, max_occ, max_occ * 0.3)

        # --- VISUALISASI MATPLOTLIB ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # === GRAFIK 1: FLOW ===
        ax1.plot(df.index, df[col_flow], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_flow_saat_ini = df.iloc[posisi_saat_ini][col_flow]
        # Label titik merah menampilkan Jam
        ax1.scatter(posisi_saat_ini, nilai_flow_saat_ini, color='red', s=100, zorder=5, label=f'Pukul {waktu_aktual_str}')
        ax1.axhline(y=threshold_flow, color='#ffcc00', linestyle='--', linewidth=1.5, label='Threshold Macet')
        
        ax1.set_title("Flow Historis dengan Prediksi Saat Ini", fontsize=14, pad=15)
        ax1.set_ylabel("Flow")
        ax1.set_xlabel("interval")
        ax1.legend(loc='upper right', frameon=True, facecolor='white', labelcolor='black')
        ax1.grid(False)

        # === GRAFIK 2: OCCUPANCY ===
        ax2.plot(df.index, df[col_occ], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_occ_saat_ini = df.iloc[posisi_saat_ini][col_occ]
        ax2.scatter(posisi_saat_ini, nilai_occ_saat_ini, color='red', s=100, zorder=5, label=f'Pukul {waktu_aktual_str}')
        ax2.axhline(y=threshold_occ, color='#ffcc00', linestyle='--', linewidth=1.5, label='Threshold Macet')
        
        ax2.set_title("Occupancy Historis dengan Prediksi Saat Ini", fontsize=14, pad=15)
        ax2.set_ylabel("Occupancy (%)") 
        ax2.set_xlabel("interval")
        ax2.legend(loc='upper right', frameon=True, facecolor='white', labelcolor='black')
        ax2.grid(False)

        st.pyplot(fig)

        # --- LOGIKA STATUS & PREDIKSI ---
        st.divider()
        st.subheader("🏁 Hasil Analisis Status")

        # Logika Penentuan Status
        status_text = "LANCAR 🟢"
        warna_pesan = "success"

        if nilai_occ_saat_ini > threshold_occ:
            status_text = "MACET (Occupancy Tinggi) 🔴"
            warna_pesan = "error"
        elif nilai_flow_saat_ini > threshold_flow:
            status_text = "PADAT (Flow Tinggi) 🟠"
            warna_pesan = "warning"

        # Tampilkan Status
        st.write(f"**Analisis Pukul {waktu_aktual_str}:**")
        if warna_pesan == "error":
            st.error(f"Status Lalu Lintas: **{status_text}**")
        elif warna_pesan == "warning":
            st.warning(f"Status Lalu Lintas: **{status_text}**")
        else:
            st.success(f"Status Lalu Lintas: **{status_text}**")

        # Tampilkan Angka Prediksi
        c1, c2, c3 = st.columns(3)
        
        c1.metric("Waktu Terpilih", waktu_aktual_str)
        
        c2.metric(
            label="Prediksi Flow", 
            value=f"{nilai_flow_saat_ini:.2f}", 
            delta=f"{nilai_flow_saat_ini - threshold_flow:.2f} dari batas",
            delta_color="inverse" 
        )
        
        c3.metric(
            label="Prediksi Occupancy", 
            value=f"{nilai_occ_saat_ini:.2f}%", # Persen
            delta=f"{nilai_occ_saat_ini - threshold_occ:.2f}% dari batas",
            delta_color="inverse"
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.info("Tips: Pastikan Kolom Waktu yang dipilih benar formatnya (Tanggal/Jam).")

else:
    st.info("👋 Silakan upload file CSV Birmingham/Traffic Anda.")
