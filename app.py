import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Posisi Prediksi Data", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema Gelap untuk Grafik
plt.style.use('dark_background')

st.title("📈 Posisi Prediksi pada Data Historis")
st.markdown("Visualisasi posisi data saat ini terhadap keseluruhan data historis.")

# --- 2. SIDEBAR: UPLOAD ---
st.sidebar.header("📂 Panel Kontrol")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Baca Data
        df = pd.read_csv(uploaded_file)
        
        # --- 3. AUTO-DETECT KOLOM (Tanpa Pilihan User) ---
        # Logika otomatis mencari kolom yang tepat
        cols = df.columns
        
        # A. Cari Kolom Waktu
        # Mencari kolom yang mengandung kata 'date', 'time', atau 'tgl'
        col_time = next((c for c in cols if 'date' in c.lower() or 'time' in c.lower() or 'tgl' in c.lower()), cols[0])
        
        # B. Cari Kolom Angka
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # C. Cari Kolom Flow (Prioritas nama: flow, volume, jumlah)
        col_flow = next((c for c in numeric_cols if 'flow' in c.lower() or 'vol' in c.lower()), numeric_cols[0] if numeric_cols else None)
        
        # D. Cari Kolom Occupancy (Prioritas nama: occ, speed, kepadatan)
        # Pastikan tidak memilih kolom yang sama dengan flow
        remaining_cols = [c for c in numeric_cols if c != col_flow]
        col_occ = next((c for c in remaining_cols if 'occ' in c.lower() or 'speed' in c.lower()), remaining_cols[0] if remaining_cols else None)

        # Proses Konversi Waktu (Di belakang layar)
        df['datetime_convert'] = pd.to_datetime(df[col_time], errors='coerce')
        df = df.dropna(subset=['datetime_convert'])
        
        # Tampilkan info kecil kolom apa yang terdeteksi (Opsional, agar user tau)
        st.caption(f"ℹ️ **Auto-Detect:** Waktu='{col_time}', Flow='{col_flow}', Occupancy='{col_occ}'")

        # --- 4. INPUT WAKTU (JAM/MENIT/DETIK) ---
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Input Waktu Simulasi")
        
        min_time = df['datetime_convert'].min().time()
        
        # Input Waktu
        waktu_input = st.sidebar.time_input("Masukkan Waktu (Jam:Menit:Detik):", value=min_time, step=1)
        
        # Logika Pencarian Data Terdekat (Nearest Match)
        tanggal_referensi = df['datetime_convert'].iloc[0].date()
        target_timestamp = pd.Timestamp.combine(tanggal_referensi, waktu_input)
        posisi_saat_ini = (df['datetime_convert'] - target_timestamp).abs().idxmin()

        # --- 5. SLIDER THRESHOLD ---
        st.sidebar.divider()
        st.sidebar.subheader("⚠️ Batas Ambang (Threshold)")
        
        max_flow = int(df[col_flow].max())
        max_occ = float(df[col_occ].max())
        
        threshold_flow = st.sidebar.slider("Batas Macet (Flow)", 0, max_flow, int(max_flow * 0.7))
        threshold_occ = st.sidebar.slider("Batas Macet (Occupancy)", 0.0, max_occ, max_occ * 0.3)

        # --- 6. VISUALISASI ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Background Grafik Transparan
        fig.patch.set_facecolor('black')
        fig.patch.set_alpha(0.0)

        # === GRAFIK 1: FLOW ===
        ax1.plot(df.index, df[col_flow], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_flow_saat_ini = df.iloc[posisi_saat_ini][col_flow]
        ax1.scatter(posisi_saat_ini, nilai_flow_saat_ini, color='red', s=150, zorder=5, label='Input Waktu', edgecolors='white')
        ax1.axhline(y=threshold_flow, color='#ffcc00', linestyle='--', linewidth=1.5, label='Batas Macet')
        
        ax1.set_title(f"Grafik Flow", fontsize=14, color='white', pad=10)
        ax1.set_ylabel("Flow", color='white')
        ax1.tick_params(colors='white')
        ax1.grid(False)
        ax1.set_facecolor('black')
        leg1 = ax1.legend(facecolor='#262730', edgecolor='white')
        for text in leg1.get_texts(): text.set_color("white")

        # === GRAFIK 2: OCCUPANCY ===
        ax2.plot(df.index, df[col_occ], color='white', alpha=0.15, label='Data Historis', linewidth=1)
        nilai_occ_saat_ini = df.iloc[posisi_saat_ini][col_occ]
        ax2.scatter(posisi_saat_ini, nilai_occ_saat_ini, color='red', s=150, zorder=5, label='Input Waktu', edgecolors='white')
        ax2.axhline(y=threshold_occ, color='#ffcc00', linestyle='--', linewidth=1.5, label='Batas Macet')
        
        ax2.set_title(f"Grafik Occupancy", fontsize=14, color='white', pad=10)
        ax2.set_ylabel("Occupancy (%)", color='white')
        ax2.tick_params(colors='white')
        ax2.grid(False)
        ax2.set_facecolor('black')
        leg2 = ax2.legend(facecolor='#262730', edgecolor='white')
        for text in leg2.get_texts(): text.set_color("white")

        st.pyplot(fig)

        # --- 7. HASIL ANALISIS STATUS ---
        st.divider()
        st.subheader("🏁 Hasil Analisis Status")

        # Ambil waktu aktual dari data
        waktu_aktual_data = df.iloc[posisi_saat_ini]['datetime_convert'].strftime("%H:%M:%S")

        # Logika Status
        status_text = "LANCAR 🟢"
        warna_pesan = "success"

        if nilai_occ_saat_ini > threshold_occ:
            status_text = "MACET (Occupancy Tinggi) 🔴"
            warna_pesan = "error"
        elif nilai_flow_saat_ini > threshold_flow:
            status_text = "PADAT (Flow Tinggi) 🟠"
            warna_pesan = "warning"

        if warna_pesan == "error":
            st.error(f"Pukul **{waktu_aktual_data}** -> {status_text}")
        elif warna_pesan == "warning":
            st.warning(f"Pukul **{waktu_aktual_data}** -> {status_text}")
        else:
            st.success(f"Pukul **{waktu_aktual_data}** -> {status_text}")

        # Metric Cards
        c1, c2, c3 = st.columns(3)
        
        c1.metric("Waktu Data Terpilih", waktu_aktual_data)
        
        c2.metric(
            label="Prediksi Flow", 
            value=f"{nilai_flow_saat_ini:.0f}", 
            delta=f"{nilai_flow_saat_ini - threshold_flow:.0f} dari batas",
            delta_color="inverse" 
        )
        
        c3.metric(
            label="Prediksi Occupancy", 
            value=f"{nilai_occ_saat_ini:.2f}%", 
            delta=f"{nilai_occ_saat_ini - threshold_occ:.2f}% dari batas",
            delta_color="inverse"
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan pembacaan data: {e}")
        st.warning("Pastikan file CSV memiliki kolom: Waktu, Flow, dan Occupancy.")

else:
    st.info("👋 Silakan upload file CSV Traffic Anda.")
