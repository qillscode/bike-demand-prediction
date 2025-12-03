import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Sewa Sepeda", page_icon="🚲")

# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Muat model yang sudah dilatih
try:
    model = load_model('bike_demand.pkl')
except FileNotFoundError:
    st.error("File model 'bike_demand.pkl' tidak ditemukan.")
    st.stop()

# --- Tampilan Aplikasi ---
st.title('Prediksi Permintaan Sepeda Berbagi 🚲')

# --- Input dari Pengguna di Sidebar ---
st.sidebar.header('Masukkan Kondisi Prediksi:')

# Dictionary mapping
season_map = {1: 'Musim Semi', 2: 'Musim Panas', 3: 'Musim Gugur', 4: 'Musim Dingin'}
weather_map = {1: 'Cerah', 2: 'Berkabut/Berawan', 3: 'Hujan Ringan', 4: 'Hujan Lebat'}
weekday_map = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}

# Input utama yang simpel
hr = st.sidebar.slider('Jam', 0, 23, 17)
temp = st.sidebar.slider('Suhu (°C)', -10.0, 40.0, 28.0)
hum = st.sidebar.slider('Kelembapan (%)', 0.0, 100.0, 60.0)
windspeed = st.sidebar.slider('Kecepatan Angin (km/h)', 0.0, 70.0, 15.0)
season_label = st.sidebar.selectbox('Musim', options=list(season_map.values()))
weather_label = st.sidebar.selectbox('Kondisi Cuaca', options=list(weather_map.values()))
weekday_label = st.sidebar.selectbox('Hari', options=list(weekday_map.values()), index=datetime.today().weekday())

# Logika otomatis berdasarkan nama hari
st.sidebar.markdown("---")
day_of_week_num = [k for k, v in weekday_map.items() if v == weekday_label][0]
is_weekend = day_of_week_num >= 5

if is_weekend:
    workingday_auto = 0
    holiday_auto = 1
    st.sidebar.info("Status Otomatis: **Bukan Hari Kerja**")
else:
    workingday_auto = 1
    holiday_auto = 0
    st.sidebar.info("Status Otomatis: **Hari Kerja**")


# --- Tombol Prediksi ---
if st.button('Prediksi Permintaan', use_container_width=True, type="primary"):
    season = [k for k, v in season_map.items() if v == season_label][0]
    weathersit = [k for k, v in weather_map.items() if v == weather_label][0]
    
    # Feature Engineering
    rush_hours_list = [7, 8, 9, 16, 17, 18, 19]
    rush_hours_val = 1 if hr in rush_hours_list and workingday_auto == 1 else 0
    if 5 <= hr < 12: part_of_day_val = 1
    elif 12 <= hr < 17: part_of_day_val = 2
    elif 17 <= hr < 22: part_of_day_val = 3
    else: part_of_day_val = 4
    day_interaction_val = part_of_day_val * 10 + workingday_auto

    # Buat DataFrame dari input
    input_data = pd.DataFrame({
        'season': [season], 'hr': [hr], 'holiday': [holiday_auto],
        'workingday': [workingday_auto], 'weathersit': [weathersit],
        'temp': [temp / 41.0], 'hum': [hum / 100.0], 'windspeed': [windspeed / 67.0],
        'month': [datetime.now().month], 'day_of_week': [day_of_week_num], 
        'year': [datetime.now().year - 2011],
        'rush_hours': [rush_hours_val],      # <-- INI YANG DIPERBAIKI
        'part_of_day': [part_of_day_val],
        'day_interaction': [day_interaction_val]
    })

    # Prediksi
    prediction = model.predict(input_data)
    st.header('Hasil Prediksi')
    st.metric(label=f"Prediksi untuk hari {weekday_label}", value=f"~ {int(prediction[0])} sepeda")
    st.info("Catatan: Prediksi ini dihasilkan oleh model LightGBM yang telah dioptimalkan dengan feature engineering.")

