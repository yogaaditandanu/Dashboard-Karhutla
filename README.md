# 🔥 Dashboard Karhutla Indonesia (Real-Time)

Dashboard interaktif untuk memantau titik panas (hotspot) Kebakaran Hutan dan Lahan (Karhutla) di Indonesia secara *real-time*. Aplikasi ini menarik data *Near Real-Time* (NRT) dari API satelit NASA FIRMS (sensor VIIRS) dan menerapkan model *Machine Learning* untuk mendeteksi pola spasial, anomali, dan tingkat risiko kebakaran.

## ✨ Fitur Utama

* **Data Satelit Real-Time**: Terintegrasi otomatis dengan API NASA FIRMS untuk mengambil data titik panas selama 5 hari terakhir di area kordinat Indonesia.
* **Analitik Machine Learning**:
  * 🧩 **DBSCAN (Density-Based Spatial Clustering)**: Mengelompokkan titik-titik panas yang saling berdekatan menjadi satu Kesatuan/Kompleks Kebakaran.
  * ⚠️ **Isolation Forest**: Mendeteksi anomali pada data titik panas (mengurangi *noise* seperti pantulan panas industri yang tidak lazim).
  * 📊 **K-Means Clustering**: Mengklasifikasikan tingkat risiko (Rendah, Sedang, Tinggi, Ekstrem) secara otomatis berdasarkan kekuatan api (FRP), ukuran kluster, dan persistensi harian.
* **Scoring System**: Menghitung Skor Kepercayaan Komposit (0-100) menggunakan data *confidence* bawaan satelit, FRP ternormalisasi, dan konsistensi kemunculan titik di lokasi yang sama.
* **Interactive Mapping**: Visualisasi peta menggunakan komponen Leaflet.js terintegrasi (Heatmap layer, Cluster Markers, Info Popups).
* **Data Cuaca Dinamis**: Menarik data arah dan kecepatan angin secara *client-side* dari Open-Meteo API saat pengguna mengklik marker.
* **Reverse Geocoding**: Menerjemahkan titik koordinat menjadi nama Kecamatan, Kabupaten, dan Provinsi secara dinamis.

## 🛠️ Teknologi yang Digunakan

* **Backend & Data Processing**: Python, Pandas, NumPy
* **Machine Learning**: Scikit-Learn
* **Web Framework**: Streamlit
* **Frontend/Visualisasi**: HTML/CSS/JavaScript, Leaflet.js, Leaflet.heat
* **Eksternal APIs**: NASA FIRMS, Open-Meteo, BigDataCloud

## 🚀 Cara Instalasi dan Penggunaan

1. **Clone repository ini**
   ```bash
   git clone https://github.com/yogaaditandanu/karhutla-dashboard.git
   cd karhutla-dashboard
   ```

2. **Install dependensi**
   Pastikan Anda sudah menginstal Python versi 3.8 ke atas.
   ```bash
   pip install -r requirements.txt
   ```
   *(Isi `requirements.txt`: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `requests`)*

3. **Jalankan aplikasi Streamlit**
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan otomatis terbuka di peramban web bawaan Anda pada `http://localhost:8501`.

## ⚙️ Catatan Konfigurasi

**Menghindari Limit API Reverse Geocoding**
Aplikasi ini menggunakan API publik tanpa otentikasi dari BigDataCloud untuk proses translasi koordinat ke nama wilayah. Untuk mencegah *timeout* (loading lama) akibat *rate limit* dari server tersebut, skrip ini membatasi jumlah titik yang diproses menggunakan variabel `TOP_N` di dalam `app.py`.
* Default `TOP_N` disarankan disetel di angka `50` atau `100`. 
* Jika *dashboard* terasa *stuck* saat memuat (*loading* lama), silakan turunkan nilai `TOP_N` tersebut.

## 👤 Author
**Yoga Adi Tandanu**
