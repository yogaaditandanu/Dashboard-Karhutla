import json
import math
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# KONFIGURASI HALAMAN STREAMLIT
# ============================================================
st.set_page_config(page_title="Dashboard Karhutla Indonesia", layout="wide")
st.title("🔥 Dashboard Karhutla Indonesia (Real-Time)")
st.markdown("Data ditarik langsung dari API satelit NASA FIRMS dan diproses menggunakan *Machine Learning* (DBSCAN, Isolation Forest, K-Means).")

# ============================================================
# FUNGSI CACHE & PEMROSESAN DATA
# ttl=3600 -> Data akan ditarik ulang dari NASA setiap 1 jam
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def generate_dashboard_html():
    MAP_KEY = "376cfbf35a34fbbad8c30858f7627529" 
    area_coords = "95,-11,141,6"   
    day_range = 5                    
    sumber_list = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT"]

    semua_df = []
    for sumber in sumber_list:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{sumber}/{area_coords}/{day_range}"
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            if "Invalid" in resp.text[:200] or "Error" in resp.text[:200]:
                raise ValueError(resp.text[:200])
            df = pd.read_csv(StringIO(resp.text))
            df['sumber_satelit'] = sumber
            semua_df.append(df)
        except Exception as e:
            pass # Lewati jika ada API yang gagal

    if not semua_df:
        return "<h3 style='color:red;'>Gagal menarik data dari server NASA. Coba lagi nanti.</h3>"

    df = pd.concat(semua_df, ignore_index=True)
    df = df[(df['latitude'] >= -11) & (df['latitude'] <= 6) &
            (df['longitude'] >= 95) & (df['longitude'] <= 141)].copy()

    if 'confidence' in df.columns:
        df['confidence_str'] = df['confidence'].astype(str).str.lower()
    else:
        df['confidence_str'] = 'n'

    df['lat_bulat'] = df['latitude'].round(3)
    df['lon_bulat'] = df['longitude'].round(3)
    df = df.drop_duplicates(subset=['lat_bulat', 'lon_bulat', 'acq_date'])

    # ML #1 — DBSCAN
    KM_PER_RADIAN = 6371.0088
    koordinat_radian = np.radians(df[['latitude', 'longitude']].values)
    db = DBSCAN(eps=3.0 / KM_PER_RADIAN, min_samples=3, metric='haversine').fit(koordinat_radian)
    df['cluster_id'] = db.labels_
    jumlah_cluster = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    # ML #2 — ISOLATION FOREST
    ukuran_cluster_map = df['cluster_id'].value_counts().to_dict()
    df['ukuran_cluster'] = df['cluster_id'].map(lambda c: ukuran_cluster_map.get(c, 0) if c != -1 else 0)
    df['frp'] = df.get('frp', 0).fillna(0)

    X = StandardScaler().fit_transform(df[['frp', 'ukuran_cluster']])
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    df['is_anomali'] = (iso.fit_predict(X) == -1)

    # PERSISTENCE SCORE
    df['lokasi_key'] = df['lat_bulat'].astype(str) + '_' + df['lon_bulat'].astype(str)
    persistence_map = df.groupby('lokasi_key')['acq_date'].nunique().to_dict()
    df['hari_persisten'] = df['lokasi_key'].map(persistence_map)
    df['persistence_norm'] = (df['hari_persisten'] - 1) / (day_range - 1 + 1e-9)

    # ML #3 — K-MEANS
    fitur_risiko = df[['frp', 'ukuran_cluster']].copy()
    fitur_risiko['persistence'] = df['hari_persisten']
    X_risiko = StandardScaler().fit_transform(fitur_risiko)

    K_RISIKO = 4
    kmeans = KMeans(n_clusters=K_RISIKO, random_state=42, n_init=10)
    df['risiko_cluster_raw'] = kmeans.fit_predict(X_risiko)

    urutan_label = ['Rendah', 'Sedang', 'Tinggi', 'Ekstrem']
    rata_frp_per_grup = df.groupby('risiko_cluster_raw')['frp'].mean().sort_values()
    mapping_risiko = {grup: urutan_label[i] for i, grup in enumerate(rata_frp_per_grup.index)}
    df['tingkat_risiko'] = df['risiko_cluster_raw'].map(mapping_risiko)

    # SKOR KEPERCAYAAN
    peta_confidence_dasar = {'h': 90, 'high': 90, 'n': 60, 'nominal': 60, 'l': 30, 'low': 30}
    df['confidence_dasar'] = df['confidence_str'].map(peta_confidence_dasar).fillna(60)

    frp_norm = (df['frp'] - df['frp'].min()) / (df['frp'].max() - df['frp'].min() + 1e-9)
    df['skor_kepercayaan'] = (
        0.50 * df['confidence_dasar']
        + 0.25 * (frp_norm * 100)
        + 0.25 * (df['persistence_norm'] * 100)
    ).round(1)

    df['skor_prioritas'] = (0.6 * frp_norm) + (0.4 * df['persistence_norm'])

    TOP_N = 50 
    top_df = df.sort_values('skor_prioritas', ascending=False).head(TOP_N).copy()

    # REVERSE GEOCODING
    def ambil_nama_lokasi(lat, lon):
        url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=id"
        try:
            r = requests.get(url, timeout=8).json()
            kota = r.get('city') or r.get('locality') or '-'
            admin = r.get('localityInfo', {}).get('administrative', [])
            kec = next((a['name'] for a in admin if a.get('adminLevel') == 8), None)
            kab = next((a['name'] for a in admin if a.get('adminLevel') == 6), kota)
            prov = r.get('principalSubdivision', '-')
            return kec or '-', kab, prov
        except Exception:
            return '-', 'Tidak diketahui', '-'

    lokasi_cache = {}
    kecamatan_list, kabupaten_list, provinsi_list = [], [], []
    for _, row in top_df.iterrows():
        key = (round(row['latitude'], 2), round(row['longitude'], 2))
        if key not in lokasi_cache:
            lokasi_cache[key] = ambil_nama_lokasi(row['latitude'], row['longitude'])
        kec, kab, prov = lokasi_cache[key]
        kecamatan_list.append(kec)
        kabupaten_list.append(kab)
        provinsi_list.append(prov)

    top_df['kecamatan'] = kecamatan_list
    top_df['kabupaten'] = kabupaten_list
    top_df['provinsi'] = provinsi_list

    peta_label_satelit = {'h': 'Tinggi', 'high': 'Tinggi', 'n': 'Sedang', 'nominal': 'Sedang', 'l': 'Rendah', 'low': 'Rendah'}
    top_df['confidence_label'] = top_df['confidence_str'].map(peta_label_satelit).fillna('Sedang')

    # SUSUN JSON
    markers_json = []
    for _, r in top_df.iterrows():
        markers_json.append({
            "lat": round(float(r['latitude']), 5),
            "lon": round(float(r['longitude']), 5),
            "tanggal": str(r['acq_date']),
            "waktu": f"{str(int(r['acq_time'])).zfill(4)[:2]}:{str(int(r['acq_time'])).zfill(4)[2:]} UTC",
            "kecamatan": r['kecamatan'],
            "kabupaten": r['kabupaten'],
            "provinsi": r['provinsi'],
            "confidence_label": r['confidence_label'],
            "skor_kepercayaan": float(r['skor_kepercayaan']),
            "tingkat_risiko": r['tingkat_risiko'],
            "hari_persisten": int(r['hari_persisten']),
            "satelit": r['sumber_satelit'].replace('_NRT', '').replace('_', ' '),
            "frp": round(float(r['frp']), 2),
            "cluster_id": int(r['cluster_id']),
            "is_anomali": bool(r['is_anomali']),
        })

    cluster_summary = (
        df[df['cluster_id'] != -1]
        .groupby('cluster_id')
        .agg(jumlah_titik=('latitude', 'count'), total_frp=('frp', 'sum'),
             pusat_lat=('latitude', 'mean'), pusat_lon=('longitude', 'mean'))
        .reset_index()
    )

    if not cluster_summary.empty:
        dominan_risiko = (
            df[df['cluster_id'] != -1]
            .groupby('cluster_id')['tingkat_risiko']
            .agg(lambda x: x.value_counts().idxmax())
        )
        cluster_summary['tingkat_dominan'] = cluster_summary['cluster_id'].map(dominan_risiko)
        rentang_tanggal = (
            df[df['cluster_id'] != -1]
            .groupby('cluster_id')['acq_date']
            .agg(['min', 'max'])
        )
        cluster_summary['tanggal_awal'] = cluster_summary['cluster_id'].map(rentang_tanggal['min'])
        cluster_summary['tanggal_akhir'] = cluster_summary['cluster_id'].map(rentang_tanggal['max'])
        clusters_json = cluster_summary.to_dict('records')
    else:
        clusters_json = []

    semua_titik_json = df[['latitude', 'longitude', 'acq_date']].rename(
        columns={'latitude': 'lat', 'longitude': 'lon', 'acq_date': 'tanggal'}
    ).to_dict('records')

    tanggal_tersedia = sorted(df['acq_date'].unique().tolist())
    tanggal_terbaru = tanggal_tersedia[-1] if tanggal_tersedia else datetime.now().strftime('%Y-%m-%d')

    DATA = {
        "markers": markers_json,
        "clusters": clusters_json,
        "semua_titik": semua_titik_json,
        "tanggal_tersedia": tanggal_tersedia,
        "tanggal_terbaru": tanggal_terbaru,
        "total_titik": len(df),
        "jumlah_cluster": jumlah_cluster,
        "jumlah_anomali": int(df['is_anomali'].sum()),
        "jumlah_ekstrem": int((df['tingkat_risiko'] == 'Ekstrem').sum()),
        "update_server": datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    data_js = json.dumps(DATA, ensure_ascii=False)

    # ============================================================
    # TEMPLATE HTML
    # ============================================================
    html_template = """<!DOCTYPE html>
    <html lang="id">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
    <style>
      :root{
        --bg-page:#ffffff;
        --panel-bg:#ffffff;
        --ink:#1b2430;
        --ink-soft:#5b6472;
        --line:#e4e8ee;
        --brand:#c2410c;
        --ekstrem:#991b1b;
        --tinggi:#dc2626;
        --sedang:#f59e0b;
        --rendah:#eab308;
        --anomali:#111827;
        --radius:14px;
        --shadow:0 8px 24px rgba(20,30,50,0.10), 0 1px 2px rgba(20,30,50,0.06);
      }
      *{box-sizing:border-box;}
      body{margin:0;font-family:'Inter',sans-serif;background:var(--bg-page);color:var(--ink);}
      .mono{font-family:'JetBrains Mono',monospace;}

      .statbar{
        display:flex;flex-wrap:wrap;gap:10px;justify-content:center;
        padding:10px 16px;
      }
      .stat-pill{
        background:var(--panel-bg);border:1px solid var(--line);border-radius:999px;
        padding:10px 18px;display:flex;align-items:center;gap:8px;
        box-shadow:var(--shadow);font-weight:600;font-size:14px;white-space:nowrap;
      }
      .stat-pill .ic{font-size:16px;}
      .stat-pill.brand{color:var(--brand);}
      .stat-pill .val{color:var(--ink);}
      .stat-pill .lbl{color:var(--ink-soft);font-weight:500;}

      .map-wrap{
        position:relative;max-width:100%;margin:14px auto 28px;
        border-radius:20px;overflow:hidden;box-shadow:var(--shadow);
        border:1px solid var(--line);
      }
      #map{height:75vh;min-height:560px;width:100%;background:#dce3ea;}

      .info-panel{
        position:absolute;top:16px;right:16px;width:300px;
        background:var(--panel-bg);border-radius:var(--radius);box-shadow:var(--shadow);
        padding:18px;z-index:900;display:none;
      }
      .info-panel.show{display:block;}
      .info-panel .close-btn{
        position:absolute;top:10px;right:12px;border:none;background:none;
        font-size:18px;color:var(--ink-soft);cursor:pointer;
      }
      .info-eyebrow{font-size:11px;letter-spacing:.06em;color:var(--ink-soft);
        text-transform:uppercase;font-weight:700;display:flex;align-items:center;gap:6px;margin-bottom:6px;}
      .info-title{font-size:19px;font-weight:800;color:var(--brand);line-height:1.25;margin:0;}
      .info-sub{font-size:13px;color:var(--ink-soft);margin:2px 0 14px;}
      .info-row{
        background:#f7f8fa;border-radius:10px;padding:10px 12px;margin-bottom:8px;
        display:flex;justify-content:space-between;align-items:center;
      }
      .info-row .k{font-size:12px;color:var(--ink-soft);font-weight:600;}
      .info-row .v{font-size:13px;font-weight:700;}
      .badge{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;color:#fff;}
      .badge.ekstrem{background:var(--ekstrem);}
      .badge.tinggi{background:var(--tinggi);}
      .badge.sedang{background:var(--sedang);}
      .badge.rendah{background:var(--rendah);color:#3a2c00;}
      .badge.anomali{background:var(--anomali);}
      .coord-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px;}
      .coord-grid div{background:#f7f8fa;border-radius:10px;padding:8px 10px;font-size:12px;}
      .coord-grid .k{color:var(--ink-soft);display:block;font-size:10px;text-transform:uppercase;margin-bottom:2px;}

      .legend{
        position:absolute;bottom:16px;left:16px;background:var(--panel-bg);
        border-radius:var(--radius);box-shadow:var(--shadow);padding:14px 16px;
        z-index:900;font-size:13px;
      }
      .legend .row{display:flex;align-items:center;gap:8px;margin:5px 0;font-weight:600;}
      .dot{width:11px;height:11px;border-radius:50%;display:inline-block;}
      .dot.ekstrem{background:var(--ekstrem);}
      .dot.tinggi{background:var(--tinggi);} .dot.sedang{background:var(--sedang);}
      .dot.rendah{background:var(--rendah);} .dot.anomali{background:var(--anomali);}

      .layer-toggles{
        position:absolute;top:16px;left:60px;z-index:900;display:flex;gap:8px;flex-wrap:wrap;max-width:60%;
      }
      .chip{
        background:var(--panel-bg);border:1px solid var(--line);border-radius:999px;
        padding:7px 14px;font-size:12.5px;font-weight:700;cursor:pointer;box-shadow:var(--shadow);
        display:flex;align-items:center;gap:6px;color:var(--ink-soft);user-select:none;
      }
      .chip.active{background:var(--brand);color:#fff;border-color:var(--brand);}

      .date-tabs{
        position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
        background:var(--panel-bg);border-radius:16px;box-shadow:var(--shadow);
        padding:10px;display:flex;gap:6px;z-index:900;
      }
      .date-tab{
        border:none;background:none;border-radius:10px;padding:8px 12px;text-align:center;
        cursor:pointer;font-family:inherit;min-width:52px;
      }
      .date-tab .d{display:block;font-size:10px;color:var(--ink-soft);font-weight:700;text-transform:uppercase;}
      .date-tab .n{display:block;font-size:16px;font-weight:800;color:var(--ink);}
      .date-tab .m{display:block;font-size:10px;color:var(--ink-soft);font-weight:700;text-transform:uppercase;}
      .date-tab.active{background:var(--brand);}
      .date-tab.active .d, .date-tab.active .n, .date-tab.active .m{color:#fff;}
    </style>
    </head>
    <body>

    <div class="statbar" id="statbar"></div>

    <div class="map-wrap">
      <div id="map"></div>
      <div class="layer-toggles">
        <div class="chip active" id="toggle-heat" onclick="toggleLayer('heat')">🔥 Heatmap</div>
        <div class="chip active" id="toggle-cluster" onclick="toggleLayer('cluster')">🧩 Kompleks Kebakaran</div>
        <div class="chip active" id="toggle-anomali" onclick="toggleLayer('anomali')">⚠️ Anomali (ML)</div>
      </div>
      <div class="info-panel" id="infoPanel">
        <button class="close-btn" onclick="tutupPanel()">✕</button>
        <div class="info-eyebrow" id="ip-eyebrow">📍 Info Titik Panas</div>
        <p class="info-title" id="ip-kecamatan">-</p>
        <p class="info-sub" id="ip-kabkota">-</p>
        <div class="info-row"><span class="k" id="ip-label-risiko">Tingkat Risiko</span><span class="v" id="ip-confidence"></span></div>
        <div class="info-row" id="ip-satelit-row"><span class="k">Satelit</span><span class="v" id="ip-satelit">-</span></div>
        <div class="info-row"><span class="k" id="ip-label-frp">FRP (Kekuatan Api)</span><span class="v" id="ip-frp">-</span></div>
        <div class="info-row" style="flex-direction:column;align-items:flex-start;gap:4px;">
          <span class="k">💨 Angin di Lokasi</span><span class="v" id="ip-angin" style="font-size:13px;">Memuat...</span>
        </div>
        <div class="info-row" style="flex-direction:column;align-items:flex-start;gap:4px;" id="ip-skor-row">
          <span class="k">Skor Kepercayaan (ML)</span><span class="v" id="ip-skor" style="font-size:12px;"></span>
        </div>
        <div class="info-row" id="ip-anomali-row"><span class="k">Status Anomali</span><span class="v" id="ip-anomali">-</span></div>
        <div class="coord-grid">
          <div><span class="k">LAT</span><span class="mono" id="ip-lat">-</span></div>
          <div><span class="k">LON</span><span class="mono" id="ip-lon">-</span></div>
        </div>
        <p class="info-sub" style="margin-top:10px;margin-bottom:0;" id="ip-waktu">-</p>
      </div>
      <div class="legend">
        <div class="row"><span class="dot ekstrem"></span> Ekstrem</div>
        <div class="row"><span class="dot tinggi"></span> Tinggi</div>
        <div class="row"><span class="dot sedang"></span> Sedang</div>
        <div class="row"><span class="dot rendah"></span> Rendah</div>
        <div class="row"><span class="dot anomali"></span> Anomali (ML)</div>
      </div>
      <div class="date-tabs" id="dateTabs"></div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <script>
    const DATA = __DATA_JSON__;

    const map = L.map('map', { zoomControl: true }).setView([-2.5, 118.0], 5);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap &copy; CartoDB', maxZoom: 18
    }).addTo(map);

    let currentDate = DATA.tanggal_terbaru;
    let markerLayer = L.layerGroup().addTo(map);
    let heatLayer = null;
    let clusterLayer = L.layerGroup().addTo(map);
    let windLayer = L.layerGroup().addTo(map);
    let anomaliVisible = true, clusterVisible = true, heatVisible = true;

    const WARNA_RISIKO = { 'Ekstrem':'#991b1b', 'Tinggi':'#dc2626', 'Sedang':'#f59e0b', 'Rendah':'#eab308' };
    const KELAS_RISIKO = { 'Ekstrem':'ekstrem', 'Tinggi':'tinggi', 'Sedang':'sedang', 'Rendah':'rendah' };

    function warnaByRisiko(tingkat){ return WARNA_RISIKO[tingkat] || WARNA_RISIKO['Sedang']; }

    function arahMataAngin(deg){
      const arah = ['Utara','Timur Laut','Timur','Tenggara','Selatan','Barat Daya','Barat','Barat Laut'];
      return arah[Math.round(deg / 45) % 8];
    }

    async function ambilAngin(lat, lon){
      try{
        const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true`;
        const r = await fetch(url);
        const j = await r.json();
        return { speed: j.current_weather.windspeed, dir: j.current_weather.winddirection };
      }catch(e){ return { speed: null, dir: null }; }
    }

    async function ambilLokasi(lat, lon){
      try{
        const url = `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=id`;
        const r = await fetch(url);
        const j = await r.json();
        const kota = j.city || j.locality || '-';
        const admin = (j.localityInfo && j.localityInfo.administrative) || [];
        const kec = (admin.find(a => a.adminLevel === 8) || {}).name || '-';
        const kab = (admin.find(a => a.adminLevel === 6) || {}).name || kota;
        const prov = j.principalSubdivision || '-';
        return { kecamatan: kec, kabupaten: kab, provinsi: prov };
      }catch(e){ return { kecamatan: '-', kabupaten: 'Tidak diketahui', provinsi: '-' }; }
    }

    function gambarPanahAngin(lat, lon, speed, dir){
      windLayer.clearLayers();
      if (speed == null) return;
      const arahTujuan = (dir + 180) % 360; 
      const jarakKm = Math.max(3, speed * 0.6);
      const rad = (arahTujuan * Math.PI) / 180;
      const dLat = (jarakKm * Math.cos(rad)) / 111;
      const dLon = (jarakKm * Math.sin(rad)) / (111 * Math.cos(lat * Math.PI / 180));
      const ujung = [lat + dLat, lon + dLon];
      L.polyline([[lat, lon], ujung], { color: '#0284c7', weight: 3, dashArray: '5,5' }).addTo(windLayer);
      L.circleMarker(ujung, { radius: 5, color: '#fff', weight: 1.5, fillColor: '#0284c7', fillOpacity: 1 })
        .bindTooltip('Arah potensi rambatan').addTo(windLayer);
    }

    async function tampilkanAngin(lat, lon){
      document.getElementById('ip-angin').textContent = 'Memuat data angin...';
      const angin = await ambilAngin(lat, lon);
      if (angin.speed == null){
        document.getElementById('ip-angin').textContent = 'Data angin tidak tersedia';
        return;
      }
      document.getElementById('ip-angin').innerHTML =
        `${angin.speed} km/h dari arah <b>${arahMataAngin(angin.dir)}</b> (${angin.dir}°)`;
      gambarPanahAngin(lat, lon, angin.speed, angin.dir);
    }

    function bukaPanelTitik(m){
      document.getElementById('ip-eyebrow').textContent = '📍 Info Titik Panas';
      document.getElementById('ip-label-risiko').textContent = 'Tingkat Risiko';
      document.getElementById('ip-label-frp').textContent = 'FRP (Kekuatan Api)';
      document.getElementById('ip-satelit-row').style.display = 'flex';
      document.getElementById('ip-skor-row').style.display = 'flex';
      document.getElementById('ip-anomali-row').style.display = 'flex';

      document.getElementById('ip-kecamatan').textContent = (m.kecamatan && m.kecamatan !== '-') ? 'Kec. ' + m.kecamatan : m.kabupaten;
      document.getElementById('ip-kabkota').textContent = m.kabupaten + ', ' + m.provinsi;
      const kelas = KELAS_RISIKO[m.tingkat_risiko] || 'sedang';
      document.getElementById('ip-confidence').innerHTML = `<span class="badge ${kelas}">${m.tingkat_risiko}</span>`;
      document.getElementById('ip-satelit').textContent = m.satelit;
      document.getElementById('ip-frp').textContent = m.frp + ' MW';
      document.getElementById('ip-skor').innerHTML =
        `<span class="mono">${m.skor_kepercayaan.toFixed(0)}/100</span> <span style="color:var(--ink-soft);font-weight:500;">(persisten ${m.hari_persisten} hari)</span>`;
      document.getElementById('ip-anomali').innerHTML = m.is_anomali
        ? '<span class="badge anomali">⚠️ Anomali</span>' : 'Normal';
      document.getElementById('ip-lat').textContent = m.lat.toFixed(5);
      document.getElementById('ip-lon').textContent = m.lon.toFixed(5);
      document.getElementById('ip-waktu').textContent = '📅 ' + m.tanggal + '  •  🕒 ' + m.waktu;
      document.getElementById('infoPanel').classList.add('show');

      tampilkanAngin(m.lat, m.lon);
    }

    async function bukaPanelCluster(c){
      document.getElementById('ip-eyebrow').textContent = '🧩 Info Kompleks Kebakaran';
      document.getElementById('ip-label-risiko').textContent = 'Risiko Dominan';
      document.getElementById('ip-label-frp').textContent = 'Total FRP Kompleks';
      document.getElementById('ip-satelit-row').style.display = 'none';
      document.getElementById('ip-skor-row').style.display = 'none';
      document.getElementById('ip-anomali-row').style.display = 'none';

      document.getElementById('ip-kecamatan').textContent = 'Memuat lokasi...';
      document.getElementById('ip-kabkota').textContent = '';
      const kelas = KELAS_RISIKO[c.tingkat_dominan] || 'sedang';
      document.getElementById('ip-confidence').innerHTML = `<span class="badge ${kelas}">${c.tingkat_dominan}</span>`;
      document.getElementById('ip-frp').textContent = `${c.total_frp.toFixed(1)} MW  (${c.jumlah_titik} titik panas)`;
      document.getElementById('ip-lat').textContent = c.pusat_lat.toFixed(5);
      document.getElementById('ip-lon').textContent = c.pusat_lon.toFixed(5);
      const rentang = c.tanggal_awal === c.tanggal_akhir ? c.tanggal_awal : `${c.tanggal_awal} s/d ${c.tanggal_akhir}`;
      document.getElementById('ip-waktu').textContent = `🧩 Kompleks #${c.cluster_id}  •  📅 Aktif: ${rentang}`;
      document.getElementById('infoPanel').classList.add('show');

      const [lokasi] = await Promise.all([
        ambilLokasi(c.pusat_lat, c.pusat_lon),
        tampilkanAngin(c.pusat_lat, c.pusat_lon)
      ]);
      document.getElementById('ip-kecamatan').textContent = (lokasi.kecamatan && lokasi.kecamatan !== '-') ? 'Kec. ' + lokasi.kecamatan : lokasi.kabupaten;
      document.getElementById('ip-kabkota').textContent = lokasi.kabupaten + ', ' + lokasi.provinsi;
    }

    function tutupPanel(){
      document.getElementById('infoPanel').classList.remove('show');
      windLayer.clearLayers();
    }

    function renderMarkers(tanggal){
      markerLayer.clearLayers();
      const subset = DATA.markers.filter(m => m.tanggal === tanggal);
      subset.forEach(m => {
        const warna = m.is_anomali ? '#111827' : warnaByRisiko(m.tingkat_risiko);
        const radius = m.is_anomali ? 9 : (m.tingkat_risiko === 'Ekstrem' ? 8 : 6);
        const marker = L.circleMarker([m.lat, m.lon], {
          radius: radius, color: '#fff', weight: 1.5, fillColor: warna, fillOpacity: 0.95
        });
        marker.on('click', () => bukaPanelTitik(m));
        if (!m.is_anomali || anomaliVisible) marker.addTo(markerLayer);
      });
      document.getElementById('stat-total').textContent = subset.length.toLocaleString('id-ID');
    }

    function renderHeat(){
      if (heatLayer) map.removeLayer(heatLayer);
      const pts = DATA.semua_titik.map(t => [t.lat, t.lon, 0.5]);
      heatLayer = L.heatLayer(pts, { radius: 14, blur: 18, minOpacity: 0.25 });
      if (heatVisible) heatLayer.addTo(map);
    }

    function renderClusters(){
      clusterLayer.clearLayers();
      if (!clusterVisible) return;
      const palet = ['#c2410c','#0891b2','#7c3aed','#059669','#db2777','#65a30d','#0284c7','#ea580c'];
      DATA.clusters.forEach((c, i) => {
        const warna = palet[i % palet.length];
        const lingkaran = L.circle([c.pusat_lat, c.pusat_lon], {
          radius: Math.max(2500, Math.sqrt(c.jumlah_titik) * 1800),
          color: warna, weight: 2, fillColor: warna, fillOpacity: 0.10
        }).bindTooltip(`Kompleks #${c.cluster_id} — ${c.jumlah_titik} titik, ${c.total_frp.toFixed(0)} MW`);
        lingkaran.on('click', () => bukaPanelCluster(c));
        lingkaran.addTo(clusterLayer);
      });
    }

    function toggleLayer(nama){
      if (nama === 'heat'){
        heatVisible = !heatVisible;
        document.getElementById('toggle-heat').classList.toggle('active', heatVisible);
        if (heatVisible) heatLayer.addTo(map); else map.removeLayer(heatLayer);
      }
      if (nama === 'cluster'){
        clusterVisible = !clusterVisible;
        document.getElementById('toggle-cluster').classList.toggle('active', clusterVisible);
        renderClusters();
      }
      if (nama === 'anomali'){
        anomaliVisible = !anomaliVisible;
        document.getElementById('toggle-anomali').classList.toggle('active', anomaliVisible);
        renderMarkers(currentDate);
      }
    }

    function renderDateTabs(){
      const wrap = document.getElementById('dateTabs');
      wrap.innerHTML = '';
      const namaHari = ['MIN','SEN','SEL','RAB','KAM','JUM','SAB'];
      const namaBulan = ['JAN','FEB','MAR','APR','MEI','JUN','JUL','AGU','SEP','OKT','NOV','DES'];
      DATA.tanggal_tersedia.forEach(tgl => {
        const d = new Date(tgl + 'T00:00:00');
        const btn = document.createElement('button');
        btn.className = 'date-tab' + (tgl === currentDate ? ' active' : '');
        btn.innerHTML = `<span class="d">${namaHari[d.getDay()]}</span><span class="n">${d.getDate()}</span><span class="m">${namaBulan[d.getMonth()]}</span>`;
        btn.onclick = () => {
          currentDate = tgl;
          document.querySelectorAll('.date-tab').forEach(el => el.classList.remove('active'));
          btn.classList.add('active');
          renderMarkers(currentDate);
          document.getElementById('stat-tanggal').textContent = tgl;
          tutupPanel();
        };
        wrap.appendChild(btn);
      });
    }

    function renderStatbar(){
      document.getElementById('statbar').innerHTML = `
        <div class="stat-pill brand"><span class="ic">🔥</span><span class="val" id="stat-total">${DATA.total_titik.toLocaleString('id-ID')}</span><span class="lbl">Titik Panas</span></div>
        <div class="stat-pill"><span class="ic">📅</span><span class="val" id="stat-tanggal">${DATA.tanggal_terbaru}</span></div>
        <div class="stat-pill"><span class="ic">🔄</span><span class="lbl">Update:</span><span class="val">${DATA.update_server}</span></div>
        <div class="stat-pill"><span class="ic">🧩</span><span class="val">${DATA.jumlah_cluster}</span><span class="lbl">Kompleks</span></div>
        <div class="stat-pill"><span class="ic">🟥</span><span class="val">${DATA.jumlah_ekstrem}</span><span class="lbl">Risiko Ekstrem</span></div>
      `;
    }

    renderStatbar();
    renderDateTabs();
    renderMarkers(currentDate);
    renderHeat();
    renderClusters();
    </script>
    </body>
    </html>
    """

    return html_template.replace("__DATA_JSON__", data_js)

# ============================================================
# TAMPILKAN KE STREAMLIT UI
# ============================================================
with st.spinner("⏳ Menarik data titik panas terbaru & memproses Machine Learning..."):
    html_output = generate_dashboard_html()

components.html(html_output, height=850, scrolling=True)