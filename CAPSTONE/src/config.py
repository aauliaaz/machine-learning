"""Konstanta global untuk pipeline CAPSTONE."""
from pathlib import Path

# ---- Path ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPSTONE_DIR = PROJECT_ROOT / "CAPSTONE"
RAW_MERGED_DIR = PROJECT_ROOT / "Data Penelitian" / "data gabungan aod cuaca pm2.5"

PROCESSED_DIR = CAPSTONE_DIR / "data" / "processed"
EXTERNAL_DIR = CAPSTONE_DIR / "data" / "external"
RESULTS_DIR = CAPSTONE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODEL_DIR = RESULTS_DIR / "models"

for d in (PROCESSED_DIR, EXTERNAL_DIR, FIG_DIR, METRICS_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Stasiun & koordinat (Tabel 2 skripsi, sudah dikoreksi: lat negatif, lon positif) ----
STATIONS = {
    "us_embassy_1":     {"lat": -6.1811, "lon": 106.8279},
    "us_embassy_2":     {"lat": -6.2366, "lon": 106.7931},
    "jakarta_gbk":      {"lat": -6.2155, "lon": 106.8030},
    "bundaran_hi":      {"lat": -6.1946, "lon": 106.8235},
    "kelapa_gading":    {"lat": -6.1535, "lon": 106.9108},
    "jagakarsa":        {"lat": -6.3569, "lon": 106.8036},
    "lubang_buaya":     {"lat": -6.2888, "lon": 106.9091},
    "kebun_jeruk":      {"lat": -6.2073, "lon": 106.7531},
}

# Stasiun yang dianggap sehat untuk eksperimen (Jakarta GBK punya 2 tahun missing)
HEALTHY_STATIONS = [s for s in STATIONS if s != "jakarta_gbk"]

# ---- Kolom fitur ----
TARGET = "ISPU PM2.5"
WEATHER_COLS = ["temp", "dew", "humidity", "precip", "windspeed"]
AOD_COL = "AOD"
DATE_COL = "datetime"

# ---- Konfigurasi training ----
LOOKBACK = 30
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
