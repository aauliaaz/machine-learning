# CAPSTONE — Penyempurnaan Model Prediksi PM2.5 Jakarta

Folder ini berisi penyempurnaan model prediksi PM2.5 berbasis LSTM yang sebelumnya
dibangun pada skripsi *Steven Hesang (2025) — "Model Prediksi Konsentrasi PM2.5 di
Jakarta Menggunakan LSTM Berdasarkan Data AOD Himawari"*.

**Catatan penting:** Folder asli `Software/Kode LSTM ...` dan `Data Penelitian/`
**TIDAK diubah**. Semua eksperimen baru di-self-contained di dalam folder `CAPSTONE/`.

---

## Struktur Folder

```
CAPSTONE/
├── README.md                      <- file ini
├── requirements.txt               <- dependensi tambahan (optuna, earthengine-api, dll.)
├── METHODOLOGY_NOTES.md           <- argumen leakage-fix + FE untuk laporan
├── src/                           <- modul Python yang dapat di-reuse
│   ├── __init__.py
│   ├── config.py                  <- konstanta (path, daftar stasiun, koordinat)
│   ├── data_loader.py             <- load PM2.5 + AOD + cuaca, sinkronisasi waktu
│   ├── preprocessing.py           <- imputasi, normalisasi, train/val/test split
│   ├── feature_engineering.py     <- lag, rolling stats, calendar (cyclic)
│   ├── model.py                   <- arsitektur LSTM + helper training
│   ├── tuning_grid.py             <- grid search wrapper
│   ├── tuning_optuna.py           <- optuna wrapper
│   └── evaluation.py              <- R², MSE, MAE, plotting
├── notebooks/                     <- 1 notebook per tahap eksperimen
│   ├── 01_data_quality_check.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_baseline_grid_search.ipynb
│   ├── 04_optuna_tuning.ipynb
│   ├── 05_univariate_vs_multivariate.ipynb
│   ├── 06_add_lat_long.ipynb
│   ├── 07_add_ndvi_gee.ipynb
│   ├── 08_final_evaluation.ipynb
│   └── 09_feature_engineering.ipynb   <- recovery R² via lag/rolling/calendar
├── data/
│   ├── processed/                 <- dataset hasil preprocessing per stasiun
│   └── external/                  <- NDVI dari GEE, dll.
└── results/
    ├── figures/                   <- semua plot
    ├── metrics/                   <- CSV hasil metrik tiap eksperimen
    └── models/                    <- file .keras model terbaik per eksperimen
```

---

## Perbandingan dengan Pipeline Asli

| Aspek                        | Skripsi Asli               | CAPSTONE                                          |
|------------------------------|----------------------------|---------------------------------------------------|
| Scaler fit                   | seluruh data (leakage)     | hanya data train                                  |
| Imputasi                     | linear interpolation       | linear + (ffill+bfill) sebagai fallback           |
| Smoothing AOD                | rolling(7, center=True)    | rolling causal (tidak bocor masa depan), opsional |
| Tuning                       | Grid Search                | Grid Search **+ Optuna** + perbandingan           |
| Skenario fitur               | semua-fitur saja           | Univariate vs Multivariate, ±lat/lon, ±NDVI       |
| Modularitas                  | copy-paste 16 notebook     | modul `src/` + 8 notebook eksperimen              |
| Early stopping               | tidak ada                  | `EarlyStopping` + `ModelCheckpoint`               |
| Metrik                       | R² ternormalisasi          | R², MSE, MAE pada **skala asli µg/m³**            |

---

## Arahan Dosen yang Diimplementasikan

1. ✅ Cek missing value, distribusi, sinkronisasi waktu — `notebooks/01_*`
2. ✅ Imputasi (interpolation atau ffill+bfill), MinMaxScaler — `src/preprocessing.py`
3. ✅ Gabungkan dataset & split train/val/test — `src/data_loader.py`
4. ✅ Baseline LSTM + Grid Search — `notebooks/03_*`
5. ✅ Optuna Tuning + perbandingan — `notebooks/04_*`
6. ✅ Univariate vs Multivariate — `notebooks/05_*`
7. ✅ Tambah Lat/Long — `notebooks/06_*`
8. ✅ Tambah NDVI dari GEE — `notebooks/07_*`
9. ✅ Evaluasi final — `notebooks/08_*`
10. ✅ **Bonus** — Feature Engineering (lag, rolling, calendar) — `notebooks/09_*`
    untuk menutup gap R² akibat perbaikan leakage. Lihat
    [METHODOLOGY_NOTES.md](METHODOLOGY_NOTES.md) untuk argumen lengkap.

---

## Cara Menjalankan

```bash
# 1. install dependency (jika belum)
pip install -r CAPSTONE/requirements.txt

# 2. jalankan notebook secara berurutan dari 01 → 08
#    setiap notebook akan menulis output ke results/ dan data/processed/
```

Penulis CAPSTONE: dilanjutkan dari skripsi Steven Hesang (2025).
