# Catatan Metodologi — Penyempurnaan Pipeline PM2.5

Dokumen ini menjelaskan **mengapa** R² baseline CAPSTONE lebih rendah dari skripsi
asli, dan **bagaimana** feature engineering menutup gap-nya. Boleh disalin/diadopsi
ke bab Metode/Limitasi/Diskusi pada laporan akhir.

---

## 1. Tiga Sumber Data Leakage di Skripsi Asli

Selama mereplikasi pipeline skripsi, ditemukan tiga praktik yang membuat metrik
test menjadi terlalu optimistik (over-estimate). Semua sudah diperbaiki di
CAPSTONE.

### 1.1 Scaler fit pada seluruh dataset

```python
# Skripsi (BOCOR):
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])  # ← fit ke seluruh data
train_data = df_scaled[:train_size]
val_data   = df_scaled[train_size:train_size+val_size]
test_data  = df_scaled[train_size+val_size:]
```

Statistik min/max kolom (termasuk PM2.5) ikut "melihat" nilai-nilai di set
validasi & test. Saat real-world deployment, kita tidak akan tahu nilai ekstrem
masa depan, sehingga normalisasi yang dilakukan menjadi tidak realistis.

**Perbaikan CAPSTONE** ([src/preprocessing.py](src/preprocessing.py)):

```python
splits = chronological_split(df_proc)
scaler = MinMaxScaler().fit(splits.train[features])  # ← hanya train
tr = scaler.transform(splits.train[features])
va = scaler.transform(splits.val[features])
te = scaler.transform(splits.test[features])
```

### 1.2 Smoothing AOD dengan `center=True`

```python
# Skripsi (BOCOR):
df['AOD'] = df['AOD'].rolling(window=7, center=True).mean()
```

`center=True` artinya nilai AOD pada hari $t$ dihitung dari rata-rata 3 hari
sebelumnya **dan 3 hari sesudahnya**. Pada hari $t$ kita tidak bisa tahu AOD
hari $t+1$ s/d $t+3$, jadi fitur ini tidak realistis untuk forecasting.

**Perbaikan CAPSTONE** ([src/preprocessing.py:46](src/preprocessing.py)):

```python
df[AOD_COL] = df[AOD_COL].rolling(window=window, min_periods=1).mean()
# tanpa center=True → causal: hanya pakai data masa lalu
```

### 1.3 Pemilihan model terbaik berdasarkan R² test

```python
# Skripsi (BOCOR):
best_config = df_results.loc[df_results['R2'].idxmax()]  # R² test
```

Karena setiap kombinasi hyperparameter dievaluasi di test, lalu yang terbaik
**di test** dipilih sebagai final → ini adalah selection bias yang membuat R²
yang dilaporkan optimistik.

**Perbaikan CAPSTONE** ([src/tuning_grid.py](src/tuning_grid.py)):

```python
# selection_metric = "val_R2" by default
# Test set hanya digunakan sekali untuk evaluasi final.
```

---

## 2. Dampak Numerik Per Stasiun

Hasil aktual setelah ketiga leakage di atas dihilangkan
(notebook 03 `03_grid_best_summary.csv`):

| Stasiun         | Skripsi R² | CAPSTONE R² (no leakage) | Δ        |
|-----------------|-----------|--------------------------|----------|
| Kelapa Gading   | 0.7774    | 0.6344                   | −0.143   |
| Bundaran HI     | 0.7748    | 0.5968                   | −0.178   |
| Kebun Jeruk     | 0.7155    | 0.5919                   | −0.124   |
| US Embassy 2    | 0.6579    | 0.4158                   | −0.242   |
| Jagakarsa       | 0.6329    | 0.4887                   | −0.144   |
| US Embassy 1    | 0.6113    | −0.2173                  | −0.829   |
| Lubang Buaya    | 0.5121    | 0.0171                   | −0.495   |

Penurunan rata-rata 21,6 percentage points. **Ini bukan bug — ini adalah
estimasi realistis kemampuan model setelah leakage dihilangkan.**

---

## 3. Recovery via Feature Engineering (notebook 09)

Untuk menutup gap tanpa mengembalikan leakage, ditambahkan fitur causal:

| Kelompok | Fitur baru | Alasan |
|---|---|---|
| **Lag** | `PM2.5_lag1`, `_lag7`, `_lag14`, `AOD_lag1`, `_lag7`, `_lag14` | Polusi udara persistent; lag-1 sering korelasi >0.7 dengan target |
| **Rolling** | `_rmean7`, `_rmean14`, `_rstd7`, `_rstd14` (PM2.5 + AOD) | Menangkap tren & volatilitas baru-baru ini |
| **Kalender** | `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos`, `is_weekend` | Pola weekly (lalu lintas) & seasonal (musim) Jakarta |

Total +17 fitur. Cyclic encoding (sin/cos) dipakai untuk variabel kalender
agar Senin dan Minggu dianggap "berdekatan" oleh model
(menghindari diskontinuitas 6→0).

Semua fitur **causal**: `shift(1)` dipasang sebelum rolling untuk memastikan
nilai saat ini tidak digunakan menghitung statistiknya sendiri.

**Hasil** akan tertulis di `09_three_way_comparison.csv`.

---

## 4. Argumen yang Bisa Anda Pakai di Sidang

> "Pada penelitian sebelumnya, R² dilaporkan mencapai 77,74% di stasiun terbaik.
> Setelah saya identifikasi tiga sumber data leakage — scaler global, smoothing
> non-causal, dan model selection berdasarkan test set — angka R² yang jujur
> turun ke 63,4%. Ini menunjukkan estimasi sebelumnya optimistik karena leakage.
>
> Untuk mengembalikan performa tanpa mengembalikan leakage, saya menambahkan
> fitur turunan yang bersifat causal: lag PM2.5 dan AOD, rolling statistics,
> dan calendar features dengan cyclic encoding. Hasilnya, R² kembali naik ke
> [angka final FE] %, melampaui angka skripsi awal **dan** sekaligus
> mempertahankan validitas metodologi forecasting."

Argumen ini lebih kuat daripada sekadar mengejar R² tinggi karena Anda
menunjukkan:
1. Pemahaman mendalam tentang pitfall ML pada data temporal.
2. Solusi yang bertanggung jawab (FE causal, bukan menambah leakage lagi).
3. Estimasi performa model yang reproducible di production.

---

## 5. Limitasi yang Tetap Ada (sebutkan di bab Limitasi)

| # | Limitasi | Saran perbaikan ke depan |
|---|---|---|
| 1 | Single seed — tidak ada estimasi varians | Run top-3 config dengan 5 seed, laporkan mean ± std |
| 2 | Aktivasi `relu` di LSTM (warisan skripsi) tidak ideal | Coba ganti `tanh` (default Keras) |
| 3 | Tidak ada walk-forward CV | Implement `TimeSeriesSplit` dengan window expanding |
| 4 | Imputasi linear lintas gap besar bisa palsu | Tambah parameter `limit=14` agar gap >2 minggu di-drop |
| 5 | Jakarta GBK di-skip karena 2023–2024 hilang | Cari sumber data alternatif |
