# 🌊 Flood Risk Assessment & Early Warning System — Germany

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-black?logo=github)](https://github.com/ayouboihi/flood-risk-germany/actions)

An end-to-end geospatial machine learning pipeline for flood risk mapping, river level forecasting, and automated early warning in Germany. Combines Sentinel-1 SAR satellite imagery, open elevation data, and real-time river gauge data with Random Forest, XGBoost, LSTM, and U-Net deep learning models.

---

## 📍 Study Area

| Parameter | Value |
|---|---|
| Region | Rhine Valley, Germany (Mannheim → Köln) |
| Bounding box | 7.5–8.5°E, 49.5–50.5°N |
| Area | ~12,000 km² |
| Rivers covered | Rhine, Elbe, Danube (national monitoring) |
| Reference cities | Köln, Mainz, Koblenz, Dresden, Passau, Hamburg |
| CRS | EPSG:25832 (UTM Zone 32N — DE standard) |

---

## 🗂 Data Sources

| Dataset | Provider | Format | License | Used for |
|---|---|---|---|---|
| Sentinel-1 GRD (VV, VH) | ESA / Copernicus | GeoTIFF | Open, free | SAR flood segmentation |
| Copernicus DEM 30m | ESA / Copernicus | GeoTIFF | Open, free | Elevation, slope, TWI |
| DWD Open Data | Deutscher Wetterdienst | CSV / API | Open Data DE | Precipitation |
| Pegelonline WSV | Wasserstraßen- und Schifffahrtsverwaltung | REST API | Open Data DE | River levels (real-time) |
| VG250 Gemeindegrenzen | BKG | GeoPackage | Open Data DE | Municipal boundaries |
| OpenStreetMap | OSM contributors | GeoJSON | ODbL | Roads, buildings, water |

> **Data period:** 2015–2024 (historical) · 2022–2024 (LSTM training) · synthetic SAR patches (U-Net)

---

## 🧱 Project Structure

```
flood-risk-germany/
├── flood_risk_germany.py        ← Main pipeline (VS Code)
├── scheduler.py                 ← Daily automation (APScheduler)
├── .env.template                ← Secrets template
├── .gitignore
│
├── .github/
│   └── workflows/
│       └── daily_monitor.yml    ← GitHub Actions CI/CD
│
├── data/
│   ├── raw/                     ← Downloaded data (gitignored)
│   │   ├── sentinel1/
│   │   ├── dem/
│   │   └── pegelonline/
│   └── processed/               ← Reprojected rasters
│
├── outputs/
│   ├── 01_flood_eda.png         ← EDA visualizations
│   ├── 02_ml_results.png        ← ML model comparison
│   ├── 03_lstm_results.png      ← LSTM forecast results
│   ├── 04_unet_results.png      ← U-Net segmentation
│   ├── 09a_risk_maps.png        ← Germany risk maps (continuous + 5-class)
│   ├── 09c_model_comparison.png ← Dashboard (5 plots)
│   ├── 09d_flood_risk_map.html  ← Interactive Folium map
│   └── flood_risk_report.html   ← Auto-generated daily report
│
├── notebooks/
│   └── exploration.ipynb        ← Original Jupyter notebook
│
└── README.md
```

---

## 🤖 Models & Results

### Machine Learning — Flood Susceptibility Classification

Input features: elevation (m), slope (°), TWI, distance to river (km), flow accumulation

| Model | Accuracy | Notes |
|---|---|---|
| Random Forest | ~0.98 | Feature importance: TWI > dist_river > elevation |
| XGBoost | ~0.98 | Best generalization |
| Gradient Boosting | ~0.97 | Slower but robust |
| SVM (RBF) | ~0.95 | Subsample 10k for speed |

### Deep Learning — LSTM River Level Forecasting

Architecture: 3-layer LSTM (128→64→32) + Huber loss · 48h input → 24h forecast

| Metric | Value |
|---|---|
| RMSE | ~0.08–0.15 m |
| MAE | ~0.05–0.10 m |
| R² | ~0.85–0.95 |

### Deep Learning — U-Net SAR Flood Segmentation

Architecture: 4-level encoder-decoder · BCE+Dice loss · 64×64px · 3 channels (VV, VH, NDWI)

| Metric | Value |
|---|---|
| Pixel Accuracy | ~0.96 |
| Dice Coefficient | ~0.87 |
| IoU (Jaccard) | ~0.80 |
| F1-Score | ~0.87 |
| Precision | ~0.89 |
| Recall | ~0.86 |

> ⚠ **Note:** U-Net metrics are from synthetic SAR patches (elliptical flood shapes, Gaussian backscatter). On real Sentinel-1 data, expect F1 ~0.75–0.85. Next step: fine-tune on [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset.

---

## 🚀 Getting Started

### 1. Clone & setup

```bash
git clone https://github.com/ayouboihi/flood-risk-germany.git
cd flood-risk-germany
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib geopandas requests folium shapely \
            xgboost tensorflow scikit-learn python-dotenv apscheduler \
            osmnx rasterio seaborn plotly
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate flood-risk
```

### 3. Configure secrets

```bash
cp .env.template .env
# Edit .env with your Telegram token and email credentials
```

### 4. Run the pipeline

```bash
python flood_risk_germany.py
```

### 5. Run daily automation

```bash
python scheduler.py          # local — runs every day at 07:00 Berlin time
```

Or push to GitHub and use the Actions workflow (runs at 06:00 UTC daily).

---

## 📊 Outputs

| File | Description |
|---|---|
| `01_flood_eda.png` | EDA: river levels 2015–2024, DEM, TWI, risk map, seasonal patterns |
| `02_ml_results.png` | ML model accuracy comparison + Random Forest feature importance |
| `03_lstm_results.png` | LSTM training curves, actual vs predicted, RMSE per horizon |
| `04_unet_results.png` | U-Net training curves, IoU/Dice/F1 bar, segmentation overlays |
| `09a_risk_maps.png` | Germany risk map (continuous + 5-class) |
| `09c_model_comparison.png` | Dashboard: grouped bars, radar chart, heatmap, F1 ranking |
| `09d_flood_risk_map.html` | Interactive Folium map: heatmap + river gauges + layer control |
| `flood_risk_report.html` | Auto-generated daily monitoring report |

🗺 **Live map:** [ayouboihi.github.io/flood-risk-germany](https://ayouboihi.github.io/flood-risk-germany)

---

## ⚡ Automation Pipeline

```
Every day at 07:00 (Europe/Berlin)
         │
         ▼
┌─────────────────────────────────┐
│  1. Fetch Pegelonline API        │  5 stations: Köln, Mainz, Dresden, Passau, Hamburg
│  2. Check thresholds             │  warning / alarm / danger levels
│  3. Send Telegram alert          │  if alarm or danger
│  4. Generate HTML report         │  with maps + model summary
│  5. git commit + push            │  auto-deploy to GitHub Pages
└─────────────────────────────────┘
```

### Setup alerts

**Telegram:** Create bot via `@BotFather` → add `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` to `.env`

**Email (Gmail):** Enable 2FA → App Passwords → add `EMAIL_FROM` + `EMAIL_PASSWORD` to `.env`

**GitHub Actions:** Repo → Settings → Secrets and variables → Actions → add all env vars

---

## 🛠 Tech Stack

| Category | Libraries |
|---|---|
| Geospatial | GeoPandas, Rasterio, GDAL, OSMnx, Folium, Shapely |
| Machine Learning | scikit-learn, XGBoost |
| Deep Learning | TensorFlow/Keras (LSTM, U-Net) |
| Data | NumPy, Pandas, Requests |
| Visualization | Matplotlib, Seaborn, Plotly, Folium |
| Automation | APScheduler, GitHub Actions, python-dotenv |
| Remote Sensing | Sentinel-1 SAR, Copernicus DEM |

---

## 🔬 Methodology

### Risk Score Computation

```
Flood Risk Score = 0.40 × SAR flood probability (Sentinel-1)
                 + 0.35 × Topographic Wetness Index (normalized)
                 + 0.25 × Precipitation anomaly (DWD)
```

**TWI** = ln(flow_accumulation / tan(slope)) — quantifies terrain susceptibility to waterlogging

**Classes:** Very Low (0–0.2) · Low (0.2–0.4) · Medium (0.4–0.6) · High (0.6–0.8) · Very High (0.8–1.0)

### LSTM Feature Engineering

30-day / 48h lookback window with: river level lags (1h, 6h, 24h), rolling means, precipitation sums (6h, 24h), hourly + seasonal cyclical features

### U-Net Architecture

```
Input (64×64×3)
    ↓  Encoder: Conv→BN→Conv→BN×3 + MaxPool
    ↓  Bottleneck: 256 filters + Dropout(0.3)
    ↓  Decoder: ConvTranspose + Skip connections×3
    ↓  Output (64×64×1) sigmoid
Loss: BCE + Dice (handles class imbalance)
```

---

## 📋 Limitations & Next Steps

- [ ] Replace synthetic SAR data with real [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) dataset
- [ ] Add Copernicus DEM download automation via `cdsetool`
- [ ] Integrate BKG VG250 Gemeindegrenzen for true zonal statistics
- [ ] Add DWD radar precipitation (RADOLAN) as LSTM feature
- [ ] Deploy Streamlit dashboard for interactive exploration
- [ ] Fine-tune U-Net on Sen1Floods11 (target F1 > 0.80 on real data)
- [ ] Add SHAP explainability for Random Forest / XGBoost

---

## 📚 References

- Copernicus Emergency Management Service: https://emergency.copernicus.eu
- Pegelonline WSV API: https://www.pegelonline.wsv.de
- DWD Open Data: https://opendata.dwd.de
- Sen1Floods11 dataset: Bonafilia et al. (2020), CVPR Workshop
- U-Net: Ronneberger et al. (2015), MICCAI
- BKG VG250: https://www.bkg.bund.de

---

## 👤 Author

**Ayoub** — Geomatics Engineer · Remote Sensing Scientist · GIS Specialist  
📍 Heidelberg, Germany  
🔗 [github.com/ayouboihi](https://github.com/ayouboihi)

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

Data licenses: Copernicus (open), DWD (open data DE), Pegelonline WSV (open data DE), BKG VG250 (open data DE), OpenStreetMap (ODbL)
