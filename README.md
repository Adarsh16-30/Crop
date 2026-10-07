# 🌱 Quantum-Enhanced Crop Disease Detection System

**Pre-symptomatic crop disease detection** using multispectral sensing, temporal AI models, quantum-inspired optimization, and real-time geospatial monitoring.

## Overview

This system detects crop diseases **before visible symptoms appear** by analyzing Red-Edge and SWIR spectral band reflectance patterns over time. It combines:

- **Synthetic multispectral sensor data** (simulating AS7265x + SWIR sensors)
- **Classical ML** (Random Forest, XGBoost) and **Deep Learning** (LSTM, CNN-LSTM) models
- **Quantum-inspired QUBO optimization** for feature selection via simulated annealing
- **Geo-spatial risk mapping** with interactive Folium heatmaps
- **Real-time Grafana dashboard** with InfluxDB time-series storage and MQTT data streaming
- **FastAPI inference endpoint** for on-demand predictions

## Architecture

```
Synthetic Data → MQTT (Mosquitto) → InfluxDB → Grafana Dashboard
                                  ↓
              Preprocessing → ML/DL Models → QUBO Optimization
                                  ↓
                        Geo-Spatial Risk Maps → Folium Heatmaps
                                  ↓
                           FastAPI REST API
```

## Prerequisites

- **Python 3.14.2**
- **Docker Desktop** (for InfluxDB, Grafana, Mosquitto)
- **CUDA GPU** (optional, for faster training)

## Quick Start

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For CUDA GPU support (optional):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Start Infrastructure Services

```bash
docker compose up -d
```

This starts:
- **InfluxDB** → `http://localhost:8086` (admin / cropdetection2024)
- **Grafana** → `http://localhost:3000` (admin / cropdetection)
- **Mosquitto** → `localhost:1883`

### 3. Run the Full Pipeline

```bash
python run_pipeline.py --samples 500 --seq-len 30 --epochs 50
```

This executes all phases:
1. Generates synthetic spectral data
2. Trains RF, XGBoost, LSTM, CNN-LSTM models
3. Runs QUBO quantum feature selection
4. Creates geo-spatial risk maps
5. Prepares dashboard data

### 4. Start Real-Time Monitoring

```bash
# Terminal 1: MQTT→InfluxDB bridge (with inference)
python -m src.dashboard.mqtt_subscriber --model-path models/checkpoints/lstm_best.pt

# Terminal 2: Simulated sensor publisher
python -m src.data.mqtt_publisher --profile mixed --interval 2

# Terminal 3: FastAPI endpoint
uvicorn src.api.main:app --reload --port 8000
```

Then open:
- **Grafana Dashboard**: http://localhost:3000
- **FastAPI Docs**: http://localhost:8000/docs
- **Risk Map**: Open `reports/disease_risk_map.html` in browser

## Project Structure

```
quantum-crop-disease/
├── config/
│   ├── influxdb.ini              # InfluxDB connection config
│   ├── mqtt.ini                  # MQTT broker config
│   └── mosquitto.conf            # Mosquitto broker config
├── data/
│   ├── raw/                      # Synthetic CSVs
│   └── processed/                # Preprocessed numpy arrays
├── docker-compose.yml            # InfluxDB + Grafana + Mosquitto
├── models/
│   └── checkpoints/              # Saved model weights
├── notebooks/
│   ├── 01_eda_and_visualization.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_qubo_experiments.ipynb
├── reports/
│   ├── disease_risk_map.html     # Interactive Folium map
│   ├── model_comparison.csv      # Model comparison table
│   └── qubo_feature_selection_results.csv
├── run_pipeline.py               # End-to-end pipeline runner
├── requirements.txt
├── src/
│   ├── api/main.py               # FastAPI inference endpoint
│   ├── dashboard/
│   │   ├── grafana/              # Grafana provisioning configs
│   │   ├── influxdb_setup.py     # InfluxDB initialization
│   │   └── mqtt_subscriber.py    # MQTT→InfluxDB bridge
│   ├── data/
│   │   ├── data_generator.py     # Synthetic data generation
│   │   ├── mqtt_publisher.py     # Simulated sensor publisher
│   │   └── preprocessor.py       # Data preprocessing pipeline
│   ├── evaluation/evaluate.py    # Model evaluation & comparison
│   ├── features/spectral_indices.py  # Vegetation index computation
│   ├── geospatial/
│   │   ├── heatmap.py            # Folium map generation
│   │   └── risk_mapper.py        # GeoDataFrame & risk scoring
│   ├── models/
│   │   ├── classical_models.py   # RF & XGBoost
│   │   ├── cnn_lstm_model.py     # CNN-LSTM hybrid
│   │   └── lstm_model.py         # SpectralLSTM
│   ├── quantum/
│   │   ├── annealing_optimizer.py # SA & QAOA solvers
│   │   ├── integration.py        # QUBO pipeline orchestration
│   │   └── qubo_feature_selector.py # QUBO formulation
│   └── training/train.py         # Unified training script
└── tests/                        # pytest test suite
```

## Disease Classes

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Healthy | Normal crop reflectance |
| 1 | Early Stress | Pre-symptomatic — spectral anomalies detected |
| 2 | Active Disease | Visible disease symptoms, confirmed spectral degradation |

## API Usage

### Predict Disease

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": [[0.35,0.55,0.65,0.80,0.20,0.15,0.45,0.60,0.90,0.78,0.25,1.03]] * 30}'
```

Response:
```json
{
  "predicted_label": 0,
  "label_name": "Healthy",
  "confidence": 0.9234,
  "risk_score": 0.0,
  "probabilities": [0.9234, 0.0521, 0.0245],
  "inference_time_ms": 12.5
}
```

## Running Tests

```bash
# Full test suite
pytest tests/ -v --cov=src --cov-report=html

# Specific modules
pytest tests/test_data.py -v
pytest tests/test_models.py -v
pytest tests/test_quantum.py -v
```

## Key Technologies

| Layer | Technology |
|-------|-----------|
| ML/DL | PyTorch, Scikit-Learn, XGBoost |
| Quantum | Qiskit, D-Wave Neal (simulated annealing) |
| Geo-Spatial | GeoPandas, Folium, Shapely |
| Dashboard | Grafana, InfluxDB, MQTT |
| API | FastAPI |

## Future: Hardware Integration

When the software stack is validated, hardware sensors can replace the synthetic data:
- **AS7265x** multispectral sensor → replace MQTT publisher
- **Raspberry Pi / Jetson Nano** → edge inference via ONNX
- **Neo-6M GPS** → real GPS coordinates
- **LoRa/GSM** → remote data transmission

---

*Quantum-Enhanced Edge-AI Crop Disease Detection System — Software Phase v1.0*
