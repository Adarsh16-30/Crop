# Crop Disease Detection from Multispectral Time Series

A research prototype that classifies crop health (**Healthy / Early Stress /
Active Disease**) from time series of Red-Edge, NIR and SWIR reflectance, with
classical and deep sequence models, quantum-inspired feature selection,
geo-spatial risk maps and a live MQTT → InfluxDB → Grafana dashboard.

> **Limitations — read first.**
> - **All data is synthetic.** It comes from `src/data/data_generator.py`, and
>   the class labels are defined by that generator. The reported accuracy
>   measures how well the models recover the generator's signal model. It is
>   **not** evidence of real-world or pre-symptomatic detection. Nothing here
>   has been validated on field data.
> - **"Quantum" means quantum-inspired.** Feature selection is a QUBO solved
>   by classical simulated annealing on a CPU. QAOA (`--use-qaoa`) runs on a
>   statevector simulator for the top-8-feature sub-problem only (about 2.5
>   minutes; its runtime grows about 5× per 2 extra variables). Its answer is
>   checked against the exact brute-force optimum, and the result is reported
>   in `qubo_comparison.json`. No quantum hardware is used.
> - **The disease labels have no spatial structure.** The generator draws
>   labels independently of location, so the risk maps show the method, not
>   real hotspots.
> - **Sensor choice is not settled.** The band set (705/740/783/842/1610/2190
>   nm) assumes a dedicated SWIR sensor alongside a VIS/NIR sensor such as
>   the AS7265x.

## Architecture

```
generator ─► preprocessing (split by sample → scaler fit on train only)
                 │
                 ├─► RF / XGBoost  (mean, std, slope, delta per feature)
                 ├─► LSTM / CNN-LSTM (+ temperature calibration)
                 ├─► QUBO feature selection vs. top-K / mRMR / random baselines
                 └─► best model ─► risk maps (Folium, metric grid)

simulator ─MQTT─► bridge ─► src.inference (indices + train scaler + model) ─► InfluxDB ─► Grafana
REST client ─────► FastAPI ─► src.inference (same code path)
```

Training, the API and the MQTT bridge all go through one inference path
(`src/inference/`). It computes the vegetation indices, applies the scaler
saved inside the checkpoint, and applies the calibrated model. There is no
separate preprocessing to keep in sync.

## Quick start

Requires Python 3.12+ (tested on 3.14). Docker is needed only for the live
dashboard.

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements-dev.txt
# optional, for --use-qaoa:  pip install -r requirements-quantum.txt
```

### Run the pipeline

```bash
python run_pipeline.py --samples 1500 --seq-len 30 --epochs 50 --skip-mqtt
```

The main flags are `--models lstm cnn_lstm`, `--max-features 8`,
`--use-qaoa`, `--grid-size-m 5000`, `--skip-quantum`, `--skip-geo`,
`--seed`, `--lr`, `--hidden-dim` and `--dropout`.

| Output | Contents |
|---|---|
| `data/processed/` | Split arrays, `scaler_params.json` (train-only fit), `sample_meta.csv`, `predictions.csv` (every sample, real id, split) |
| `models/checkpoints/*.pt` | Self-describing checkpoints: weights, architecture, scaler, temperature, versions. Load with `weights_only=True`. |
| `reports/model_comparison.csv` | Test metrics with 95% bootstrap CIs |
| `reports/model_significance.csv` | Pairwise McNemar tests |
| `reports/classical_cv.csv` | 5-fold CV, random vs. spatial-zone hold-out |
| `reports/feature_selection_comparison.csv` | QUBO vs. top-K, mRMR and random-K |
| `reports/disease_risk_map.html`, `zone_risk_map.html` | Interactive maps |

### Live dashboard

```bash
cp .env.example .env              # then replace every "change-me" value
bash scripts/gen_mqtt_certs.sh    # local CA + broker TLS certificate (certs/mqtt/)
docker compose up -d              # InfluxDB, Grafana, Mosquitto (bound to 127.0.0.1)
python -m src.dashboard.influxdb_setup --create-tokens   # paste tokens into .env
docker compose --profile app up -d                      # API, bridge, simulator
```

You can also run the services directly:

```bash
python -m src.dashboard.mqtt_subscriber --model-path models/checkpoints/lstm_best.pt
python -m src.data.mqtt_publisher --profile mixed --interval 1
uvicorn src.api.main:app --port 8000
```

Grafana runs at http://localhost:3000. Four linked dashboards are provisioned
in the **Crop Disease** folder:

| Dashboard | Shows |
|---|---|
| **Field Overview** | KPIs (stations, readings, predictions, mean risk, high-risk stations, live agreement with ground truth), a station map colored by latest risk, the latest prediction per station, risk over time, a class timeline per station, the prediction mix, class probabilities and active alerts |
| **Spectral Analysis** | Band reflectance per station (one panel per station), six vegetation indices over time, an NDVI heatmap, NDVI distribution by true profile, and the latest indices per station |
| **Model Performance** | Live: rolling agreement with ground truth, live confusion matrix, confidence distribution and out-of-distribution count. Offline, per pipeline run: model metrics with 95% CIs, McNemar tests, training curves by epoch, random vs spatial CV, QUBO vs baseline feature selection, the batch risk map, and macro-F1 across runs |
| **System Health** | Data freshness, ingest rate, station heartbeat, data by source, bridge throughput and error counters, and per-station freshness |

Every live panel can be filtered by **Station** and **Source**. Source is
`simulator` for the live MQTT stream and `backfill` for generated history.
Three alert rules are provisioned:

- High risk at a station: latest risk above 0.7.
- No sensor readings for 5 minutes.
- Bridge dropping payloads or failing writes.

Point them at your own contact point (email, Slack and so on) in
Grafana → Alerting. Out of the box Grafana tries email, which needs SMTP.

Where the data comes from:

```bash
# Live: the bridge writes spectral_readings, disease_predictions and bridge_stats
# History: 12 h of past readings through the same bridge code path (source=backfill)
python -m src.dashboard.seed_history --model-path models/checkpoints/cnn_lstm_best.pt --hours 12
python -m src.dashboard.seed_history --clear        # remove it again (uses INFLUX_TOKEN)
# Offline results: run_pipeline.py publishes these automatically when InfluxDB is
# reachable; to publish an existing run by hand:
python -m src.dashboard.publish_pipeline_metrics
```

The dashboards are generated from `src/dashboard/grafana/build_dashboards.py`.
Edit that file, not the JSON:

```bash
python -m src.dashboard.grafana.build_dashboards
```

After regenerating, restart Grafana (`docker compose restart grafana`) if the
change doesn't show up. Grafana polls the files every 30 seconds, but on
Docker Desktop for Windows file changes in bind mounts are not always
detected.

Each simulated station publishes one complete 30-reading crop cycle at a
time. The bridge predicts once per completed cycle, so with 5 stations at 1 s
intervals the first live predictions appear after about 2.5 minutes. Run the
backfill to have history immediately.

Security defaults:

- Every credential is read from `.env`, never from the repo.
- MQTT is TLS-only: the broker listens on port 8883 with no plaintext
  listener, and requires TLS 1.2 or newer (TLS 1.3 is negotiated in practice).
  Clients trust only `certs/mqtt/ca.crt` and verify the broker's hostname.
  The certificate covers `mosquitto`, `localhost` and `127.0.0.1`; add more
  names with `MQTT_CERT_HOSTS="host1,10.0.0.5" bash scripts/gen_mqtt_certs.sh --force`.
  Keep `certs/mqtt/ca.key` private. The `certs/` folder is excluded from git
  and from the Docker image.
- MQTT also requires a username and password, with ACLs limiting access to `crop/#`.
- The bridge writes with a write-only InfluxDB token, and Grafana reads with a read-only one.
- The API requires `X-API-Key`. For local development only, set `API_ALLOW_UNAUTHENTICATED=true`.

## API

The API takes **raw band reflectances**, exactly `seq_len` rows of 6 values
in [0, 1]. Band order is `red_edge_705, red_edge_740, red_edge_783, nir_842,
swir_1610, swir_2190`. The server computes the indices and the scaling.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"readings": [[0.35,0.55,0.65,0.80,0.20,0.15], ... 30 rows ...]}'
```

```json
{
  "predicted_label": 0,
  "label_name": "Healthy",
  "confidence": 0.93,
  "risk_score": 0.05,
  "probabilities": [0.93, 0.04, 0.03],
  "status": "ok",
  "out_of_distribution": false,
  "model_type": "lstm",
  "inference_time_ms": 3.1
}
```

The values above are illustrative.

- **Probabilities and risk.** Probabilities are temperature-calibrated.
  `risk_score` is the probability-weighted risk Σ pᵢ·wᵢ, with w = (0, 0.5, 1).
- **`status`** is one of:
  - `ok`
  - `low_confidence`: confidence is below `LOW_CONFIDENCE_THRESHOLD`
  - `out_of_distribution`: any standardized feature has |z| > 8
- **Other endpoints:**
  - `POST /predict/batch`
  - `GET /info`: model metadata, including `seq_len`
  - `GET /health`: 503 when no model is loaded
  - `GET /metrics`: Prometheus
- **Limits:** rate limiting, a body-size cap, a batch-size cap and CORS are
  all configured by environment variables (see `src/api/main.py`).

## Results (synthetic data)

These numbers come from `python run_pipeline.py --samples 1500 --seq-len 30 --epochs 50 --seed 42`.
The test set is 225 samples, and the intervals are 95% bootstrap CIs. Rerunning
produces `reports/model_comparison.csv`.

| Model | Accuracy [95% CI] | Macro-F1 | Early-Stress F1 |
|---|---|---|---|
| CNN-LSTM | 0.969 [0.942, 0.991] | 0.971 | 0.944 |
| Random Forest | 0.960 [0.933, 0.982] | 0.956 | 0.927 |
| XGBoost | 0.951 [0.920, 0.978] | 0.945 | 0.909 |
| RF, 8 QUBO-selected features | 0.942 [0.911, 0.969] | 0.935 | 0.894 |
| LSTM | 0.933 [0.898, 0.964] | 0.940 | 0.887 |

How to read these results:

- **Most differences are not significant.** McNemar tests
  (`reports/model_significance.csv`) find only one significant pair:
  CNN-LSTM beats LSTM, p = 0.04. Every other pair, including CNN-LSTM vs.
  Random Forest, is within noise. Once the classical models get trend
  features (slope and delta), they are as good as the sequence models on
  this data.
- **Spatial CV matches random CV** (Random Forest macro-F1 0.970 vs. 0.968).
  This is expected, because the generator's labels have no spatial
  structure. It is a sanity check, not evidence of geographic generalization.
- **QUBO feature selection** (8 of 48 features) is compared with the same
  number of features from simpler methods (5-fold CV macro-F1 / test
  macro-F1):

  | Selection method | CV macro-F1 | Test macro-F1 |
  |---|---|---|
  | QUBO | 0.956 | 0.935 |
  | mRMR | 0.945 | 0.935 |
  | Top-K by importance | 0.945 | 0.919 |
  | Random K | 0.883 | 0.882 |
  | All 48 features | 0.965 | 0.956 |

  QUBO is competitive with standard selectors but does not beat using all
  features.
- **None of this transfers to real fields** until validated on field data
  (see Limitations).

## Tests

```bash
pytest -m "not slow"          # ~110 fast tests: inference parity, API, streaming, QUBO, geo
pytest -m slow                # end-to-end pipeline on a tiny dataset (~1.5 min)
pytest --cov                  # coverage
```

The API tests run against a real, freshly trained model. The parity test
asserts that raw readings sent through the API path produce the same
probabilities as the offline preprocessed path.

## Project structure

```
config/            non-secret INI settings, Mosquitto config + ACL
scripts/           gen_mqtt_certs.sh (MQTT TLS certificates)
src/
  constants.py     bands, labels, risk weights (single source of truth)
  config.py        INI + env loading, MQTT client factory
  data/            generator, preprocessing, MQTT simulator
  features/        vegetation indices (array/DataFrame, finite-checked)
  models/          LSTM, CNN-LSTM, RF/XGBoost + CV, model factory
  training/        training loop, calibration, self-describing checkpoints
  inference/       feature pipeline + predictor (used by API, bridge, pipeline)
  evaluation/      metrics, bootstrap CIs, McNemar, plots
  quantum/         QUBO formulation, solvers, baseline comparison
  geospatial/      risk GeoDataFrames, metric zone grid, Folium maps
  api/             FastAPI service
  dashboard/       MQTT→InfluxDB bridge, InfluxDB setup, history backfill,
                   pipeline-metrics export, Grafana dashboards-as-code + provisioning
run_pipeline.py    end-to-end runner
tests/             pytest suite
```

## Disease classes

| Class | Label | Meaning in the generator |
|---|---|---|
| 0 | Healthy | No disease trajectory |
| 1 | Early Stress | Late, slow, low-amplitude logistic decline |
| 2 | Active Disease | Earlier, faster, stronger decline |
