# Project Guide — Crop Disease Detection from Multispectral Time Series

This guide explains how the project works end to end. It covers what every
file does, how data moves through the offline pipeline and the live
streaming stack, and what could be built next. For quick-start commands and
the headline results, see [README.md](README.md).

---

## 1. What the project does

The system classifies the health of a crop as **Healthy (0)**, **Early Stress
(1)** or **Active Disease (2)**. Its input is a *time series* of reflectance
readings in six spectral bands:

| Band | Wavelength | Why it matters |
|---|---|---|
| `red_edge_705`, `red_edge_740`, `red_edge_783` | Red-edge | Chlorophyll content; the red edge shifts toward shorter wavelengths under stress |
| `nir_842` | Near-infrared | Leaf cell structure and canopy vigour |
| `swir_1610`, `swir_2190` | Short-wave infrared | Leaf water content (moisture stress) |

The project has two halves that share **one inference code path**:

1. **Offline research pipeline** (`run_pipeline.py`):
   - Loads real Sentinel-2 sequences labelled by CDM outbreak reports
     (`src/real_data/`), preprocesses them with a site-grouped split, tunes
     every model by site-grouped cross-validation, then trains classical and
     deep models (LSTM, CNN-LSTM, CNN-LSTM-Attention).
   - Runs quantum-inspired feature selection, evaluates everything with
     confidence intervals and significance tests, and builds geospatial risk
     maps.
2. **Live system** (Docker Compose):
   - Field stations replay real Sentinel-2 sequences of held-out CDM sites
     over encrypted MQTT.
   - A bridge assembles each station's crop cycle, runs the trained model and
     writes to InfluxDB. Grafana visualises the data and raises alerts.
   - A FastAPI service exposes the same model over REST.

> The labels are weak: CDM reports are positives only, so classes are defined
> by time relative to the report date (see §2.1, Phase 1). A calendar-only
> baseline is part of every comparison. See the limitations in the README.

---

## 2. How the pipeline works

### 2.1 Offline pipeline — `run_pipeline.py`

```
 Phase 1  CDM + Sentinel-2 sequences ─► preprocess (split by SITE, scaler on train only)
                                        │
             ┌──────────────────────────┼────────────────────────────┐
             ▼                          ▼                            ▼
 Phase 2  RF / XGBoost         Phase 3  LSTM / CNN-LSTM /   Phase 4  QUBO feature selection
          grouped-CV tuning,            CNN-LSTM-Attention           (simulated annealing,
          site & spatial CV,            grouped-CV tuning,           optional QAOA) vs top-K,
          context-only and              early stopping, temp.        mRMR, random-K, all
          spectral-only ablations       calibration                  features
             └──────────────────────────┬────────────────────────────┘
                                        ▼
                    compare_models: metrics + 95% bootstrap CIs + McNemar tests
                                        ▼
 Phase 5  best sequence model (by val macro-F1) ─► predictions.csv ─► Folium risk maps + zone GeoJSON
                                        ▼
 Phase 6  InfluxDB check ─► publish run metrics for Grafana (Model Performance dashboard)
```

#### Phase 1 — Data foundation

1. **Load the real sequences.** `python -m src.real_data temporal` builds
   `data/real/cdm_s2/cdm_s2_sequences.csv`: for every CDM outbreak report
   with GPS, the last 5 QC-passing Sentinel-2 acquisitions up to each anchor
   date, labelled by the anchor's offset d from the report date:
   d < −21 days → Healthy, −21 ≤ d < 0 → Early Stress, 0 ≤ d ≤ 45 → Active
   Disease. B05/B06/B07/B08/B11/B12 are exactly the six project bands.
2. **Preprocess.** `src/data/preprocessor.py`:
   - Computes the six vegetation indices, giving 6 bands + 6 indices = **12
     features per timestep**.
   - Splits **by site** (`group_col='site_id'`, label-stratified ~70/15/15),
     so overlapping sequences of one field never span two splits.
   - Fits the `StandardScaler` on **training rows only**.
   - Appends 4 **acquisition-context channels** to every step
     (`src/features/context.py`): day of year as sine/cosine, latitude and
     longitude. `--no-context` turns them off.
   - Builds two views of the same split:
     - **sequences** `(n, seq_len, 16)` for the deep models (6 bands,
       6 indices, 4 context);
     - **aggregated features** `(n, 52)`: mean, std, slope and delta of each
       of the 12 spectral features, plus the context at the last
       acquisition, for RF/XGBoost and QUBO.
   - Drops samples shorter than `seq_len`; they are never zero-padded.
   - Saves the arrays, `scaler_params.json` and `sample_meta.csv` (id,
     label, lat, lon) to `data/processed/`.

#### Phase 2 — Classical models

- **Tunes** Random Forest (depth, leaf size, features per split) and XGBoost
  (depth, learning rate, trees, min child weight, L2) by site-grouped 5-fold
  CV on train+val (`src/training/tuning.py`, see §6), then fits the chosen
  configuration on train+val with balanced class weights and evaluates it
  on the test sites.
- Runs 5-fold CV twice with the tuned settings: **site** hold-out and
  **spatial-zone** hold-out (~0.1° GPS zones from `constants.zone_key`).
- Adds two **ablations**: an RF on the context only (date + location, no
  spectra) and an RF on the spectral features only. Because the labels are
  time-relative, the first is the score a model gets from the calendar and
  map alone; the second is what the spectra give on their own.
- Writes `reports/classical_cv.csv` and `reports/ablation_cv.csv`.

#### Phase 3 — Deep models

0. **Tune** each of LSTM, CNN-LSTM and CNN-LSTM-Attention over hidden size,
   dropout, weight decay and input-noise augmentation by site-grouped 5-fold
   CV; early stopping inside each fold uses an inner site-grouped hold-out,
   never the scored fold.
1. `src/training/train.py`:
   - Seeds everything.
   - Trains with class-weighted cross-entropy (label smoothing 0.05), AdamW,
     LR scheduling, gradient clipping, optional Gaussian input noise and
     early stopping.
   - Checkpoints on the best **validation macro-F1**.
2. **Calibrate.** A single temperature `T` is fitted on validation logits
   (Guo et al., 2017), so the probabilities are calibrated. It does not
   change which class is predicted.
3. **Save a self-describing checkpoint.** It holds the weights plus
   architecture, `seq_len`, the scaler, the temperature, label names and
   library versions, all as plain types. Load it with
   `torch.load(weights_only=True)`.
4. **Evaluate and plot.** The test set is scored through `Predictor`, and the
   confusion matrix, ROC curves and training curves are saved.

#### Phase 4 — Quantum-inspired feature selection

1. **Relevance:** Random Forest importances on the training set.
   **Redundancy:** |correlation| between the 48 features on the training set.
2. **Build the QUBO:**

   ```
   E(x) = −Σ rᵢxᵢ  +  λ Σ_{|cᵢⱼ|>τ} |cᵢⱼ| xᵢxⱼ  +  P (Σxᵢ − K)²
   ```

   The cardinality penalty `P` makes "exactly K features" the cheapest
   outcome, so no post-hoc truncation is needed.
3. **Solve** with simulated annealing (`dwave.samplers`). If no annealer is
   installed, it falls back to a NumPy solver.
4. **Optional QAOA** (`--use-qaoa`, Qiskit statevector simulator): runs on
   the top 8 features only and is checked against the exact brute-force
   optimum.
5. **Compare** with the same K from top-K by importance, greedy mRMR and
   random-K, plus all features, using CV and the test set. The RF trained on
   the QUBO subset is saved and added to the model comparison.

#### Comparison

`src/evaluation/evaluate.py` reports accuracy, macro/weighted F1 and per-class
F1, with **95% bootstrap confidence intervals**. It also runs **pairwise
McNemar tests**, written to `model_comparison.csv` and
`model_significance.csv`.

#### Phase 5 — Geospatial

1. **Predict.** The sequence model with the best validation macro-F1
   predicts every sample in every split, keeping its real id and split.
2. **Score risk:**

   ```
   risk = Σ pᵢ·wᵢ,  w = (0, 0.5, 1)
   ```

   and bin it into Low / Moderate / High / Critical.
3. **Aggregate and map.** Samples are grouped onto a metric grid
   (`--grid-size-m`, a projected CRS). Folium maps use OpenStreetMap and Esri
   imagery, and the zones are exported as GeoJSON.

#### Phase 6 — Dashboard

Checks that InfluxDB is reachable, then publishes this run's reports as
InfluxDB points (`publish_pipeline_metrics`). Each run gets a `run_id` tag
that the Model Performance dashboard can select. Skip with `--skip-mqtt`.

### 2.2 Live streaming pipeline

```
 mqtt_publisher (replay)               Mosquitto (TLS 8883, auth, ACL)
 5 real test sites, one 5-step ──MQTT──►  topic crop/spectral  ──►  mqtt_subscriber (bridge)
 real sequence at a time, QoS 1                                        │
                                                                       │ validate (pydantic)
                                                                       │ compute indices
                                                                       │ SequenceAssembler: wait for
                                                                       │   timesteps 0..seq_len-1 of
                                                                       │   one (station, cycle)
                                                                       │ Predictor.predict_raw
                                                                       ▼
                       Grafana  ◄──Flux (read-only token)──  InfluxDB (write-only token)
                       4 dashboards + 3 alert rules          spectral_readings / disease_predictions / bridge_stats

 REST client ──X-API-Key──► FastAPI /predict ──► Predictor.predict_raw (same code path)
```

**The replay publisher (`src/data/mqtt_publisher.py`):**
- Each station is a real CDM site from the held-out **test** split (read
  from `data/processed/sample_meta.csv`), at its real coordinates.
- A "crop cycle" is one of that site's real labelled Sentinel-2 sequences,
  replayed one reading per timestep, in a seeded random order.
- Each payload carries station id, cycle id, timestep, coordinates, the six
  bands, a timestamp, the sequence's CDM label as `profile`, its
  `sample_id` and `source=simulator`.
- It waits until the TLS connection is up before publishing.

**The bridge (`src/dashboard/mqtt_subscriber.py`):**
- Uses a persistent MQTT session (fixed client id, `clean_session=False`), so
  QoS 1 messages are queued by the broker while it restarts.
- Rejects invalid payloads and counts them.
- Every reading becomes a `spectral_readings` point.
- When a cycle is complete, the model runs and a `disease_predictions` point
  is written: label, confidence, risk, class probabilities and an OOD flag.
  For replayed sequences it also records their CDM label (`true_label`,
  `correct`), which is what makes live agreement measurable.
- Points use the **sensor's** timestamp, so delayed data lands at the right
  time.
- Counters go to `bridge_stats`.

**The shared inference path (`src/inference/`):**
- `FeaturePipeline` turns raw band readings into indices, then
  standardised features using the scaler saved inside the checkpoint.
- `Predictor` applies the model, temperature scaling and the risk score.
- It flags an input **out-of-distribution** if any standardised feature has
  |z| > 8.
- The API, the bridge, the backfill tool and the pipeline all use it, and a
  parity test asserts the API path matches the offline path.

**Getting data into Grafana quickly (`src/dashboard/seed_history.py`):** live
data alone yields only one prediction per station every 2.5 minutes. The
backfill tool replays N hours of history through the *same* bridge function
(`reading_to_records`), tagged `source=backfill`.

### 2.3 Security model

| Layer | Measure |
|---|---|
| Secrets | Read only from `.env` (git-ignored); `.env.example` documents them |
| MQTT | TLS 1.2+ only, no plaintext listener. Private CA from `scripts/gen_mqtt_certs.sh`. Clients verify the CA and hostname. Username/password required, with an ACL limiting access to `crop/#` |
| InfluxDB | The bridge uses a **write-only** token, Grafana a **read-only** token, and the admin token is used only for setup and deletes |
| API | `X-API-Key` auth (constant-time compare), per-client rate limit, body-size cap, batch-size cap, CORS off by default, Prometheus `/metrics` |
| Network | All Docker ports bound to `127.0.0.1` |
| Checkpoints | Loaded with `weights_only=True`, so no arbitrary pickle code runs |

---

## 3. File-by-file reference

### 3.1 Root

| File | Purpose |
|---|---|
| `run_pipeline.py` | End-to-end offline runner (Phases 1–6, flags in §5) |
| `README.md` | Quick start, API contract, results, limitations |
| `PROJECT_GUIDE.md` | This document |
| `docker-compose.yml` | InfluxDB 2.7, Grafana 10.4 and Mosquitto 2.0 (default profile), plus the API, bridge and replay publisher (`--profile app`). Mosquitto generates its password file and copies the ACL and certificates into its volume at startup |
| `Dockerfile` | Image for the API, bridge and replay publisher (installs `requirements.txt`) |
| `.env.example` | Template for all secrets: Influx admin and tokens, Grafana admin, MQTT user, API keys |
| `.gitignore` / `.dockerignore` | Keep data, checkpoints, reports, `.env` and `certs/` out of git and the image |
| `.gitattributes` | Forces LF endings for shell scripts and broker config (needed inside Linux containers) |
| `pyproject.toml` | Project metadata, ruff and pytest configuration (markers: `slow`) |
| `requirements.txt` | Every dependency of the project in one file: pipeline, real-data acquisition (rasterio, pystac-client, OpenCV), API and streaming, Qiskit for `--use-qaoa`, and the test / lint tooling |
| `.pre-commit-config.yaml` | Ruff lint and format hooks |
| `.github/workflows/ci.yml` | CI: lint plus tests |

### 3.2 `config/`

| File | Purpose |
|---|---|
| `mqtt.ini` | Broker host and port (8883), topic, QoS, keepalive, client id, CA path. Credentials come from env |
| `influxdb.ini` | URL, org and bucket. Tokens come from env (`INFLUX_TOKEN`, `INFLUX_WRITE_TOKEN`, `INFLUX_READ_TOKEN`) |
| `mosquitto.conf` | TLS-only listener on 8883, `allow_anonymous false`, password and ACL files, persistence |
| `mosquitto.acl` | The configured user may only read and write `crop/#` |

### 3.3 `scripts/`

| File | Purpose |
|---|---|
| `gen_mqtt_certs.sh` | Creates a local CA and a broker certificate in `certs/mqtt/`. Default SANs: `mosquitto`, `localhost`, `127.0.0.1`. Add more with `MQTT_CERT_HOSTS`; regenerate with `--force` |

### 3.4 `src/` — shared foundations

| File | Purpose |
|---|---|
| `constants.py` | Single source of truth for `BANDS`, `INDICES`, `LABEL_NAMES`, `PROFILE_LABELS`, `NUM_CLASSES`, `RISK_WEIGHTS`, `RISK_BINS` and `RISK_CATEGORIES`, plus `expected_risk(probs)` and `zone_key(lat, lon)` (the ~0.1° spatial group) |
| `config.py` | Loads INI files and overlays env variables (`load_mqtt_config`, `load_influx_config(token_env=…)`). Also `make_tls_context` (TLS 1.2+, hostname check, `CERT_REQUIRED`) and `make_mqtt_client` (paho with auth and TLS) |

### 3.5 `src/data/`

| File | Purpose |
|---|---|
| `preprocessor.py` | `split_sample_ids` (stratified split by sample, or by group with `group_col`), `normalize_features` (train-only fit), `create_sequences`, `create_aggregated_features` (mean/std/slope/delta, giving 48 features), `run_preprocessing_pipeline` (writes `data/processed/`) |
| `mqtt_publisher.py` | `SpectralPublisher`: replays real Sentinel-2 sequences of held-out CDM sites as field stations. `load_replay_sequences(csv, sample_meta, split)`. CLI: `--sequences`, `--split test\|val\|train\|all`, `--interval`, `--stations`, `--seed` |

### 3.6 `src/features/`

| File | Purpose |
|---|---|
| `context.py` | Acquisition-context channels (`CONTEXT_FEATURES`: day-of-year sine/cosine, lat, lon). `context_features(dates, lat, lon)` for arrays (inference), `add_context_columns(df)` for the long table (preprocessing) |
| `spectral_indices.py` | NDRE, NDVI (red-edge variant), MSI, SWIR ratio, REIP (stable linear interpolation clipped to 705–783 nm) and CRI. Denominators are floored, so results are always finite. `compute_indices(df)` for DataFrames, `compute_index_values(bands)` for one reading, `raw_to_features(raw)` for arrays, `get_feature_columns()` for the canonical order |

### 3.7 `src/models/`

| File | Purpose |
|---|---|
| `lstm_model.py` | `SpectralLSTM`: a stacked LSTM (default 2 layers). The final hidden state feeds an MLP head (64, ReLU, dropout, then classes). Xavier and orthogonal initialisation |
| `cnn_lstm_model.py` | `CNNLSTMModel`: Conv1d blocks over time (channels hidden/2 and hidden, BatchNorm, ReLU, dropout), an LSTM, then pooling: **additive temporal attention** (`attention=True`, the CNN-LSTM-Attention model; `forward(x, return_attention=True)` returns the weights) or the last LSTM output (`attention=False`, the plain CNN-LSTM ablation). Widths scale with the hidden size |
| `factory.py` | `MODEL_TYPES = lstm, cnn_lstm, cnn_lstm_attention`, `DISPLAY_NAMES`, `build_model(type, …)`, `build_model_from_config(config)` (rebuilds a model from checkpoint metadata, including conv widths) and `count_parameters` |
| `classical_models.py` | RF and XGBoost factories (depth, leaf size, max features, min child weight, L2 all configurable) with balanced sample weights, `cross_validate` (random or grouped), evaluation, joblib save and load, and feature importances |

### 3.8 `src/training/`

| File | Purpose |
|---|---|
| `train.py` | `set_seed`, `get_device`, `compute_class_weights` (from `y_train`), data loaders, `train_one_epoch` (optional Gaussian input noise), `validate`, `fit_temperature` (LBFGS on log-T), `train_model` (macro-F1 checkpointing, early stopping, CSV training log), `train_from_arrays` (used by the pipeline) and `load_checkpoint` |
| `tuning.py` | Overfitting-aware hyperparameter search: `grouped_folds` (site-grouped, stratified), `tune_classical` and `tune_sequence_model` (grids `RF_GRID`, `XGB_GRID`, `DL_GRID`; train and CV macro-F1 per fold), `one_standard_error_choice` (simplest configuration within 1 SE of the best), `save_tuning` (`reports/hyperparameter_tuning.csv`) |

### 3.9 `src/inference/`

| File | Purpose |
|---|---|
| `pipeline.py` | `FeaturePipeline`: the **only** place raw readings become model input (indices, then the training scaler). Also checks out-of-distribution inputs (|z| > `DEFAULT_OOD_Z` = 8) |
| `predictor.py` | `load_checkpoint_dict` (safe load) and `Predictor`: `predict_raw` (raw bands + acquisition dates and location → calibrated probabilities, label, confidence, risk, OOD flag), `predict_proba_scaled` (already-preprocessed arrays), plus `seq_len`, `uses_context`, `model_type` and training metrics |

### 3.10 `src/evaluation/`

| File | Purpose |
|---|---|
| `evaluate.py` | `compute_metrics` (accuracy, F1 variants, per-class F1, confusion matrix), `bootstrap_ci`, `mcnemar_test`, `evaluate_dl_model`, plots (confusion matrix, ROC, training curves) and `compare_models` (CSV, plot and pairwise significance) |

### 3.11 `src/quantum/`

| File | Purpose |
|---|---|
| `qubo_feature_selector.py` | Relevance and correlation inputs, `build_qubo_matrix` (relevance + redundancy + cardinality penalty in D-Wave dict format), `qubo_energy`, an optional Qiskit `QuadraticProgram`, and result interpretation |
| `annealing_optimizer.py` | Solvers: `run_simulated_annealing` (dwave.samplers / neal), `run_qaoa` (Qiskit statevector), `numpy_fallback`, brute force for ≤ 20 variables and cardinality repair. `get_solver_info()` reports which backends are available |
| `integration.py` | `run_quantum_feature_selection`: the full Phase 4 flow, baselines (`select_top_k`, `select_mrmr`, random-K, all), QAOA on the top-8 sub-problem (`QAOA_MAX_CANDIDATES`) vs the exact optimum. Writes `qubo_comparison.json` and `feature_selection_comparison.csv` and saves `rf_qubo` |

### 3.12 `src/geospatial/`

| File | Purpose |
|---|---|
| `risk_mapper.py` | `build_geodataframe` (points, expected risk, category), `aggregate_zone_risk` (metric grid in a projected CRS), `export_geojson`, `get_risk_summary` |
| `heatmap.py` | Folium maps: `generate_risk_heatmap` (heat layer plus coloured sample markers with id and split) and `generate_zone_map` (zone choropleth). Base layers are OpenStreetMap and Esri imagery |

### 3.13 `src/api/`

| File | Purpose |
|---|---|
| `main.py` | FastAPI app. Pydantic contracts (`SpectralInput`: exactly `seq_len` × 6 values in [0, 1]), `Settings` from env, API-key auth, `RateLimiter`, `BodySizeLimitMiddleware`, CORS and Prometheus metrics. Endpoints: `/predict`, `/predict/batch`, `/info`, `/health`, `/metrics`. Status is `ok`, `low_confidence` or `out_of_distribution` |

### 3.14 `src/dashboard/`

| File | Purpose |
|---|---|
| `mqtt_subscriber.py` | The bridge. `SensorReading` (payload schema), `SequenceAssembler` (one sequence per station and cycle; discards abandoned cycles), `build_reading_point` / `build_prediction_point` (sensor timestamp, coordinates, ground truth), `reading_to_records` (shared with the backfill), and `MQTTInfluxBridge` (persistent session, batched writes with retries, stats, clean shutdown) |
| `influxdb_setup.py` | Checks InfluxDB health and the bucket. `--create-tokens` mints the scoped write-only and read-only tokens to paste into `.env` |
| `seed_history.py` | Backfills N hours of replayed real sequences (`source=backfill`) through the real bridge path. `--clear` deletes it (needs the admin token) |
| `publish_pipeline_metrics.py` | Turns `reports/*.csv` and `predictions.csv` into InfluxDB measurements `pipeline_run`, `model_evaluation`, `model_significance`, `training_history`, `cross_validation`, `feature_selection` and `batch_risk`, tagged with `run_id` |
| `grafana/build_dashboards.py` | **Dashboards as code.** Flux query helpers, panel builders (stat, timeseries, table, geomap, state timeline, pie, bar gauge, bar chart, trend, histogram, heatmap, alert list), template variables (`station`, `source`, `run_id`) and the four boards. `--check` fails if the committed JSON is stale |
| `grafana/dashboards/crop_*.json` | Generated output for **Field Overview**, **Spectral Analysis**, **Model Performance** and **System Health**. Do not edit by hand |
| `grafana/datasources/influxdb.yaml` | Provisions the Flux InfluxDB datasource (uid `influxdb-crop`) using the read-only token |
| `grafana/alerting/rules.yaml` | Alert rules: high risk at a station (latest risk > 0.7), no data for 5 minutes, and bridge errors (invalid payloads or failed writes) |

#### InfluxDB schema

| Measurement | Tags | Key fields |
|---|---|---|
| `spectral_readings` | station_id, source, profile, lat, lon | 6 bands, 6 indices, latitude, longitude |
| `disease_predictions` | station_id, source, label, true_profile* | predicted_label, label_name, confidence, risk_score, prob_*, out_of_distribution, true_label*, correct* |
| `bridge_stats` | — | processed, invalid, predictions, write_errors |
| `pipeline_run`, `model_evaluation`, `model_significance`, `training_history`, `cross_validation`, `feature_selection`, `batch_risk` | run_id, model, … | metrics from `reports/` |

\* Only present for replayed sequences, which carry their CDM label.

### 3.15 `tests/`

| File | Covers |
|---|---|
| `fixtures/` | Committed snapshot of the real CDM + Sentinel-2 sequences (`cdm_s2_sequences.csv`, `cdm_s2_samples.csv`); every test runs on real readings |
| `conftest.py` | Shared fixtures: the fixture data preprocessed with the site-grouped split, a freshly trained tiny checkpoint, and the replay sites |
| `test_features.py` | Vegetation indices on real readings: finite, REIP in range, array/DataFrame parity |
| `test_preprocessing.py` | Split integrity (no site in two splits), train-only scaler, aggregated features, dropping short samples |
| `test_models.py` | Model shapes, attention on/off, width scaling, factory and config round-trip (incl. CNN-LSTM-Attention) |
| `test_tuning.py` | Grouped folds, the one-standard-error rule, classical and sequence-model searches, overfit-gap reporting |
| `test_inference.py` | Checkpoint round-trip, calibration, OOD flag, and **train/serve parity** |
| `test_api.py` | Auth, validation, rate and body limits, batch, health, info and metrics against a real model |
| `test_streaming.py` | Replay publisher (real sequences, real sites, split filter), payload validation, sequence assembly, TLS context, bridge records |
| `test_quantum.py` | QUBO construction, cardinality penalty, solver agreement with brute force |
| `test_geospatial.py` | GeoDataFrame building, metric grid, maps and GeoJSON |
| `test_dashboards.py` | Generated JSON is current, panel IDs and layout, variables, **every queried field is actually written**, bridge points, backfill and pipeline metrics |
| `test_performance.py` | After a pipeline run: every spectral model beats a majority-class predictor, CNN-LSTM-Attention is present, and no chosen configuration has a train-vs-CV gap above 0.35 |
| `test_integration.py` | End-to-end pipeline on the fixture data, with (shrunken) tuning and with `--no-tune` (`-m slow`) |
| `test_real_data.py` | CDM parsing and flagging, Sentinel-2 QC rules, CDSE response parsing, time-relative labelling, site-grouped splits, feature schema, MicaSense radiometry, panel detection, Vineyard experiment (all offline) |

### 3.15a `src/real_data/`

| File | What it does |
|---|---|
| `common.py` | Output folders, HTTP session with retries, seekable HTTP range file (pulls single members out of the 4.8 GB Vineyard ZIP) |
| `feature_schema.py` | Sensor-independent schema: maps Sentinel-2, MicaSense RedEdge-P and the project sensor onto canonical regions; indices computed only when every needed band exists; `common_indices()` lists what is comparable across sensors |
| `cdm.py` | CDM reports from the Bugwood API behind cdm.ipmpipe.org; normalises columns, merges an optional export with host/incidence/severity, flags usable events (GPS, region, repeat reports) |
| `sentinel2.py` | Point time series of B05/B06/B07/B08/B11/B12 on a 20 m grid; Earth Search COG backend (anonymous) and CDSE Statistical API backend; SCL masking and pixel-purity QC; per-point CSV cache |
| `temporal.py` | CDM + Sentinel-2 to labelled sequences in the pipeline's CSV format; seasonality-only features |
| `vineyard.py` | Selective ZIP download; DN to radiance to DLS reflectance; panel detection and per-site panel correction; ECC band alignment; vegetation-masked capture features; leave-one-session-out experiment with permutation test |
| `ethct.py` | EthCT2020 from the Mendeley dataset ZIP (`--zip-path`, or found automatically), sampling, purity statistics per crop, crop-type separability as a positive control |
| `__main__.py` | CLI: `cdm`, `s2-cdm`, `temporal`, `vineyard-fetch`, `vineyard`, `ethct`, `schema` |

### 3.16 Generated folders (not in git)

| Folder | Contents |
|---|---|
| `data/real/` | CDM tables, Sentinel-2 observations + per-point cache, CDM+S2 sequences (pipeline input), Vineyard TIFFs and features, EthCT2020 shapefile |
| `data/processed/` | `.npy` arrays, `scaler_params.json`, `sample_meta.csv` (with `site_id` and split), `predictions.csv` |
| `models/checkpoints/` | `lstm_best.pt`, `cnn_lstm_best.pt`, `cnn_lstm_attention_best.pt`, `rf_model.joblib`, `xgb_model.joblib`, `rf_qubo_model.joblib` |
| `reports/` | Pipeline reports (`model_comparison.csv`, `hyperparameter_tuning.csv`, CV, McNemar, QUBO, plots, maps), `event_qc_summary.csv`, `feature_schema.csv`, and `vineyard/`, `ethct/` |
| `certs/mqtt/` | CA and broker certificate and key (keep `ca.key` private) |

---

## 4. Running everything

```bash
# 1. Data and offline pipeline
pip install -r requirements.txt
python -m src.real_data cdm                 # CDM reports
python -m src.real_data s2-cdm              # Sentinel-2 around each event (~1 h, resumable)
python -m src.real_data temporal            # labelled 5-step sequences
python run_pipeline.py --skip-mqtt          # tune + train + evaluate everything

# 2. Live stack
cp .env.example .env                       # fill in every "change-me"
bash scripts/gen_mqtt_certs.sh
docker compose up -d                       # InfluxDB, Grafana, Mosquitto
python -m src.dashboard.influxdb_setup --create-tokens   # paste tokens into .env
docker compose --profile app up -d         # API, bridge, replay publisher

# 3. Fill the dashboards
python -m src.dashboard.seed_history --hours 12
python -m src.dashboard.publish_pipeline_metrics

# 4. Tests
pytest -m "not slow"
pytest -m slow

# 5. Side experiments (see README "Real data")
python -m src.real_data vineyard-fetch && python -m src.real_data vineyard
python -m src.real_data ethct --zip-path "Ethiopian Crop Type 2020 (EthCT2020) dataset.zip" --n-fields 120
```

Grafana runs at http://localhost:3000, in the **Crop Disease** folder. If
you are upgrading from an older version of the stack whose InfluxDB volume
has different credentials, run `docker compose down -v` once first.

---

## 5. Pipeline flags

| Flag | Default | Effect |
|---|---|---|
| `--seq-len` | inferred | Timesteps per sequence (also the live crop-cycle length); must match the dataset |
| `--epochs` / `--patience` | 80 / 12 | Training length and early stopping |
| `--no-tune` | off | Skip grouped-CV tuning and use the hyperparameters below |
| `--cv-folds` | 5 | Site-grouped folds for tuning |
| `--hidden-dim`, `--dropout`, `--weight-decay`, `--noise-std` | 32, 0.5, 1e-2, 0.1 | Deep-model hyperparameters when not tuning |
| `--batch-size`, `--lr` | 32, 1e-3 | Optimiser settings |
| `--models` | `lstm cnn_lstm cnn_lstm_attention` | Which sequence models to train |
| `--max-features` | 8 | K for QUBO selection |
| `--use-qaoa` | off | Also run QAOA on the top-8 sub-problem (~2.5 min) |
| `--grid-size-m` | 50000 | Zone grid cell size in metres |
| `--skip-quantum`, `--skip-geo`, `--skip-mqtt` | off | Skip Phases 4, 5 and 6 |
| `--seed` | 42 | Seeds everything |

---

## 6. Design decisions worth knowing

- **One inference path.** Training stores the scaler and temperature inside
  the checkpoint, and every consumer goes through `src/inference`. This
  removes train/serve skew by construction, and a test enforces it.
- **The split is by site, not by sample.** Consecutive sequences of one
  field overlap in time and look alike; a site never appears in two splits,
  and every CV used for tuning holds out whole sites.
- **Tuning that resists overfitting.** Hyperparameters are chosen by
  site-grouped CV with the one-standard-error rule (the simplest
  configuration whose CV score is within one standard error of the best),
  deep models early-stop on an inner site hold-out, and the train-vs-CV gap
  of every configuration is reported. The test sites are scored once.
- **Date and location are inputs, and their share is measured.** The
  labels are time-relative, so the season predicts a lot of them. Every
  real reading has a date and a location, so the models use them (CV
  macro-F1 0.38 → 0.61 for RF). The context-only and spectral-only RF
  ablations are in every comparison, so the spectral contribution stays
  visible. Spatial-zone CV shows the gain is not memorised geography.
- **Macro-F1 for model selection.** Early Stress is the hardest and most
  valuable class, and accuracy hides failures on it.
- **Calibrated, probability-weighted risk.** Risk uses calibrated
  probabilities, not just the predicted class, so an uncertain prediction
  yields intermediate risk.
- **One prediction per complete crop cycle.** The model was trained on full
  sequences, so the bridge never predicts on a partial or padded one.
- **Honest baselines for "quantum".** QUBO is compared with top-K, mRMR,
  random-K and all features, and QAOA is checked against the exact optimum.
- **Dashboards are code.** The JSON is generated and tests check it against
  what the bridge actually writes.

---

## 7. Future enhancements

### 7.1 Data and science (highest value)

1. **More and better real field data.** CDM + Sentinel-2, Vineyard UAV and
   EthCT2020 are now integrated (`src/real_data/`), but the CDM labels are
   weak (positives only, time-relative) and only 49 sites survive QC; on
   that data no spectral model beats a calendar-only baseline. Next steps:
   - plot-level ground truth with negative (healthy) controls;
   - host / incidence / severity from ipmPIPE (not in the public API);
   - PlanetScope or UAV resolution for plots smaller than 60 m;
   - a small instrumented trial, which would also test whether
     "pre-symptomatic" detection holds up.
2. **Spatial risk from outbreak spread.** Add neighbouring reports
   (distance- and time-weighted CDM pressure) and weather as features, so
   the risk maps reflect regional epidemics, not just one site's spectra.
3. **Sensor characterisation.** Model the real noise, drift and calibration
   of the chosen sensors (e.g. AS7265x plus a SWIR photodiode), and add a
   calibration-panel workflow.
4. **Irregular sequences.** Sentinel-2 acquisitions are 5 to 45 days apart
   after cloud masking. Feed the time gaps to the model (masking, Time2Vec,
   or transformers with timestamps) instead of treating steps as equal.
5. **Early-warning evaluation.** Measure *how many days before* symptoms the
   model can flag stress (lead-time curves), not only final-class accuracy.

### 7.2 Modelling

6. **Stronger architectures:**
   - Temporal Convolutional Networks or a small Transformer encoder.
   - InceptionTime-style ensembles.
   - Gradient-boosted models on richer temporal features (e.g. tsfresh).
7. **Uncertainty:** deep ensembles, MC-dropout or **conformal prediction**,
   to give prediction sets with guaranteed coverage.
8. **Better OOD detection:** Mahalanobis distance or an energy score in
   place of the fixed |z| > 8 rule.
9. **Explainability:** SHAP for the tree models; integrated gradients or
   attention maps for the CNN-LSTM, shown per prediction in the API and
   Grafana.
10. **Hyperparameter search** with Optuna, plus experiment tracking with
    MLflow (model registry, artifact lineage).

### 7.3 Quantum-inspired optimisation

11. Tune λ, τ and P automatically. Compare against exact MIQP solvers and
    other heuristics (tabu, genetic algorithms).
12. Try real annealing hardware (D-Wave Leap) or QAOA on cloud quantum
    backends for the small sub-problem, with an honest wall-clock and
    quality comparison.
13. Extend QUBO to **band selection for sensor design**, i.e. which
    wavelengths are worth buying hardware for.

### 7.4 Streaming and operations

14. **Mutual TLS.** Per-device client certificates (`require_certificate
    true`) plus per-station ACLs, so one compromised sensor can't publish as
    another.
15. **Device provisioning:** certificate rotation, OTA configuration, and a
    station registry (location, crop, install date) in place of seeded
    random stations.
16. **Model hot-reload and versioning:** the bridge and API watch a model
    registry, tag every prediction with `model_version`, and support
    shadow/canary models.
17. **Drift monitoring:** population-stability or KS tests on the incoming
    feature distributions vs the training scaler, with Grafana alerts.
18. **Alert routing:** a provisioned contact point (email, Slack, Telegram,
    SMS) and notification policies per severity, and an agronomist-facing
    message with location and a recommended action.
19. **Retention and downsampling:** InfluxDB tasks that roll raw readings
    up into hourly and daily aggregates for long-term trends.
20. **Scalability:** shared subscriptions or Kafka for many bridges,
    batched GPU inference, and a Kubernetes/Helm deployment.
21. **Observability:** scrape the API's `/metrics` and the bridge stats with
    Prometheus, and add tracing (OpenTelemetry).

### 7.5 Product and UX

22. **Farmer-facing app:** a mobile or web view of field risk with
    push notifications and treatment advice.
23. **Satellite and drone fusion:** combine station time series with
    Sentinel-2 or UAV imagery for field-wide maps between stations.
24. **Weather integration:** add humidity, temperature and leaf-wetness
    forecasts as features and as disease-pressure context (many fungal
    diseases are weather-driven).
25. **Feedback loop:** let agronomists confirm or correct predictions in the
    UI, store the labels, and retrain periodically (active learning).

### 7.6 Engineering hygiene

26. Add exploratory notebooks (EDA, model and QUBO experiments) written
    against the current `src/` API.
27. Add a type-checking step (mypy or pyright) and raise coverage targets in
    CI.
28. Add an integration test in CI that brings up the Docker stack (InfluxDB,
    Mosquitto, Grafana), runs the replay publisher for one cycle, and asserts that
    predictions arrive and every dashboard query returns data.
29. Build multi-architecture, signed container images (e.g. for Raspberry Pi
    edge gateways) with an SBOM.
