# Crop Disease Detection from Multispectral Time Series

A research prototype that classifies crop health (**Healthy / Early Stress /
Active Disease**) from time series of Red-Edge, NIR and SWIR reflectance.
It is trained on real data: cucurbit downy mildew (CDM) outbreak reports
paired with Sentinel-2 L2A time series. It compares classical models, an
LSTM, a CNN-LSTM and a **CNN-LSTM with temporal attention**, adds
quantum-inspired feature selection and geo-spatial risk maps, and serves
predictions through a live MQTT → InfluxDB → Grafana dashboard that replays
real held-out sequences.

> **Limitations — read first.**
> - **The labels are weak and follow the calendar.** CDM reports only say
>   *when and where* an outbreak was reported. Labels are defined relative
>   to that date (see [Data](#data)). Every model therefore gets the
>   acquisition **date and location** as inputs, and most of what it
>   predicts comes from them: spectra alone reach CV macro-F1 0.38, date
>   and location alone 0.62, both together 0.61 (see the
>   [ablation](#where-the-signal-comes-from)). **These models do not show
>   that Sentinel-2 at 20 m detects CDM on its own.**
> - **The dataset is small.** 521 sequences from 49 sites; the test split
>   is 83 sequences from 7 held-out sites, so test metrics have wide
>   intervals. The site-grouped cross-validation scores are the more
>   reliable comparison.
> - **"Quantum" means quantum-inspired.** Feature selection is a QUBO solved
>   by classical simulated annealing on a CPU. QAOA (`--use-qaoa`) runs on a
>   statevector simulator for the top-8-feature sub-problem only, and is
>   checked against the exact brute-force optimum. No quantum hardware is
>   used.

## Architecture

```
CDM reports (Bugwood API) ─► Sentinel-2 L2A around each report (STAC/COG, 20 m)
        │                            │  SCL cloud mask + pixel-purity QC
        └────────► time-relative weak labels ─► sequences (5 acquisitions)
                                             │
preprocessing (site-grouped split → scaler fit on train only)
   │
   │   per step: 6 bands + 6 indices + date & location context
   ├─► RF / XGBoost  (mean, std, slope, delta per feature)  ┐
   ├─► LSTM / CNN-LSTM / CNN-LSTM-Attention                 ├ tuned by site-grouped CV
   ├─► ablations: context-only RF, spectral-only RF         ┘
   ├─► QUBO feature selection vs. top-K / mRMR / random-K
   └─► best model (by CV) ─► risk maps (Folium, metric grid)

replay of held-out sequences ─MQTT─► bridge ─► src.inference ─► InfluxDB ─► Grafana
REST client ─────────────────────────► FastAPI ─► src.inference (same code path)
```

Training, the API and the MQTT bridge all go through one inference path
(`src/inference/`). It computes the vegetation indices and the context
channels, applies the scaler saved inside the checkpoint, and applies the
calibrated model.

## Quick start

Requires Python 3.12+ (tested on 3.14). Docker is needed only for the live
dashboard. Everything — pipeline, real-data tools, QAOA, tests and linting
— installs from one file:

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

### 1. Build the dataset

```bash
python -m src.real_data cdm                 # CDM reports -> data/real/cdm/
python -m src.real_data s2-cdm              # Sentinel-2 around each event (~1 h, resumable)
python -m src.real_data temporal            # labelled sequences -> data/real/cdm_s2/
```

Sentinel-2 is read from the public Earth Search COGs (no account). The
CDSE Statistical API is an alternative (`--backend cdse`, needs
`CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` in `.env`).

### 2. Train and evaluate

```bash
python run_pipeline.py --skip-mqtt          # 15-45 min on a laptop CPU, mostly tuning
python run_pipeline.py --no-tune --skip-mqtt   # fixed hyperparameters, a few minutes
```

Main flags: `--models lstm cnn_lstm cnn_lstm_attention`, `--cv-folds 5`,
`--epochs 80`, `--patience 12`, `--max-features 8`, `--use-qaoa`,
`--grid-size-m 50000`, `--skip-quantum`, `--skip-geo`, `--seed`, and
`--no-context` (spectral features only). With
`--no-tune`, `--hidden-dim`, `--dropout`, `--weight-decay` and
`--noise-std` set the deep models directly.

| Output | Contents |
|---|---|
| `data/processed/` | Split arrays, `scaler_params.json` (train-only fit), `sample_meta.csv`, `predictions.csv` |
| `models/checkpoints/*.pt` | Self-describing checkpoints: weights, architecture, scaler, temperature, CV score. Load with `weights_only=True`. |
| `reports/hyperparameter_tuning.csv` | Every configuration tried: CV macro-F1 (mean, SE), train macro-F1, train-vs-CV gap, chosen flag |
| `reports/model_comparison.csv` | Test metrics with 95% bootstrap CIs |
| `reports/model_significance.csv` | Pairwise McNemar tests |
| `reports/classical_cv.csv` | Site-grouped and spatial-zone CV |
| `reports/ablation_cv.csv` | Site-grouped CV of the context-only and spectral-only RFs |
| `reports/feature_selection_comparison.csv` | QUBO vs. top-K, mRMR and random-K (site-grouped CV) |
| `reports/disease_risk_map.html`, `zone_risk_map.html` | Interactive maps |

### 3. Side experiments

```bash
python -m src.real_data ethct --zip-path "path/to/Ethiopian Crop Type 2020 (EthCT2020) dataset.zip"
python -m src.real_data vineyard-fetch      # raw TIFFs, bands 1-5 (~2.7 GB of the 4.8 GB ZIP)
python -m src.real_data vineyard            # calibrate -> features -> leave-session-out CV
python -m src.real_data schema              # sensor-independent feature table
```

## Data

| Dataset | Source | Role |
|---|---|---|
| CDM reports | ipmPIPE / Bugwood public API | Outbreak dates + GPS that anchor the satellite time series |
| Sentinel-2 L2A | Copernicus archive via Earth Search COGs, or CDSE | B05/B06/B07/B08/B11/B12 = the six project bands, on a 20 m grid |
| EthCT2020 (Mendeley) | 2,793 crop-type field polygons, Ethiopia 2020 (local ZIP) | Spatial-quality check of the Sentinel-2 extraction |
| Vineyard (Zenodo 14936376) | MicaSense RedEdge-P UAV imagery, FD / Esca labels per session | Single-date UAV classification |

**Sentinel-2 QC.** Every acquisition is reduced to a 3×3 block of 20 m
pixels (B08 block-averaged from 10 m). Pixels must have scene class
vegetation or bare soil and reflectance in (0, 1]. An acquisition passes if
≥ 67% of the block and its centre pixel are valid, ≥ 50% of valid pixels are
vegetation, and the NIR coefficient of variation is ≤ 0.25. Failures are
kept, with `qc_reason` saying why (`reports/event_qc_summary.csv`).

Of 2,566 acquisitions for 86 events, 761 pass QC. Rejections: 902 cloud or
no data, 706 not vegetated, 104 too cloudy, 89 heterogeneous.

**Labels.** The public API returns GPS for only 87 usable events (2021
onward) and no host, incidence or severity (`python -m src.real_data cdm
--extra-csv export.csv` merges them from an export if you have one). All
reports are positives, so each site is its own control. A sequence ending:

| Class | Label | Sequence ends |
|---|---|---|
| 0 | Healthy | more than 21 days before the report |
| 1 | Early Stress | 21–0 days before the report |
| 2 | Active Disease | 0–45 days after the report |

That gives 521 sequences of 5 acquisitions from 49 sites. Splits and CV
always hold out whole sites, so no site is ever in two splits.

**Features.** Per acquisition: the 6 bands, 6 vegetation indices (NDRE,
NDVI, MSI, SWIR ratio, REIP, CRI) and 4 **context channels**
(`src/features/context.py`): day of year as sine and cosine, latitude and
longitude. Every real reading has a date and a location, so these are
legitimate inputs at prediction time. The classical models get the mean,
std, slope and delta of each spectral feature over the sequence, plus the
context at its last acquisition (52 features).

## Training without overfitting

Hundreds of overlapping sequences from a few dozen sites are easy to
memorise, so model selection is built around that (`src/training/tuning.py`):

- **Site-grouped 5-fold CV on train+val** scores every configuration. The
  test split is not used for any choice.
- **One-standard-error rule.** Among configurations within one standard
  error of the best CV score, the simplest is chosen (fewest parameters,
  shallowest trees, strongest regularization).
- **Regularization-leaning grids.** RF: depth 2–8, min leaf 5–30, features
  per split sqrt / 50% / all.
  XGBoost: depth 1–3, min child weight 5–20, L2 1–10. Deep models: hidden
  size 8–32, dropout 0.3–0.5, weight decay 0.01–0.1, Gaussian input noise
  0.1–0.3.
- **The train-vs-CV gap is reported** for every configuration.
- **Early stopping on an inner site hold-out** inside each training fold,
  never on the fold being scored.
- **Final deep models are refit on all of train+val** for the median best
  CV epoch, and temperature-calibrated on their out-of-fold logits rather
  than on a small validation split. The fold models are less confident
  than the refit model, so that temperature may soften the refit model but
  never sharpen it (T ≥ 1). Sharpening made the CNN models overconfident
  (mean confidence 0.90 at 72% accuracy); with T ≥ 1 they lean
  under-confident, the safer side for the API's `low_confidence` flag.
- The best deep model for the dashboard and API is picked by CV score, not
  test score.

### CNN-LSTM-Attention

`src/models/cnn_lstm_model.py`: two 1-D convolution blocks extract local
temporal patterns from the 16 per-step features (bands, indices and
context), an
LSTM models their order, and **additive temporal attention** weights the
LSTM outputs into one context vector for the classifier. `forward(x,
return_attention=True)` also returns the attention weights, so you can see
which acquisitions drove a prediction. The plain CNN-LSTM is the same network using only the
last LSTM step, which makes it the attention ablation.

## Results

Run: `python run_pipeline.py --skip-mqtt`, seed 42. Chosen configurations
from site-grouped 5-fold CV (`reports/hyperparameter_tuning.csv`), with the
previous spectral-only run for comparison:

| Model | Chosen configuration | CV macro-F1 | Before (spectral only) | Train − CV gap |
|---|---|---|---|---|
| Random Forest | depth 2, min leaf 30, 50% features | **0.609 ± 0.061** | 0.362 | 0.07 |
| XGBoost | depth 1, lr 0.03, 50 trees, min child 20, λ 10 | **0.606 ± 0.045** | 0.394 | 0.07 |
| LSTM | hidden 16, dropout 0.3, wd 0.1, noise 0.3 (5.6k params) | 0.542 ± 0.025 | 0.398 | 0.12 |
| CNN-LSTM | hidden 16, dropout 0.5, wd 0.1, noise 0.3 (3.2k params) | 0.533 ± 0.046 | 0.411 | 0.11 |
| **CNN-LSTM-Attention** | hidden 8, dropout 0.5, wd 0.1, noise 0.3 (1.4k params) | **0.553 ± 0.034** | 0.424 | 0.10 |

Chance macro-F1 is about 0.33. CV improved by 0.13–0.25 for every model,
and the train-vs-CV gap shrank from 0.11–0.34 to 0.07–0.12, so the models
fit *less* noise than before, not more. The tuned models are also smaller:
tuning picked the shallowest trees and a 1.4k-parameter attention network.
The CNN-LSTM-Attention has the best CV score of the deep models and is the
one served by the API and the dashboard.

Held-out test set (83 sequences, 7 sites; 95% bootstrap CIs):

| Model | Accuracy [95% CI] | Macro-F1 | Before | Early-Stress F1 |
|---|---|---|---|---|
| Random Forest | 0.735 [0.639, 0.831] | **0.657** | 0.416 | 0.357 |
| RF, 8 QUBO features | 0.735 [0.639, 0.831] | **0.659** | 0.425 | 0.385 |
| XGBoost | 0.687 [0.578, 0.783] | 0.610 | 0.373 | 0.250 |
| CNN-LSTM-Attention | 0.723 [0.626, 0.807] | 0.595 | 0.313 | 0.200 |
| LSTM | 0.663 [0.554, 0.771] | 0.575 | 0.397 | 0.296 |
| CNN-LSTM | 0.651 [0.554, 0.747] | 0.515 | 0.399 | 0.111 |
| *Context-only RF (date, location)* | 0.651 [0.542, 0.759] | 0.522 | – | 0.091 |
| *Spectral-only RF* | 0.506 [0.398, 0.614] | 0.402 | – | 0.121 |

### Where the signal comes from

| Inputs (RF, site-grouped CV) | CV macro-F1 |
|---|---|
| Spectral only | 0.384 ± 0.079 |
| Date + location only | 0.622 ± 0.100 |
| Both (tuned RF) | 0.609 ± 0.122 |
| Both, 8 QUBO-selected features | 0.618 ± 0.126 |

- **Most of the gain comes from date and location.** In CV, adding the
  spectra to the context does not help. On the test sites it does: the
  full RF beats the context-only RF (McNemar p = 0.016; 7 sequences only
  the full RF gets right, 0 the other way). That is a hint of spectral
  signal, from 7 sites, not proof.
- **The gain is not memorised geography.** Holding out whole ~0.1° spatial
  zones instead of sites gives the same scores (RF 0.612, XGBoost 0.610).
  The QUBO selection drops latitude and longitude entirely and keeps the
  two day-of-year channels plus six spectral features (SWIR-1610 and REIP
  means, red-edge 705/740 variability, REIP slope, red-edge 740 change).
  It matches all 52 features with 8.
- **Every model is significantly better than spectral-only** (McNemar
  p ≤ 0.036). The differences between the full models are not
  significant.
- **QUBO feature selection** (8 of 52 features, site-grouped CV):

  | Selection method | CV macro-F1 | Test macro-F1 |
  |---|---|---|
  | QUBO (simulated annealing) | 0.618 ± 0.126 | 0.659 |
  | Top-K by importance | 0.613 ± 0.120 | 0.657 |
  | All 52 features | 0.609 ± 0.122 | 0.657 |
  | mRMR | 0.608 ± 0.103 | 0.609 |
  | Random K (mean of 10) | 0.383 ± 0.057 | 0.436 |

  The informed selectors are tied with each other and with all features;
  all clearly beat random subsets, which usually miss the date channels.

### Side experiments

**EthCT2020 (120 fields, 6 crops, Sep–Nov 2020)**, loaded from the local
ZIP. 46% of in-season
acquisitions pass QC, and every field has at least one usable acquisition.
Among cloud-free acquisitions, 28% fail purity, nearly all as *not
vegetated*. The NIR-homogeneity rule rarely fires (median CV 0.06–0.08), so
the 0.25 threshold is lenient. Positive control: crop type predicted from
in-season spectra with spatially held-out folds reaches balanced accuracy
0.31–0.33 vs. 0.17 chance. That is above chance, so the extraction carries
real signal, but six similar cereal/legume crops are hard at 20 m. The
purity filter did not change it.

**Vineyard (UAV).** The processed PNGs in the archive are 8-bit and
contrast-stretched, so features are computed from the raw 16-bit TIFFs:
DN → radiance (MicaSense radiometric calibration, 2-D vignetting, row
gradient) → reflectance via the downwelling light sensor (DLS). Bands are
aligned by ECC homography, and pixels are masked to vegetation (NDVI > 0.4).

The calibration panel, found automatically, read 0.52–0.71 (site 1) and
0.86–1.26 (site 2) with DLS alone, against a nominal ~0.5. So each site is
corrected with its panel (`--panel-albedo`, default 0.5; use your panel's
certified values if known). Corrected vine canopy reads red ≈ 0.02,
red edge ≈ 0.17 and NIR ≈ 0.44.

Labels exist per session only, all on one date, so CV is
leave-one-session-out. Significance comes from permuting session labels.
Zenodo throttled the download after 111 of ~170 captures. All 28 labelled
sessions are covered, and rerunning `vineyard-fetch` resumes the rest.

| Healthy vs. diseased (FD or Esca), 28 sessions | Balanced acc. (capture) | Balanced acc. (session) | ROC-AUC (session) | Permutation p |
|---|---|---|---|---|
| Random Forest | 0.627 | 0.638 | 0.612 | 0.085 |
| Logistic Regression | 0.606 | 0.600 | 0.587 | 0.114 |
| XGBoost | 0.575 | 0.550 | 0.613 | – |

The direction is plausible: diseased sessions have lower red-edge
chlorophyll indices (NDRE p10 0.29 vs. 0.34, CI-red-edge p10 0.82 vs. 1.05).
But nothing is significant: the best permutation p is 0.085, and no single
feature survives FDR correction (q ≥ 0.34). With 8 healthy sessions, and
variety and site partly confounded with labels (`label_table_binary.csv`),
this dataset cannot establish detection.

## Septoria hyperspectral dataset (lab, in progress)

[Data INRAE doi:10.57745/WVP0FJ](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/WVP0FJ):
durum wheat leaves inoculated with *Zymoseptoria tritici* or with water
(controls), imaged by a VNIR (411–993 nm) and a SWIR (964–2494 nm) HySpex
camera on 18 dates from −1 to 20 days post inoculation (dpi). 611 GB in
total, one ZIP per date and camera. `src/real_data/septoria.py` reads only
the needed members out of the remote ZIPs and reduces each image to
per-leaf spectra and features (`data/real/septoria/spectra/`).

What was verified on the files (not assumed):
- **Matching.** Archive contents match the annotation tables (1,174 of
  1,175 images, 3,320 leaf observations present in both cameras). The leaf
  numbering refers to the same physical leaf across dates (each leaf has an
  ink mark visible in every preview).
- **Calibration.** Cubes are dark-subtracted raw counts, not reflectance; a
  white panel is scanned first in every image, so reflectance = leaf ÷
  panel per band and column. Leaf masks are 1-based and mirrored in SWIR
  but not in VNIR; the orientation is detected per image and the
  extraction fails loudly if a mask does not land on a leaf.
- **Symptom timing is not annotated.** `Data_description.csv` gives one
  infection-confidence value per leaf from a visual assessment, not
  per-date symptom status, so inoculation alone cannot define
  "pre-symptomatic". Onset is derived per leaf from the imagery: the first
  date (confirmed on the next) on which the leaf's chlorotic-pixel fraction
  exceeds every same-day control leaf of its genotype. Controls are scored
  the same way to measure the rule's false-positive rate. Control leaves
  also yellow and dry from ~15 dpi (detached-leaf senescence), so late
  dates are not disease-specific.

Cohort: pots 37–48 (6 inoculated, 6 control, both genotypes), left side,
dpi −1, 4, 7, 10, 12, 13, 14, 15.

```bash
python -m src.real_data septoria-fetch --sensors SWIR       # remote, resumable
# VNIR archives downloaded by hand go to data/real/septoria/archives/VNIR/<date>.zip
python -m src.real_data septoria-fetch --sensors VNIR --local-only
python -m src.real_data septoria          # nested pot-grouped CV -> reports/septoria/
```

The server throttles each connection (0.03–0.25 MB/s), so the remote
reader uses several parallel range requests and re-resolves expired
download links.

## Live dashboard

```bash
cp .env.example .env              # then replace every "change-me" value
bash scripts/gen_mqtt_certs.sh    # local CA + broker TLS certificate (certs/mqtt/)
docker compose up -d              # InfluxDB, Grafana, Mosquitto (bound to 127.0.0.1)
python -m src.dashboard.influxdb_setup --create-tokens   # paste tokens into .env
docker compose --profile app up -d                      # API, bridge, replay publisher
```

Or run the services directly:

```bash
python -m src.dashboard.mqtt_subscriber --model-path models/checkpoints/cnn_lstm_attention_best.pt
python -m src.data.mqtt_publisher --interval 1          # replays held-out test sequences
uvicorn src.api.main:app --port 8000
```

There is no simulator. The publisher **replays real Sentinel-2 sequences**
from `data/real/cdm_s2/cdm_s2_sequences.csv` — by default the test split, so
the live model never sees data it was trained on (`--split val|train|all`
to change). Each station is a real CDM site, and each payload carries the
sequence's weak label as ground truth, so the dashboard's live agreement is
a genuine held-out score. The bridge predicts once per completed
5-acquisition sequence.

Grafana runs at http://localhost:3000. Four linked dashboards are
provisioned in the **Crop Disease** folder:

| Dashboard | Shows |
|---|---|
| **Field Overview** | KPIs, a station map colored by latest risk, latest prediction per station, risk over time, class timeline, prediction mix, class probabilities, active alerts |
| **Spectral Analysis** | Band reflectance per station, six vegetation indices over time, an NDVI heatmap and distribution by label, latest indices per station |
| **Model Performance** | Live agreement with the labels, live confusion matrix, confidence and out-of-distribution counts; per pipeline run: model metrics with CIs, McNemar tests, training curves, site vs spatial CV, QUBO vs baselines, the risk map, macro-F1 across runs |
| **System Health** | Data freshness, ingest rate, station heartbeat, bridge throughput and errors |

```bash
# History through the same bridge code path (source=backfill)
python -m src.dashboard.seed_history --hours 12
python -m src.dashboard.seed_history --clear
# Offline results (run_pipeline.py publishes these automatically when InfluxDB is reachable)
python -m src.dashboard.publish_pipeline_metrics
# Dashboards are code: edit build_dashboards.py, not the JSON
python -m src.dashboard.grafana.build_dashboards
```

Three alert rules are provisioned: high risk at a station (latest risk
above 0.7), no readings for 5 minutes, and the bridge dropping payloads or
failing writes. Point them at your own contact point in Grafana → Alerting.

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

The API takes **raw band reflectances**: exactly `seq_len` rows (5 for the
current model) of 6 values in [0, 1], in band order `red_edge_705,
red_edge_740, red_edge_783, nir_842, swir_1610, swir_2190`. Models trained
with context (`GET /info` → `"requires_context": true`, the default) also
need the **acquisition date of every row** and the **field location**. The
server computes the indices, the context channels and the scaling.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"readings": [[0.08,0.13,0.15,0.18,0.20,0.11], ... 5 rows ...],
       "dates": ["2023-06-01","2023-06-06","2023-06-11","2023-06-16","2023-06-21"],
       "lat": 39.59, "lon": -74.69}'
```

`POST /predict/batch` takes `sequences` plus one `dates` list and one
`lat` / `lon` per sequence. MQTT payloads carry `date` (the acquisition
date) and `lat` / `lon`; a live sensor that omits `date` gets the date of
its own timestamp.

```json
{
  "predicted_label": 0,
  "label_name": "Healthy",
  "confidence": 0.93,
  "risk_score": 0.05,
  "probabilities": [0.93, 0.04, 0.03],
  "status": "ok",
  "out_of_distribution": false,
  "model_type": "cnn_lstm_attention",
  "inference_time_ms": 3.1
}
```

The values above are illustrative.

- **Probabilities and risk.** Probabilities are temperature-calibrated.
  `risk_score` is the probability-weighted risk Σ pᵢ·wᵢ, with w = (0, 0.5, 1).
- **`status`** is one of:
  - `ok`
  - `low_confidence`: confidence is below `LOW_CONFIDENCE_THRESHOLD`
  - `out_of_distribution`: any standardized feature has |z| > 8 (this
    includes a field far outside the training region)
- **Other endpoints:**
  - `POST /predict/batch`
  - `GET /info`: model metadata, including `seq_len` and `requires_context`
  - `GET /health`: 503 when no model is loaded
  - `GET /metrics`: Prometheus
- **Limits:** rate limiting, a body-size cap, a batch-size cap and CORS are
  all configured by environment variables (see `src/api/main.py`).

## Tests

```bash
pytest -m "not slow"          # fast tests: features, inference parity, API, streaming, QUBO, tuning
pytest -m slow                # end-to-end pipeline + tuning on a small real fixture
pytest --cov                  # coverage (CI requires >= 70%)
ruff check .
```

The tests use a small fixture of real CDM + Sentinel-2 sequences
(`tests/fixtures/`), so they need no network. The API tests run against a
freshly trained model, and the parity test asserts that raw readings sent
through the API path give the same probabilities as the offline path.

## Project structure

```
config/            non-secret INI settings, Mosquitto config + ACL
scripts/           gen_mqtt_certs.sh (MQTT TLS certificates)
src/
  constants.py     bands, labels, risk weights (single source of truth)
  config.py        INI + env loading, MQTT client factory
  real_data/       CDM reports, Sentinel-2 extraction + QC, weak labels,
                   EthCT2020 check, Vineyard UAV calibration, feature schema
  data/            preprocessing (site-grouped splits), MQTT replay publisher
  features/        vegetation indices (array/DataFrame, finite-checked)
  models/          LSTM, CNN-LSTM(-Attention), RF/XGBoost + CV, model factory
  training/        training loop, grouped-CV tuning, calibration, checkpoints
  inference/       feature pipeline + predictor (used by API, bridge, pipeline)
  evaluation/      metrics, bootstrap CIs, McNemar, plots
  quantum/         QUBO formulation, solvers, baseline comparison
  geospatial/      risk GeoDataFrames, metric zone grid, Folium maps
  api/             FastAPI service
  dashboard/       MQTT→InfluxDB bridge, InfluxDB setup, history backfill,
                   pipeline-metrics export, Grafana dashboards-as-code
run_pipeline.py    end-to-end training + evaluation
requirements.txt   all dependencies (runtime, real data, QAOA, tests, lint)
tests/             pytest suite + real-data fixtures
```
