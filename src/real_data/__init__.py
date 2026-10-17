"""
Real-data integration
=====================
Acquisition, quality control and feature extraction for the field datasets
that the project is trained and checked on:

- ``vineyard``    MicaSense RedEdge-P UAV imagery with Flavescence doree /
                  Esca annotations (Zenodo 14936376): radiometric
                  calibration -> per-capture spectral features ->
                  leave-session-out classification.
- ``cdm``         Cucurbit downy mildew field reports (ipmPIPE / Bugwood API):
                  dates and GPS of confirmed outbreaks.
- ``sentinel2``   Sentinel-2 Level-2A surface reflectance around each report,
                  with scene-classification cloud masking and pixel-purity
                  checks on a 20 m grid.
- ``temporal``    CDM + Sentinel-2 -> labelled sequences in the same raw CSV
                  format the pipeline consumes, so the existing
                  preprocessing, models, evaluation and risk maps run on it.
- ``ethct``       EthCT2020 crop-field samples: a spatial-quality check of the
                  Sentinel-2 extraction (purity statistics + crop-type
                  separability as a positive control).
- ``feature_schema``  sensor-independent band / index names shared by the
                  UAV and satellite sensors.

Run ``python -m src.real_data --help`` for the command-line interface.
"""
