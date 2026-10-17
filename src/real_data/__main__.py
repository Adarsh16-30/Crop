"""
Real-data command line.

    python -m src.real_data cdm                 # CDM reports -> data/real/cdm/
    python -m src.real_data s2-cdm              # Sentinel-2 around each CDM event
    python -m src.real_data temporal            # labelled sequences for run_pipeline.py
    python -m src.real_data vineyard-fetch      # selected Vineyard ZIP members
    python -m src.real_data vineyard            # calibrate, extract features, classify
    python -m src.real_data ethct               # EthCT2020 spatial-quality check
    python -m src.real_data schema              # sensor-independent feature schema table

Then tune, train and evaluate every model on the CDM + Sentinel-2 sequences:

    python run_pipeline.py
"""

import argparse
import datetime as dt
import json
import logging
import os
import sys

import pandas as pd

from src.real_data.common import data_dir, reports_dir

logger = logging.getLogger('real_data')


def add_purity_args(p: argparse.ArgumentParser) -> None:
    from src.real_data.sentinel2 import PurityConfig

    d = PurityConfig()
    p.add_argument('--backend', choices=['earthsearch', 'cdse'], default='earthsearch')
    p.add_argument('--window-px', type=int, default=d.window_px,
                   help='footprint size in 20 m pixels (odd)')
    p.add_argument('--min-valid-fraction', type=float, default=d.min_valid_fraction)
    p.add_argument('--min-vegetation-fraction', type=float, default=d.min_vegetation_fraction)
    p.add_argument('--max-nir-cv', type=float, default=d.max_nir_cv)
    p.add_argument('--max-scene-cloud', type=float, default=d.max_scene_cloud)
    p.add_argument('--no-center-check', action='store_true')
    p.add_argument('--workers', type=int, default=8, help='parallel reads per point')
    p.add_argument('--point-workers', type=int, default=4, help='points processed at once')


def purity_config(args):
    from src.real_data.sentinel2 import PurityConfig

    return PurityConfig(window_px=args.window_px, min_valid_fraction=args.min_valid_fraction,
                        require_center_valid=not args.no_center_check,
                        min_vegetation_fraction=args.min_vegetation_fraction,
                        max_nir_cv=args.max_nir_cv, max_scene_cloud=args.max_scene_cloud)


def cmd_cdm(args) -> None:
    from src.real_data.cdm import build_cdm_table

    build_cdm_table(args.years, extra_csv=args.extra_csv,
                    min_days_between=args.min_days_between)


def cmd_s2_cdm(args) -> None:
    from src.real_data.cdm import load_events
    from src.real_data.sentinel2 import extract_points, make_backend, save_config
    from src.real_data.temporal import event_points

    events = load_events()
    if args.limit:
        events = events.head(args.limit)
    points = event_points(events, args.days_before, args.days_after)
    cfg = purity_config(args)
    out_dir = data_dir('s2')
    obs = extract_points(points, make_backend(args.backend, workers=args.workers), cfg,
                         cache_dir=os.path.join(out_dir, 'cache'),
                         point_workers=args.point_workers)
    obs.to_csv(os.path.join(out_dir, 'cdm_observations.csv'), index=False)
    save_config(cfg, os.path.join(out_dir, 'cdm_extraction_config.json'),
                backend=args.backend, days_before=args.days_before, days_after=args.days_after)
    logger.info("Sentinel-2: %d acquisitions for %d events, %d pass QC | QC reasons %s",
                len(obs), obs['point_id'].nunique(), int(obs['purity_pass'].sum()),
                obs['qc_reason'].fillna('pass').value_counts().to_dict())


def cmd_temporal(args) -> None:
    from src.real_data.cdm import load_events
    from src.real_data.temporal import build_sequences, observation_summary, write_dataset

    events = load_events()
    obs = pd.read_csv(os.path.join(data_dir('s2'), 'cdm_observations.csv'))
    raw, samples = build_sequences(
        events, obs, seq_len=args.seq_len, early_window=args.early_window,
        post_window=args.post_window, max_gap_days=args.max_gap_days,
        require_purity=not args.no_purity)
    paths = write_dataset(raw, samples, data_dir('cdm_s2'))
    rep = reports_dir()
    observation_summary(events, obs).to_csv(os.path.join(rep, 'event_qc_summary.csv'),
                                            index=False)
    with open(os.path.join(data_dir('cdm_s2'), 'labelling.json'), 'w') as f:
        json.dump({'seq_len': args.seq_len, 'early_window_days': args.early_window,
                   'post_window_days': args.post_window, 'max_gap_days': args.max_gap_days,
                   'require_purity': not args.no_purity}, f, indent=2)
    logger.info("Wrote %s (%d samples)", paths['raw_csv'], len(samples))


def cmd_vineyard_fetch(args) -> None:
    from src.real_data.vineyard import fetch_vineyard

    fetch_vineyard(data_dir('vineyard'), sessions=args.sessions, bands=args.bands,
                   zip_path=args.zip_path, max_captures=args.max_captures,
                   workers=args.workers)


def cmd_vineyard(args) -> None:
    from src.real_data.vineyard import extract_features, run_experiment

    features = extract_features(data_dir('vineyard'), ndvi_threshold=args.ndvi_threshold,
                                workers=args.workers, panel_albedo=args.panel_albedo or None)
    if args.features_only:
        return
    run_experiment(features, reports_dir('vineyard'), task=args.task,
                   n_permutations=args.permutations, seed=args.seed)


def cmd_ethct(args) -> None:
    from src.real_data.ethct import run_ethct_check

    run_ethct_check(n_fields=args.n_fields, start=args.start, end=args.end,
                    zip_path=args.zip_path,
                    cfg=purity_config(args), backend_name=args.backend,
                    workers=args.workers, point_workers=args.point_workers, seed=args.seed)


def cmd_schema(args) -> None:
    from src.real_data.feature_schema import common_indices, schema_table

    path = os.path.join(reports_dir(), 'feature_schema.csv')
    schema_table().to_csv(path, index=False)
    logger.info("Feature schema -> %s | comparable UAV<->Sentinel-2 indices: %s", path,
                common_indices('micasense_rededge_p', 'sentinel2'))


def cmd_septoria_fetch(args) -> None:
    from src.real_data import septoria

    status = septoria.fetch_cohort(workers=args.workers, sensors=tuple(args.sensors),
                                   local_only=args.local_only)
    failed = status[status['status'].str.startswith('failed')]
    logger.info("Septoria: %d images ok / cached, %d failed%s", len(status) - len(failed),
                len(failed), " (rerun to retry)" if len(failed) else "")


def cmd_septoria(args) -> None:
    from src.real_data import septoria, septoria_experiment

    if args.previews:
        septoria.fetch_previews()
    septoria_experiment.run(seq_len=args.seq_len, n_outer=args.outer_folds,
                            n_permutations=args.permutations, seed=args.seed,
                            sensors=tuple(args.sensors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m src.real_data',
                                     description='Real-data acquisition and experiments')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('cdm', help='fetch and quality-flag CDM reports')
    p.add_argument('--years', type=int, nargs='+',
                   default=list(range(2017, dt.date.today().year + 1)))
    p.add_argument('--extra-csv', help='export with host/incidence/severity, joined on report_id')
    p.add_argument('--min-days-between', type=int, default=30)
    p.set_defaults(func=cmd_cdm)

    p = sub.add_parser('s2-cdm', help='Sentinel-2 L2A series around every usable CDM event')
    p.add_argument('--days-before', type=int, default=120)
    p.add_argument('--days-after', type=int, default=45)
    p.add_argument('--limit', type=int, help='only the first N events (smoke test)')
    add_purity_args(p)
    p.set_defaults(func=cmd_s2_cdm)

    p = sub.add_parser('temporal', help='labelled sequences from CDM + Sentinel-2')
    p.add_argument('--seq-len', type=int, default=5)
    p.add_argument('--early-window', type=int, default=21)
    p.add_argument('--post-window', type=int, default=45)
    p.add_argument('--max-gap-days', type=int, default=45)
    p.add_argument('--no-purity', action='store_true',
                   help='keep acquisitions that fail the pixel-purity checks')
    p.set_defaults(func=cmd_temporal)

    p = sub.add_parser('vineyard-fetch', help='download Vineyard ZIP members')
    p.add_argument('--sessions', nargs='+', help='session ids, e.g. 0001 0002 (default: all)')
    p.add_argument('--bands', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                   help='RedEdge-P band numbers (6 = panchromatic, not needed)')
    p.add_argument('--max-captures', type=int, help='per session (smoke test)')
    p.add_argument('--zip-path', help='use a locally downloaded ZIP instead of HTTP ranges')
    p.add_argument('--workers', type=int, default=6, help='parallel HTTP range downloads')
    p.set_defaults(func=cmd_vineyard_fetch)

    p = sub.add_parser('vineyard', help='calibrate, extract features, classify')
    p.add_argument('--task', choices=['binary', 'multiclass', 'fd', 'esca'], default='binary')
    p.add_argument('--ndvi-threshold', type=float, default=0.4)
    p.add_argument('--panel-albedo', type=float, default=0.5,
                   help='nominal panel reflectance for the panel correction (0 = DLS only)')
    p.add_argument('--permutations', type=int, default=200)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--features-only', action='store_true')
    p.set_defaults(func=cmd_vineyard)

    p = sub.add_parser('ethct', help='EthCT2020 pixel-purity and separability check')
    p.add_argument('--zip-path', help='the EthCT2020 dataset ZIP from Mendeley Data '
                   '(default: data/real/ethct/*.shp, else any EthCT2020*.zip in data/real/ethct, '
                   'the project root or ~/Downloads)')
    p.add_argument('--n-fields', type=int, default=300)
    p.add_argument('--start', default='2020-09-01')
    p.add_argument('--end', default='2020-11-30')
    p.add_argument('--seed', type=int, default=42)
    add_purity_args(p)
    p.set_defaults(func=cmd_ethct)

    p = sub.add_parser('septoria-fetch', help='extract the Septoria leaf cohort (VNIR + SWIR) '
                       'from the Data INRAE archives (resumable)')
    p.add_argument('--workers', type=int, default=6,
                   help='parallel range requests per file (the server throttles each one)')
    p.add_argument('--sensors', nargs='+', default=['SWIR', 'VNIR'], choices=['SWIR', 'VNIR'])
    p.add_argument('--local-only', action='store_true',
                   help='only archives already in data/real/septoria/archives/<camera>/ '
                        '(no slow remote reads)')
    p.set_defaults(func=cmd_septoria_fetch)

    p = sub.add_parser('septoria', help='Septoria nested pot-grouped CV: RF, XGBoost, LSTM, '
                       'CNN-LSTM(-Attention), QUBO, pre-symptomatic test')
    p.add_argument('--seq-len', type=int, default=4)
    p.add_argument('--outer-folds', type=int, default=6)
    p.add_argument('--permutations', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sensors', nargs='+', default=['VNIR', 'SWIR'], choices=['VNIR', 'SWIR'],
                   help='cameras to use (SWIR alone until the VNIR archives are extracted)')
    p.add_argument('--previews', action='store_true',
                   help='first fetch the RGB previews (symptom measure without VNIR cubes)')
    p.set_defaults(func=cmd_septoria)

    p = sub.add_parser('schema', help='write the sensor-independent feature schema')
    p.set_defaults(func=cmd_schema)
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s',
                        datefmt='%H:%M:%S')
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
