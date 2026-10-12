"""
Backfill simulated sensor history into InfluxDB
================================================
Live simulation produces one prediction per station every few minutes, so a
fresh dashboard is mostly empty. This tool writes N hours of *past* readings
for a set of simulated stations through exactly the same code path as the
live bridge (payload validation -> cycle assembly -> model -> InfluxDB
points), timestamped in the past.

Everything it writes is tagged source="backfill", so the dashboards' Source
filter can separate it from live data (source="simulator"), and
`--clear` removes it again.

    python -m src.dashboard.seed_history --model-path models/checkpoints/cnn_lstm_best.pt
    python -m src.dashboard.seed_history --hours 24 --interval-s 20 --stations 8
    python -m src.dashboard.seed_history --clear
"""

import argparse
import logging
import time
from datetime import UTC, datetime

from src.config import load_influx_config
from src.dashboard.mqtt_subscriber import SensorReading, SequenceAssembler, reading_to_records

logger = logging.getLogger(__name__)

BACKFILL_SOURCE = 'backfill'
MEASUREMENTS = ('spectral_readings', 'disease_predictions')


def simulate_history(predictor, hours: float = 12.0, interval_s: float = 30.0,
                     n_stations: int = 5, seed: int = 42, end_time: float | None = None,
                     source: str = BACKFILL_SOURCE):
    """
    Yield (records, prediction) for every simulated reading, oldest first.

    Each station takes one reading every `interval_s` seconds over the last
    `hours` hours; stations are offset slightly so they do not all report at
    the same instant.
    """
    from src.data.mqtt_publisher import SpectralPublisher

    no_network = {'topic': 'crop/spectral', 'client_id': 'backfill', 'qos': 1,
                  'keepalive': 60, 'username': None, 'password': None, 'tls_ca_file': None}
    sim = SpectralPublisher(no_network, n_stations=n_stations,
                            cycle_length=predictor.seq_len, seed=seed)
    assembler = SequenceAssembler(seq_len=predictor.seq_len)

    end_time = end_time or time.time()
    steps = int(hours * 3600 // interval_s)
    start = end_time - steps * interval_s
    offset = interval_s / max(n_stations, 1)

    for step in range(steps):
        for i, station in enumerate(sim.stations):
            payload = sim._generate_reading(station)
            payload['timestamp'] = start + step * interval_s + i * offset
            payload['source'] = source
            reading = SensorReading.model_validate(payload)
            yield reading_to_records(reading, assembler, predictor)

            station['timestep'] += 1
            if station['timestep'] >= sim.cycle_length:
                sim._start_cycle(station)


def seed(influx_cfg: dict, model_path: str, **kwargs) -> dict:
    """Write simulated history to InfluxDB. Returns counts."""
    from influxdb_client import InfluxDBClient, WriteOptions

    from src.inference.predictor import Predictor

    predictor = Predictor(model_path)
    counts = {'readings': 0, 'predictions': 0, 'correct': 0}
    errors = []
    with InfluxDBClient(url=influx_cfg['url'], token=influx_cfg['token'],
                        org=influx_cfg['org']) as client:
        with client.write_api(write_options=WriteOptions(batch_size=2000, flush_interval=1000),
                              error_callback=lambda c, d, e: errors.append(e)) as write_api:
            for records, prediction in simulate_history(predictor, **kwargs):
                write_api.write(bucket=influx_cfg['bucket'], org=influx_cfg['org'],
                                record=records)
                counts['readings'] += 1
                if prediction is not None:
                    counts['predictions'] += 1
    if errors:
        raise RuntimeError(f"{len(errors)} InfluxDB write batches failed: {errors[0]}")
    logger.info("Backfilled %d readings and %d predictions (source=%s) using %s",
                counts['readings'], counts['predictions'],
                kwargs.get('source', BACKFILL_SOURCE), predictor.model_type)
    return counts


def clear(influx_cfg: dict, source: str = BACKFILL_SOURCE) -> None:
    """Delete everything previously written with this source tag."""
    from influxdb_client import InfluxDBClient

    with InfluxDBClient(url=influx_cfg['url'], token=influx_cfg['token'],
                        org=influx_cfg['org']) as client:
        for measurement in MEASUREMENTS:
            client.delete_api().delete(
                start=datetime(1970, 1, 1, tzinfo=UTC), stop=datetime.now(UTC),
                predicate=f'_measurement="{measurement}" AND source="{source}"',
                bucket=influx_cfg['bucket'], org=influx_cfg['org'])
    logger.info("Deleted source=%s data from %s", source, ', '.join(MEASUREMENTS))


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Backfill simulated history into InfluxDB')
    parser.add_argument('--model-path', default='models/checkpoints/lstm_best.pt')
    parser.add_argument('--hours', type=float, default=12.0)
    parser.add_argument('--interval-s', type=float, default=30.0,
                        help='Seconds between readings of one station')
    parser.add_argument('--stations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42,
                        help='Same default as the live simulator (42), so backfilled '
                             'stations have the same ids and locations as the live ones')
    parser.add_argument('--influx-config', default='config/influxdb.ini')
    parser.add_argument('--clear', action='store_true',
                        help='Delete previously backfilled data instead of writing')
    args = parser.parse_args()

    # Writing needs the write-only token; deleting needs a token with
    # delete rights on the bucket (the admin token).
    cfg = load_influx_config(args.influx_config,
                             token_env='INFLUX_TOKEN' if args.clear else 'INFLUX_WRITE_TOKEN')
    if args.clear:
        clear(cfg)
    else:
        seed(cfg, args.model_path, hours=args.hours, interval_s=args.interval_s,
             n_stations=args.stations, seed=args.seed)


if __name__ == '__main__':
    main()
