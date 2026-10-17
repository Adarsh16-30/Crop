"""
MQTT Field-Station Replay Publisher
===================================
Replays real Sentinel-2 sequences (data/real/cdm_s2/cdm_s2_sequences.csv,
built by `python -m src.real_data temporal`) as a small network of field
stations publishing raw multispectral readings to MQTT.

Each station is a real CDM site (its own coordinates). A "crop cycle" is
one real labelled sequence of that site, replayed reading by reading, so
the live stream contains exactly the kind of data the model was trained
on. Every payload carries `station_id`, `cycle_id` and `timestep`, which
lets the subscriber assemble one complete sequence per cycle, and
`profile` (the sequence's time-relative label) as ground truth for the
dashboards' live agreement panels.

By default only sites from the held-out test split are replayed (when
data/processed/sample_meta.csv exists), so live agreement is not measured
on training data.

Payload (JSON):
    station_id, cycle_id, timestep, cycle_length, timestamp, date (acquisition),
    lat, lon, <6 band reflectances>, profile, source, sample_id
"""

import argparse
import json
import logging
import os
import signal
import time

import numpy as np
import pandas as pd

from src.config import load_mqtt_config, make_mqtt_client
from src.constants import BANDS, PROFILE_LABELS

logger = logging.getLogger(__name__)

SEQUENCES_CSV = 'data/real/cdm_s2/cdm_s2_sequences.csv'
SAMPLE_META_CSV = 'data/processed/sample_meta.csv'
LABEL_PROFILES = {v: k for k, v in PROFILE_LABELS.items()}


def load_replay_sequences(sequences_csv: str = SEQUENCES_CSV,
                          sample_meta_csv: str | None = SAMPLE_META_CSV,
                          split: str | None = 'test') -> dict[str, list[pd.DataFrame]]:
    """
    Site id -> its labelled sequences (each a seq_len-row DataFrame, ordered
    by timestep), optionally restricted to the sites of one split.
    """
    if not os.path.exists(sequences_csv):
        raise FileNotFoundError(
            f"{sequences_csv} not found. Build it with python -m src.real_data temporal")
    df = pd.read_csv(sequences_csv)
    if split and sample_meta_csv and os.path.exists(sample_meta_csv):
        meta = pd.read_csv(sample_meta_csv)
        if 'split' in meta:
            df = df[df['sample_id'].isin(meta.loc[meta['split'] == split, 'sample_id'])]
    if df.empty:
        raise ValueError(f"No sequences to replay from {sequences_csv} (split={split})")
    sites: dict[str, list[pd.DataFrame]] = {}
    for _, seq in df.groupby('sample_id', sort=True):
        sites.setdefault(seq['site_id'].iloc[0], []).append(seq.sort_values('timestamp'))
    return sites


class SpectralPublisher:
    """
    Parameters
    ----------
    mqtt_cfg : dict
        From src.config.load_mqtt_config.
    interval : float
        Seconds between publications (one station per publication, round-robin).
    n_stations : int
        Number of sites replayed (at most the number available).
    cycle_length : int or None
        Readings per crop cycle; must equal the model's seq_len. Inferred
        from the sequences if None.
    seed : int
        Seeds site choice and the order of replayed sequences.
    sites : dict or None
        Pre-loaded output of load_replay_sequences (default: load from disk).
    """

    def __init__(self, mqtt_cfg: dict, interval: float = 1.0, n_stations: int = 5,
                 cycle_length: int | None = None, seed: int = 42,
                 sites: dict | None = None, split: str | None = 'test'):
        self.cfg = mqtt_cfg
        self.topic = mqtt_cfg['topic']
        self.interval = interval
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.running = False
        self.message_count = 0

        sites = sites if sites is not None else load_replay_sequences(split=split)
        first = next(iter(sites.values()))[0]
        self.cycle_length = cycle_length or len(first)
        if len(first) != self.cycle_length:
            raise ValueError(f"Sequences have {len(first)} readings, cycle_length is "
                             f"{self.cycle_length}; rebuild them with the model's seq_len")

        chosen = self.rng.permutation(sorted(sites))[:n_stations]
        self.stations = []
        for i, site in enumerate(chosen):
            seqs = sites[site]
            station = {
                'station_id': f'S{i + 1:02d}', 'site_id': site,
                'lat': round(float(seqs[0]['lat'].iloc[0]), 6),
                'lon': round(float(seqs[0]['lon'].iloc[0]), 6),
                'sequences': seqs, 'cycle_id': -1,
            }
            self._start_cycle(station)
            self.stations.append(station)

        self.client = make_mqtt_client(
            mqtt_cfg, client_id=f"{mqtt_cfg['client_id']}-publisher", clean_session=True)
        self.client.on_connect = lambda c, u, f, rc, p=None: logger.info(
            "Connected to MQTT broker (%s)", rc)
        self.client.on_disconnect = lambda c, u, f, rc, p=None: (
            logger.warning("MQTT disconnected (%s); reconnecting", rc) if rc != 0 else None)

    def _start_cycle(self, station: dict) -> None:
        """Begin replaying the station's next real sequence (random order, repeating)."""
        station['cycle_id'] += 1
        seqs = station['sequences']
        if station['cycle_id'] % len(seqs) == 0:
            station['order'] = self.rng.permutation(len(seqs))
        seq = seqs[station['order'][station['cycle_id'] % len(seqs)]]
        station['trajectory'] = seq.reset_index(drop=True)
        station['profile'] = LABEL_PROFILES[int(seq['label'].iloc[0])]
        station['sample_id'] = seq['sample_id'].iloc[0]
        station['timestep'] = 0

    def _generate_reading(self, station: dict) -> dict:
        row = station['trajectory'].iloc[station['timestep']]
        reading = {band: float(row[band]) for band in BANDS}
        reading.update({
            'station_id': station['station_id'],
            'cycle_id': station['cycle_id'],
            'timestep': station['timestep'],
            'cycle_length': self.cycle_length,
            'timestamp': time.time(),
            'date': str(row['date'])[:10],
            'lat': station['lat'],
            'lon': station['lon'],
            'profile': station['profile'],
            'source': 'simulator',
            'sample_id': station['sample_id'],
        })
        return reading

    def start(self, max_messages: int | None = None) -> None:
        """Publish until stopped (SIGINT / SIGTERM) or `max_messages` is reached."""
        def stop(sig, frame):
            logger.info("Signal %s received, stopping publisher", sig)
            self.running = False
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

        self.client.connect_async(self.cfg['host'], self.cfg['port'],
                                  keepalive=self.cfg['keepalive'])
        self.client.loop_start()
        self.running = True
        deadline = time.monotonic() + 30
        while self.running and not self.client.is_connected() and time.monotonic() < deadline:
            time.sleep(0.1)  # don't start publishing before the broker accepts us
        logger.info("Replaying real sequences to '%s' every %.2fs (stations=%d, cycle=%d)",
                    self.topic, self.interval, len(self.stations), self.cycle_length)
        try:
            while self.running:
                if max_messages and self.message_count >= max_messages:
                    break
                station = self.stations[self.message_count % len(self.stations)]
                reading = self._generate_reading(station)
                info = self.client.publish(self.topic, json.dumps(reading), qos=self.cfg['qos'])

                if info.rc == 0:
                    self.message_count += 1
                    station['timestep'] += 1
                    if station['timestep'] >= self.cycle_length:
                        self._start_cycle(station)
                    if self.message_count % 10 == 0:
                        logger.info("Published %d messages (latest %s cycle %d t=%d %s)",
                                    self.message_count, reading['station_id'],
                                    reading['cycle_id'], reading['timestep'],
                                    reading['profile'])
                else:
                    logger.warning("Publish failed (rc=%s); will retry", info.rc)
                time.sleep(self.interval)
        finally:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Publisher stopped. Total messages: %d", self.message_count)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Replay real Sentinel-2 sequences over MQTT')
    parser.add_argument('--config', type=str, default='config/mqtt.ini')
    parser.add_argument('--sequences', type=str, default=SEQUENCES_CSV)
    parser.add_argument('--split', choices=['test', 'val', 'train', 'all'], default='test',
                        help='replay only the sites of this split (read from '
                             'data/processed/sample_meta.csv)')
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--stations', type=int, default=5)
    parser.add_argument('--max-messages', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    split = None if args.split == 'all' else args.split
    SpectralPublisher(
        load_mqtt_config(args.config), interval=args.interval, n_stations=args.stations,
        seed=args.seed, sites=load_replay_sequences(args.sequences, split=split),
    ).start(max_messages=args.max_messages)


if __name__ == '__main__':
    main()
