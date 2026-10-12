"""
MQTT Simulated Sensor Data Publisher
=====================================
Simulates a small network of fixed field sensor stations publishing raw
multispectral readings to MQTT.

Each station runs "crop cycles". A cycle is one plant trajectory of
exactly `cycle_length` readings produced by the SAME generator used for the
training data (src.data.data_generator), so the live stream matches what
the model was trained on. Every payload carries `station_id`, `cycle_id`
and `timestep`, which lets the subscriber assemble one complete sequence
per cycle instead of a rolling window that straddles two plants.

Payload (JSON):
    station_id, cycle_id, timestep, cycle_length, timestamp,
    lat, lon, <6 band reflectances>, profile, source
"""

import argparse
import json
import logging
import signal
import time

import numpy as np

from src.config import load_mqtt_config, make_mqtt_client
from src.constants import BANDS, PROFILE_LABELS
from src.data.data_generator import field_zone_baseline, generate_spectral_sequence

logger = logging.getLogger(__name__)

PROFILE_WEIGHTS = [0.5, 0.3, 0.2]


class SpectralPublisher:
    """
    Parameters
    ----------
    mqtt_cfg : dict
        From src.config.load_mqtt_config.
    interval : float
        Seconds between publications (one station per publication, round-robin).
    profile : str
        'healthy', 'early_stress', 'active_disease' or 'mixed'.
    n_stations : int
        Number of fixed-location stations.
    cycle_length : int
        Readings per crop cycle. Must equal the model's seq_len for the
        subscriber to produce predictions.
    seed : int
        Seeds station placement, profile choice and readings.
    """

    def __init__(self, mqtt_cfg: dict, interval: float = 1.0, profile: str = 'mixed',
                 center_lat: float = 12.97, center_lon: float = 77.59,
                 n_stations: int = 5, cycle_length: int = 30, seed: int = 42):
        if profile != 'mixed' and profile not in PROFILE_LABELS:
            raise ValueError(f"Unknown profile {profile!r}")
        self.cfg = mqtt_cfg
        self.topic = mqtt_cfg['topic']
        self.interval = interval
        self.profile = profile
        self.cycle_length = cycle_length
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.running = False
        self.message_count = 0

        self.stations = []
        for i in range(n_stations):
            station = {
                'station_id': f'S{i + 1:02d}',
                'lat': round(center_lat + self.rng.uniform(-0.1, 0.1), 6),
                'lon': round(center_lon + self.rng.uniform(-0.1, 0.1), 6),
                'cycle_id': -1,
            }
            self._start_cycle(station)
            self.stations.append(station)

        self.client = make_mqtt_client(
            mqtt_cfg, client_id=f"{mqtt_cfg['client_id']}-publisher", clean_session=True)
        self.client.on_connect = lambda c, u, f, rc, p=None: logger.info(
            "Connected to MQTT broker (%s)", rc)
        self.client.on_disconnect = lambda c, u, f, rc, p=None: (
            logger.warning("MQTT disconnected (%s); reconnecting", rc) if rc != 0 else None)

    def _pick_profile(self) -> str:
        if self.profile == 'mixed':
            return str(self.rng.choice(list(PROFILE_LABELS), p=PROFILE_WEIGHTS))
        return self.profile

    def _start_cycle(self, station: dict) -> None:
        """Begin a new plant trajectory for a station."""
        profile = self._pick_profile()
        station['cycle_id'] += 1
        station['profile'] = profile
        station['timestep'] = 0
        station['trajectory'] = generate_spectral_sequence(
            n_timesteps=self.cycle_length,
            label=PROFILE_LABELS[profile],
            seed=int(self.rng.integers(0, 2**31)),
            field_baseline=field_zone_baseline(station['lat'], station['lon'], seed=self.seed),
        )

    def _generate_reading(self, station: dict) -> dict:
        row = station['trajectory'].iloc[station['timestep']]
        reading = {band: float(row[band]) for band in BANDS}
        reading.update({
            'station_id': station['station_id'],
            'cycle_id': station['cycle_id'],
            'timestep': station['timestep'],
            'cycle_length': self.cycle_length,
            'timestamp': time.time(),
            'lat': station['lat'],
            'lon': station['lon'],
            'profile': station['profile'],
            'source': 'simulator',
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
        logger.info("Publishing to '%s' every %.2fs (profile=%s, stations=%d, cycle=%d)",
                    self.topic, self.interval, self.profile, len(self.stations),
                    self.cycle_length)
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
    parser = argparse.ArgumentParser(description='Simulated spectral sensor publisher')
    parser.add_argument('--config', type=str, default='config/mqtt.ini')
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--profile', type=str, default='mixed',
                        choices=[*PROFILE_LABELS, 'mixed'])
    parser.add_argument('--stations', type=int, default=5)
    parser.add_argument('--cycle-length', type=int, default=30,
                        help="Readings per crop cycle; must equal the model's seq_len")
    parser.add_argument('--max-messages', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    SpectralPublisher(
        load_mqtt_config(args.config), interval=args.interval, profile=args.profile,
        n_stations=args.stations, cycle_length=args.cycle_length, seed=args.seed,
    ).start(max_messages=args.max_messages)


if __name__ == '__main__':
    main()
