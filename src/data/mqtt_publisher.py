"""
MQTT Simulated Sensor Data Publisher
=====================================
Publishes synthetic multispectral sensor readings to an MQTT broker,
simulating a live data stream from field-deployed sensors.

Publishes to topic `crop/spectral` every N seconds.
"""

import argparse
import configparser
import json
import os
import random
import signal
import sys
import time

import numpy as np


# Default spectral band base values (healthy crop)
BASE_VALUES = {
    'red_edge_705': 0.35,
    'red_edge_740': 0.55,
    'red_edge_783': 0.65,
    'nir_842': 0.80,
    'swir_1610': 0.20,
    'swir_2190': 0.15,
}

# Disease profiles with degradation rates
PROFILES = {
    'healthy': {'degradation': 0.0, 'noise': 0.015},
    'early_stress': {'degradation': 0.005, 'noise': 0.02},
    'active_disease': {'degradation': 0.012, 'noise': 0.025},
}


class SpectralPublisher:
    """
    MQTT publisher that simulates spectral sensor data.

    Parameters
    ----------
    host : str
        MQTT broker hostname.
    port : int
        MQTT broker port.
    topic : str
        MQTT topic to publish to.
    interval : float
        Seconds between publications.
    profile : str
        Simulation profile: 'healthy', 'early_stress', 'active_disease', or 'mixed'.
    center_lat : float
        Central latitude for GPS simulation.
    center_lon : float
        Central longitude for GPS simulation.
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 1883,
        topic: str = 'crop/spectral',
        interval: float = 2.0,
        profile: str = 'mixed',
        center_lat: float = 12.97,
        center_lon: float = 77.59,
        n_stations: int = 5
    ):
        import paho.mqtt.client as mqtt

        self.topic = topic
        self.interval = interval
        self.profile = profile
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.running = False
        self.message_count = 0

        # Simulate a small, fixed set of sensor stations rather than a
        # fresh random GPS point per message. Each station keeps a stable
        # location and its own coherent timestep/profile trajectory, so
        # readings from the same station form a real time series --
        # required for the subscriber's per-location buffering to ever
        # accumulate enough consecutive readings to run inference, and for
        # Grafana panels keyed on (lat, lon) to stay a handful of series
        # instead of one new series per message.
        station_rng = random.Random(42)
        self.stations = []
        for i in range(n_stations):
            lat = round(center_lat + station_rng.uniform(-0.1, 0.1), 6)
            lon = round(center_lon + station_rng.uniform(-0.1, 0.1), 6)
            self.stations.append({
                'lat': lat,
                'lon': lon,
                'timestep': 0,
                'profile_name': self._pick_profile(),
            })

        # MQTT client setup -- a unique client_id per process avoids two
        # publisher instances fighting over the same broker session.
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f'crop-spectral-publisher-{os.getpid()}'
        )
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        try:
            self.client.connect(host, port, keepalive=60)
        except ConnectionRefusedError:
            print(f"ERROR: Cannot connect to MQTT broker at {host}:{port}")
            print("Make sure Mosquitto is running: docker compose up -d")
            sys.exit(1)

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"Connected to MQTT broker (rc={rc})")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        if rc != 0:
            print(f"Unexpected MQTT disconnect (rc={rc})")

    def _pick_profile(self) -> str:
        """Pick a disease profile for a station's next crop cycle."""
        if self.profile == 'mixed':
            return random.choices(
                ['healthy', 'early_stress', 'active_disease'],
                weights=[0.5, 0.3, 0.2]
            )[0]
        return self.profile

    def _generate_reading(self, station: dict) -> dict:
        """Generate a single spectral sensor reading for one station."""
        profile_name = station['profile_name']
        profile = PROFILES[profile_name]
        degradation = profile['degradation']
        noise = profile['noise']
        timestep = station['timestep']

        reading = {}
        for band, base_val in BASE_VALUES.items():
            # Apply degradation for affected bands
            if 'nir' in band or 'red_edge' in band:
                value = base_val - degradation * timestep
            elif 'swir' in band:
                value = base_val + degradation * timestep * 0.3
            else:
                value = base_val

            # Add noise and clamp
            value += np.random.normal(0, noise)
            reading[band] = round(max(0.0, min(1.0, value)), 6)

        # Fixed station GPS -- stable across this station's readings.
        reading['lat'] = station['lat']
        reading['lon'] = station['lon']
        reading['timestamp'] = time.time()
        reading['timestep'] = timestep
        reading['profile'] = profile_name

        return reading

    def start(self, max_messages: int | None = None):
        """
        Start publishing sensor data.

        Parameters
        ----------
        max_messages : int or None
            Stop after this many messages. None = run forever.
        """
        self.running = True
        self.client.loop_start()

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\nShutting down publisher...")
            self.running = False
        signal.signal(signal.SIGINT, signal_handler)

        print(f"Publishing to '{self.topic}' every {self.interval}s "
              f"(profile: {self.profile}, stations: {len(self.stations)})")
        print("Press Ctrl+C to stop.\n")

        try:
            while self.running:
                if max_messages and self.message_count >= max_messages:
                    break

                # Round-robin through the fixed stations so each one
                # accumulates its own consistent, growing time series.
                station = self.stations[self.message_count % len(self.stations)]

                reading = self._generate_reading(station)
                payload = json.dumps(reading)

                result = self.client.publish(self.topic, payload, qos=1)

                if result.rc == 0:
                    self.message_count += 1
                    station['timestep'] += 1

                    # Start a new crop cycle for this station periodically
                    if station['timestep'] > 30:
                        station['timestep'] = 0
                        station['profile_name'] = self._pick_profile()

                    if self.message_count % 10 == 0:
                        print(f"Published {self.message_count} messages "
                              f"(latest: {reading['profile']} @ "
                              f"{reading['lat']},{reading['lon']}, "
                              f"NIR={reading['nir_842']:.4f})")
                else:
                    print(f"Publish failed (rc={result.rc})")

                time.sleep(self.interval)

        finally:
            self.client.loop_stop()
            self.client.disconnect()
            print(f"\nPublisher stopped. Total messages: {self.message_count}")


def load_mqtt_config(config_path: str = 'config/mqtt.ini') -> dict:
    """Load MQTT configuration from INI file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return {
        'host': config.get('mqtt', 'host', fallback='localhost'),
        'port': config.getint('mqtt', 'port', fallback=1883),
        'topic': config.get('mqtt', 'topic', fallback='crop/spectral'),
    }


def main():
    """CLI entrypoint for MQTT publisher."""
    parser = argparse.ArgumentParser(description='Simulated spectral sensor publisher')
    parser.add_argument('--host', type=str, default=None,
                        help='MQTT broker host (default: from config)')
    parser.add_argument('--port', type=int, default=None,
                        help='MQTT broker port (default: from config)')
    parser.add_argument('--topic', type=str, default=None,
                        help='MQTT topic (default: from config)')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Seconds between messages (default: 2.0)')
    parser.add_argument('--profile', type=str, default='mixed',
                        choices=['healthy', 'early_stress', 'active_disease', 'mixed'],
                        help='Simulation profile (default: mixed)')
    parser.add_argument('--max-messages', type=int, default=None,
                        help='Stop after N messages (default: run forever)')
    parser.add_argument('--config', type=str, default='config/mqtt.ini',
                        help='Path to MQTT config file')
    parser.add_argument('--stations', type=int, default=5,
                        help='Number of simulated fixed-location sensor '
                             'stations to round-robin through (default: 5)')
    args = parser.parse_args()

    # Load config, override with CLI args
    mqtt_config = load_mqtt_config(args.config)
    host = args.host or mqtt_config['host']
    port = args.port or mqtt_config['port']
    topic = args.topic or mqtt_config['topic']

    publisher = SpectralPublisher(
        host=host,
        port=port,
        topic=topic,
        interval=args.interval,
        profile=args.profile,
        n_stations=args.stations
    )
    publisher.start(max_messages=args.max_messages)


if __name__ == '__main__':
    main()
