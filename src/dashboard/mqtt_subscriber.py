"""
MQTT-to-InfluxDB Bridge Subscriber
====================================
Subscribes to MQTT spectral sensor data, computes vegetation indices,
optionally runs model inference, and writes everything to InfluxDB
for Grafana visualization.

Measurements written to InfluxDB:
    - spectral_readings: raw band values + computed indices
    - disease_predictions: predicted label, confidence, risk score
"""

import argparse
import configparser
import json
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np


class MQTTInfluxBridge:
    """
    Bridge service that subscribes to MQTT and writes to InfluxDB.

    Parameters
    ----------
    mqtt_host : str
        MQTT broker hostname.
    mqtt_port : int
        MQTT broker port.
    mqtt_topic : str
        MQTT topic to subscribe to.
    influx_url : str
        InfluxDB URL.
    influx_token : str
        InfluxDB API token.
    influx_org : str
        InfluxDB organization.
    influx_bucket : str
        InfluxDB bucket name.
    model_path : str or None
        Path to trained PyTorch model for inference. None = skip inference.
    """

    def __init__(
        self,
        mqtt_host: str = 'localhost',
        mqtt_port: int = 1883,
        mqtt_topic: str = 'crop/spectral',
        influx_url: str = 'http://localhost:8086',
        influx_token: str = 'crop-disease-detection-token',
        influx_org: str = 'crop-org',
        influx_bucket: str = 'crop_data',
        model_path: str | None = None
    ):
        import paho.mqtt.client as mqtt
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS

        self.mqtt_topic = mqtt_topic
        self.message_count = 0
        self.running = False

        # InfluxDB setup
        self.influx_client = InfluxDBClient(
            url=influx_url,
            token=influx_token,
            org=influx_org
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        self.influx_bucket = influx_bucket
        self.influx_org = influx_org

        # MQTT setup
        self.mqtt_client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f'crop-influx-bridge-{os.getpid()}'
        )
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        self.mqtt_client.on_disconnect = self._on_disconnect

        try:
            self.mqtt_client.connect(mqtt_host, mqtt_port, keepalive=60)
        except ConnectionRefusedError:
            print(f"ERROR: Cannot connect to MQTT broker at {mqtt_host}:{mqtt_port}")
            sys.exit(1)

        # Optional: load ML model for real-time inference
        self.model = None
        self.device = None
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

        # Buffer for temporal sequences (per GPS location)
        self.sequence_buffer: dict[str, list[dict]] = {}
        self.buffer_max_len = 30

    def _load_model(self, model_path: str):
        """Load trained PyTorch model for inference."""
        try:
            import torch
            from src.models.lstm_model import SpectralLSTM

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = SpectralLSTM(input_dim=12, hidden_dim=128,
                                       num_layers=2, num_classes=3)
            checkpoint = torch.load(model_path, map_location=self.device,
                                     weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded from {model_path} (device: {self.device})")
        except Exception as e:
            print(f"WARNING: Could not load model: {e}")
            self.model = None

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"Connected to MQTT broker (rc={rc})")
        client.subscribe(self.mqtt_topic, qos=1)
        print(f"Subscribed to topic: {self.mqtt_topic}")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        if rc != 0:
            print(f"Unexpected MQTT disconnect (rc={rc})")

    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT message."""
        try:
            payload = json.loads(msg.payload.decode())
            self._process_reading(payload)
            self.message_count += 1

            if self.message_count % 10 == 0:
                print(f"Processed {self.message_count} messages")

        except Exception as e:
            print(f"Error processing message: {e}")

    def _compute_indices(self, reading: dict) -> dict:
        """Compute vegetation indices from raw band values."""
        eps = 1e-8
        re705 = reading.get('red_edge_705', 0)
        re740 = reading.get('red_edge_740', 0)
        re783 = reading.get('red_edge_783', 0)
        nir = reading.get('nir_842', 0)
        swir1 = reading.get('swir_1610', 0)
        swir2 = reading.get('swir_2190', 0)

        return {
            'NDRE': (re783 - re705) / (re783 + re705 + eps),
            'NDVI': (nir - re705) / (nir + re705 + eps),
            'MSI': swir1 / (nir + eps),
            'SWIR_ratio': swir2 / (swir1 + eps),
            'REIP_approx': 705 + 35 * ((re705 + re783) / 2 - re740) / (re783 - re740 + eps),
            'CRI': (1.0 / (re705 + eps)) - (1.0 / (re740 + eps)),
        }

    def _run_inference(self, reading: dict, indices: dict) -> dict | None:
        """Run model inference on buffered sequence."""
        if self.model is None:
            return None

        import torch

        # Build location key for buffering
        lat = round(reading.get('lat', 0), 2)
        lon = round(reading.get('lon', 0), 2)
        loc_key = f"{lat}_{lon}"

        # Build feature vector
        features = {
            'red_edge_705': reading.get('red_edge_705', 0),
            'red_edge_740': reading.get('red_edge_740', 0),
            'red_edge_783': reading.get('red_edge_783', 0),
            'nir_842': reading.get('nir_842', 0),
            'swir_1610': reading.get('swir_1610', 0),
            'swir_2190': reading.get('swir_2190', 0),
        }
        features.update(indices)

        # Add to buffer
        if loc_key not in self.sequence_buffer:
            self.sequence_buffer[loc_key] = []
        self.sequence_buffer[loc_key].append(features)

        # Keep buffer size limited
        if len(self.sequence_buffer[loc_key]) > self.buffer_max_len:
            self.sequence_buffer[loc_key] = self.sequence_buffer[loc_key][-self.buffer_max_len:]

        # Need full sequence for prediction
        if len(self.sequence_buffer[loc_key]) < self.buffer_max_len:
            return None

        # Build tensor
        feature_keys = list(features.keys())
        sequence = []
        for step in self.sequence_buffer[loc_key]:
            sequence.append([step[k] for k in feature_keys])

        x = torch.tensor([sequence], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        label = int(np.argmax(probs))
        confidence = float(probs[label])
        risk_map = {0: 0.0, 1: 0.5, 2: 1.0}
        risk_score = risk_map[label] * confidence

        label_names = {0: 'Healthy', 1: 'Early Stress', 2: 'Active Disease'}

        return {
            'predicted_label': label,
            'label_name': label_names[label],
            'confidence': confidence,
            'risk_score': risk_score,
            'probabilities': probs.tolist(),
        }

    def _process_reading(self, reading: dict):
        """Process a single sensor reading: compute indices, infer, write to InfluxDB."""
        from influxdb_client import Point

        indices = self._compute_indices(reading)

        # Write spectral readings to InfluxDB
        point = Point("spectral_readings")
        point = point.tag("source", "mqtt_simulator")

        # Add GPS as tags for efficient querying
        if 'lat' in reading:
            point = point.tag("lat", f"{reading['lat']:.4f}")
        if 'lon' in reading:
            point = point.tag("lon", f"{reading['lon']:.4f}")
        if 'profile' in reading:
            point = point.tag("profile", reading['profile'])

        # Raw band values as fields
        for band in ['red_edge_705', 'red_edge_740', 'red_edge_783',
                      'nir_842', 'swir_1610', 'swir_2190']:
            if band in reading:
                point = point.field(band, float(reading[band]))

        # Vegetation indices as fields
        for idx_name, idx_val in indices.items():
            point = point.field(idx_name, float(idx_val))

        self.write_api.write(bucket=self.influx_bucket, org=self.influx_org,
                             record=point)

        # Run inference and write predictions
        prediction = self._run_inference(reading, indices)
        if prediction:
            pred_point = Point("disease_predictions")
            pred_point = pred_point.tag("source", "mqtt_simulator")
            if 'lat' in reading:
                pred_point = pred_point.tag("lat", f"{reading['lat']:.4f}")
            if 'lon' in reading:
                pred_point = pred_point.tag("lon", f"{reading['lon']:.4f}")

            pred_point = pred_point.field("predicted_label", prediction['predicted_label'])
            pred_point = pred_point.field("label_name", prediction['label_name'])
            pred_point = pred_point.field("confidence", prediction['confidence'])
            pred_point = pred_point.field("risk_score", prediction['risk_score'])

            for i, prob in enumerate(prediction['probabilities']):
                pred_point = pred_point.field(f"prob_class_{i}", prob)

            self.write_api.write(bucket=self.influx_bucket, org=self.influx_org,
                                 record=pred_point)

    def start(self):
        """Start the MQTT-to-InfluxDB bridge."""
        self.running = True

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\nShutting down bridge...")
            self.running = False
        signal.signal(signal.SIGINT, signal_handler)

        print(f"MQTT -> InfluxDB bridge started")
        print(f"  MQTT topic: {self.mqtt_topic}")
        print(f"  InfluxDB bucket: {self.influx_bucket}")
        print(f"  Model inference: {'enabled' if self.model else 'disabled'}")
        print("Press Ctrl+C to stop.\n")

        self.mqtt_client.loop_start()

        try:
            while self.running:
                time.sleep(1)
        finally:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            self.write_api.close()
            self.influx_client.close()
            print(f"\nBridge stopped. Total messages processed: {self.message_count}")


def load_configs(
    mqtt_config_path: str = 'config/mqtt.ini',
    influx_config_path: str = 'config/influxdb.ini'
) -> dict:
    """Load MQTT and InfluxDB configs from INI files."""
    config = configparser.ConfigParser()

    mqtt_cfg = {'host': 'localhost', 'port': 1883, 'topic': 'crop/spectral'}
    if os.path.exists(mqtt_config_path):
        config.read(mqtt_config_path)
        mqtt_cfg['host'] = config.get('mqtt', 'host', fallback='localhost')
        mqtt_cfg['port'] = config.getint('mqtt', 'port', fallback=1883)
        mqtt_cfg['topic'] = config.get('mqtt', 'topic', fallback='crop/spectral')

    influx_cfg = {
        'url': 'http://localhost:8086',
        'token': 'crop-disease-detection-token',
        'org': 'crop-org',
        'bucket': 'crop_data'
    }
    if os.path.exists(influx_config_path):
        config.read(influx_config_path)
        influx_cfg['url'] = config.get('influxdb', 'url', fallback=influx_cfg['url'])
        influx_cfg['token'] = config.get('influxdb', 'token', fallback=influx_cfg['token'])
        influx_cfg['org'] = config.get('influxdb', 'org', fallback=influx_cfg['org'])
        influx_cfg['bucket'] = config.get('influxdb', 'bucket', fallback=influx_cfg['bucket'])

    return {'mqtt': mqtt_cfg, 'influx': influx_cfg}


def main():
    """CLI entrypoint for MQTT-InfluxDB bridge."""
    parser = argparse.ArgumentParser(description='MQTT to InfluxDB bridge with inference')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model checkpoint (.pt)')
    parser.add_argument('--mqtt-config', type=str, default='config/mqtt.ini',
                        help='Path to MQTT config')
    parser.add_argument('--influx-config', type=str, default='config/influxdb.ini',
                        help='Path to InfluxDB config')
    args = parser.parse_args()

    configs = load_configs(args.mqtt_config, args.influx_config)

    bridge = MQTTInfluxBridge(
        mqtt_host=configs['mqtt']['host'],
        mqtt_port=configs['mqtt']['port'],
        mqtt_topic=configs['mqtt']['topic'],
        influx_url=configs['influx']['url'],
        influx_token=configs['influx']['token'],
        influx_org=configs['influx']['org'],
        influx_bucket=configs['influx']['bucket'],
        model_path=args.model_path
    )
    bridge.start()


if __name__ == '__main__':
    main()
