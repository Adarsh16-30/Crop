"""
MQTT-to-InfluxDB Bridge Subscriber
====================================
Subscribes to raw spectral readings, validates them, computes vegetation
indices, assembles one complete sequence per (station, crop cycle), runs
calibrated inference through the shared src.inference pipeline, and writes
everything to InfluxDB for Grafana.

Measurements written:
    spectral_readings    raw bands + indices (tags: station_id, source, profile, lat, lon)
    disease_predictions  label, confidence, risk, class probabilities, OOD flag
    bridge_stats         processed / invalid / prediction counters
"""

import argparse
import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import load_influx_config, load_mqtt_config, make_mqtt_client
from src.constants import BANDS, PROFILE_LABELS
from src.features.spectral_indices import compute_index_values

logger = logging.getLogger(__name__)

Reflectance = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class SensorReading(BaseModel):
    """Validated MQTT payload. Every band is required -- no silent zero defaults."""
    model_config = ConfigDict(extra='ignore')

    station_id: str = Field(min_length=1, max_length=64)
    cycle_id: int = Field(ge=0)
    timestep: int = Field(ge=0)
    timestamp: float | None = None
    # Acquisition date (ISO). Replayed satellite data sends the real image
    # date; a live sensor may omit it, and then its own timestamp is used.
    date: str | None = Field(default=None, pattern=r'^\d{4}-\d{2}-\d{2}')
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    red_edge_705: float = Reflectance
    red_edge_740: float = Reflectance
    red_edge_783: float = Reflectance
    nir_842: float = Reflectance
    swir_1610: float = Reflectance
    swir_2190: float = Reflectance
    profile: str | None = None
    source: str = 'unknown'

    def bands(self) -> list[float]:
        return [getattr(self, b) for b in BANDS]

    def acquisition_date(self) -> str:
        if self.date:
            return self.date[:10]
        if self.timestamp:
            return datetime.fromtimestamp(self.timestamp, tz=UTC).date().isoformat()
        return datetime.now(UTC).date().isoformat()


@dataclass
class SequenceAssembler:
    """
    Collects readings per (station_id, cycle_id) and emits a complete
    sequence once timesteps 0..seq_len-1 have all arrived. A newer cycle
    from a station discards any unfinished older cycle from that station.
    """
    seq_len: int
    _buffers: dict = field(default_factory=dict)      # (station, cycle) -> {t: bands}
    _latest_cycle: dict = field(default_factory=dict)  # station -> cycle_id

    def add(self, reading: SensorReading) -> np.ndarray | None:
        """The completed (seq_len, 6) band sequence, or None."""
        done = self.collect(reading)
        return None if done is None else done[0]

    def collect(self, reading: SensorReading) -> tuple[np.ndarray, list[str]] | None:
        """The completed band sequence and its acquisition dates, or None."""
        station, cycle = reading.station_id, reading.cycle_id
        latest = self._latest_cycle.get(station)
        if latest is not None and cycle < latest:
            return None  # late reading from an abandoned cycle
        if latest is not None and cycle > latest:
            self._buffers.pop((station, latest), None)
        self._latest_cycle[station] = cycle

        if reading.timestep >= self.seq_len:
            return None
        buf = self._buffers.setdefault((station, cycle), {})
        buf[reading.timestep] = (reading.bands(), reading.acquisition_date())
        if len(buf) < self.seq_len:
            return None
        del self._buffers[(station, cycle)]
        steps = [buf[t] for t in range(self.seq_len)]
        return (np.array([b for b, _ in steps], dtype=np.float64), [d for _, d in steps])


def _stamp(point, reading: SensorReading):
    """
    Use the sensor's own timestamp when present, so readings that arrive late
    (e.g. queued by the broker while the bridge was down) land at the time
    they were measured, not the time they were ingested.
    """
    if reading.timestamp:
        point = point.time(datetime.fromtimestamp(reading.timestamp, tz=UTC))
    return point


def build_reading_point(reading: SensorReading):
    from influxdb_client import Point

    point = (Point('spectral_readings')
             .tag('station_id', reading.station_id)
             .tag('source', reading.source)
             .tag('lat', f'{reading.lat:.4f}')
             .tag('lon', f'{reading.lon:.4f}')
             .field('latitude', float(reading.lat))
             .field('longitude', float(reading.lon))
             .field('cycle_id', reading.cycle_id)
             .field('timestep', reading.timestep))
    if reading.profile:
        point = point.tag('profile', reading.profile)
    bands = {b: getattr(reading, b) for b in BANDS}
    for name, value in bands.items():
        point = point.field(name, float(value))
    for name, value in compute_index_values(bands).items():
        point = point.field(name, float(value))
    return _stamp(point, reading)


def build_prediction_point(reading: SensorReading, prediction: dict):
    from influxdb_client import Point

    point = (Point('disease_predictions')
             .tag('station_id', reading.station_id)
             .tag('source', reading.source)
             .tag('lat', f'{reading.lat:.4f}')
             .tag('lon', f'{reading.lon:.4f}')
             .field('cycle_id', reading.cycle_id)
             .field('predicted_label', prediction['predicted_label'])
             .field('label_name', prediction['label_name'])
             .field('confidence', prediction['confidence'])
             .field('risk_score', prediction['risk_score'])
             .field('out_of_distribution', prediction['out_of_distribution']))
    point = point.field('latitude', float(reading.lat)).field('longitude', float(reading.lon))
    for i, prob in enumerate(prediction['probabilities']):
        point = point.field(f'prob_class_{i}', prob)
    # Replayed real sequences carry their CDM label: record it so the dashboard can
    # show live agreement between the model and the ground truth.
    true_label = PROFILE_LABELS.get(reading.profile or '')
    if true_label is not None:
        point = (point.tag('true_profile', reading.profile)
                 .field('true_label', true_label)
                 .field('correct', int(true_label == prediction['predicted_label'])))
    return _stamp(point, reading)


def reading_to_records(reading: SensorReading, assembler: SequenceAssembler,
                       predictor=None) -> tuple[list, dict | None]:
    """
    The bridge's per-reading logic, shared with the history backfill:
    one spectral point, plus a prediction point if this reading completes a
    crop cycle and a model is loaded.
    """
    records = [build_reading_point(reading)]
    prediction = None
    done = assembler.collect(reading)
    if done is not None and predictor is not None:
        sequence, dates = done
        prediction = predictor.predict_raw(sequence, dates=dates, lat=reading.lat,
                                           lon=reading.lon)[0]
        records.append(build_prediction_point(reading, prediction))
    return records, prediction


class MQTTInfluxBridge:
    """MQTT subscriber that validates, predicts and writes to InfluxDB."""

    def __init__(self, mqtt_cfg: dict, influx_cfg: dict, model_path: str | None = None):
        from influxdb_client import InfluxDBClient, WriteOptions

        if not influx_cfg.get('token'):
            raise RuntimeError("No InfluxDB token: set INFLUX_WRITE_TOKEN or INFLUX_TOKEN "
                               "(see .env.example)")
        self.mqtt_cfg = mqtt_cfg
        self.bucket, self.org = influx_cfg['bucket'], influx_cfg['org']
        self.stats = {'processed': 0, 'invalid': 0, 'predictions': 0, 'write_errors': 0}
        self.running = False

        self.predictor = None
        if model_path:
            from src.inference.predictor import Predictor
            self.predictor = Predictor(model_path)
            logger.info("Loaded %s model (seq_len=%d)", self.predictor.model_type,
                        self.predictor.seq_len)
        seq_len = self.predictor.seq_len if self.predictor else 30
        self.assembler = SequenceAssembler(seq_len=seq_len)

        self.influx = InfluxDBClient(url=influx_cfg['url'], token=influx_cfg['token'],
                                     org=self.org)
        self.write_api = self.influx.write_api(
            write_options=WriteOptions(batch_size=200, flush_interval=1_000,
                                       retry_interval=2_000, max_retries=5),
            error_callback=self._on_write_error,
        )

        # Fixed client id + persistent session: QoS 1 messages published while
        # the bridge is briefly down are queued by the broker, not lost.
        self.mqtt = make_mqtt_client(mqtt_cfg, client_id=f"{mqtt_cfg['client_id']}-bridge",
                                     clean_session=False)
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_message
        self.mqtt.on_disconnect = self._on_disconnect

    def _on_write_error(self, conf, data, exception):
        self.stats['write_errors'] += 1
        logger.error("InfluxDB write failed: %s", exception)

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        logger.info("Connected to MQTT broker (%s)", reason_code)
        client.subscribe(self.mqtt_cfg['topic'], qos=self.mqtt_cfg['qos'])

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code != 0:
            logger.warning("MQTT disconnected (%s); paho will reconnect", reason_code)

    def _on_message(self, client, userdata, msg):
        try:
            reading = SensorReading.model_validate(json.loads(msg.payload))
        except (ValueError, ValidationError) as exc:
            self.stats['invalid'] += 1
            logger.warning("Dropping invalid payload on %s: %s", msg.topic,
                           str(exc).splitlines()[0])
            return
        try:
            self.process_reading(reading)
        except Exception:
            logger.exception("Failed to process reading from %s", reading.station_id)

    def process_reading(self, reading: SensorReading) -> dict | None:
        """Write the reading; if it completes a cycle, predict and write that too."""
        records, prediction = reading_to_records(reading, self.assembler, self.predictor)
        if prediction is not None:
            self.stats['predictions'] += 1
            logger.info("%s cycle %d -> %s (p=%.2f, risk=%.2f%s)", reading.station_id,
                        reading.cycle_id, prediction['label_name'], prediction['confidence'],
                        prediction['risk_score'],
                        ', OUT OF DISTRIBUTION' if prediction['out_of_distribution'] else '')
        self.write_api.write(bucket=self.bucket, org=self.org, record=records)
        self.stats['processed'] += 1
        return prediction

    def _write_stats(self):
        from influxdb_client import Point
        point = Point('bridge_stats')
        for key, value in self.stats.items():
            point = point.field(key, value)
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)

    def start(self):
        def stop(sig, frame):
            logger.info("Signal %s received, stopping bridge", sig)
            self.running = False
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

        self.mqtt.connect_async(self.mqtt_cfg['host'], self.mqtt_cfg['port'],
                                keepalive=self.mqtt_cfg['keepalive'])
        self.mqtt.loop_start()
        self.running = True
        logger.info("Bridge started | topic=%s bucket=%s inference=%s", self.mqtt_cfg['topic'],
                    self.bucket, 'on' if self.predictor else 'off')
        last_stats = time.monotonic()
        try:
            while self.running:
                time.sleep(1)
                if time.monotonic() - last_stats >= 30:
                    self._write_stats()
                    logger.info("Stats: %s", self.stats)
                    last_stats = time.monotonic()
        finally:
            self.mqtt.loop_stop()
            self.mqtt.disconnect()
            self.write_api.close()  # flushes pending batches
            self.influx.close()
            logger.info("Bridge stopped. %s", self.stats)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='MQTT to InfluxDB bridge with inference')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Checkpoint (.pt) from src.training.train; omit to disable inference')
    parser.add_argument('--mqtt-config', type=str, default='config/mqtt.ini')
    parser.add_argument('--influx-config', type=str, default='config/influxdb.ini')
    args = parser.parse_args()

    MQTTInfluxBridge(
        load_mqtt_config(args.mqtt_config),
        load_influx_config(args.influx_config, token_env='INFLUX_WRITE_TOKEN'),
        model_path=args.model_path,
    ).start()


if __name__ == '__main__':
    main()
