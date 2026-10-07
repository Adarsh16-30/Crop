"""
Runtime configuration for the streaming services.

Non-secret settings come from config/*.ini; every value can be overridden
by an environment variable. Secrets (MQTT password, InfluxDB tokens) are
read ONLY from the environment / a local .env file, never from the repo.
"""

import configparser
import os
import ssl

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def _ini(path: str, section: str) -> dict:
    parser = configparser.ConfigParser()
    if os.path.exists(path):
        parser.read(path)
    return dict(parser[section]) if parser.has_section(section) else {}


def load_mqtt_config(path: str = 'config/mqtt.ini') -> dict:
    """host, port, topic, client_id, keepalive, qos, credentials and TLS settings."""
    ini = _ini(path, 'mqtt')
    env = os.environ.get
    return {
        'host': env('MQTT_HOST', ini.get('host', 'localhost')),
        'port': int(env('MQTT_PORT', ini.get('port', 8883))),
        'topic': env('MQTT_TOPIC', ini.get('topic', 'crop/spectral')),
        'client_id': env('MQTT_CLIENT_ID', ini.get('client_id', 'crop-disease-system')),
        'keepalive': int(env('MQTT_KEEPALIVE', ini.get('keepalive', 60))),
        'qos': int(env('MQTT_QOS', ini.get('qos', 1))),
        'username': env('MQTT_USERNAME') or None,
        'password': env('MQTT_PASSWORD') or None,
        'tls_ca_file': env('MQTT_TLS_CA_FILE', ini.get('tls_ca_file', '')) or None,
        'tls_client_cert': env('MQTT_TLS_CLIENT_CERT', ini.get('tls_client_cert', '')) or None,
        'tls_client_key': env('MQTT_TLS_CLIENT_KEY', ini.get('tls_client_key', '')) or None,
    }


def load_influx_config(path: str = 'config/influxdb.ini', token_env: str = 'INFLUX_TOKEN') -> dict:
    """
    url, org, bucket, token.

    `token_env` selects which token to use, so the bridge can run with a
    write-only token (INFLUX_WRITE_TOKEN) instead of the admin token.
    """
    ini = _ini(path, 'influxdb')
    env = os.environ.get
    return {
        'url': env('INFLUX_URL', ini.get('url', 'http://localhost:8086')),
        'org': env('INFLUX_ORG', ini.get('org', 'crop-org')),
        'bucket': env('INFLUX_BUCKET', ini.get('bucket', 'crop_data')),
        'token': env(token_env) or env('INFLUX_TOKEN'),
    }


def make_tls_context(ca_file: str, client_cert: str | None = None,
                     client_key: str | None = None) -> ssl.SSLContext:
    """
    TLS context that trusts only `ca_file`, requires TLS 1.2+, and verifies
    the broker's certificate AND hostname. Optional client cert/key enable
    mutual TLS if the broker is configured with require_certificate true.
    """
    if not os.path.isfile(ca_file):
        raise FileNotFoundError(
            f"MQTT TLS CA certificate not found: {ca_file}. "
            "Generate certificates with scripts/gen_mqtt_certs.sh or set MQTT_TLS_CA_FILE."
        )
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_file)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    if client_cert:
        context.load_cert_chain(client_cert, client_key)
    return context


def make_mqtt_client(cfg: dict, client_id: str, clean_session: bool = True):
    """Build a paho client with auth, TLS (when a CA is configured) and reconnect backoff."""
    import paho.mqtt.client as mqtt

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        clean_session=clean_session,
        protocol=mqtt.MQTTv311,
    )
    if cfg.get('username'):
        client.username_pw_set(cfg['username'], cfg.get('password'))
    if cfg.get('tls_ca_file'):
        client.tls_set_context(make_tls_context(
            cfg['tls_ca_file'], cfg.get('tls_client_cert'), cfg.get('tls_client_key')))
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    return client
