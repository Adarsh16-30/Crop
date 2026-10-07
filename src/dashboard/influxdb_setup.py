"""
InfluxDB Setup Script
======================
Verifies InfluxDB connectivity and the bucket, and can mint least-privilege
tokens so the services never run with the admin token:

    python -m src.dashboard.influxdb_setup                  # verify only
    python -m src.dashboard.influxdb_setup --create-tokens  # + scoped tokens

--create-tokens prints a write-only token (bridge) and a read-only token
(Grafana); copy them into .env as INFLUX_WRITE_TOKEN / INFLUX_READ_TOKEN.
Requires INFLUX_TOKEN (the admin token) in the environment.
"""

import argparse
import logging
import sys

from src.config import load_influx_config

logger = logging.getLogger(__name__)


def setup_influxdb(config_path: str = 'config/influxdb.ini', create_tokens: bool = False) -> bool:
    """Check health and bucket; optionally create scoped tokens. Returns success."""
    from influxdb_client import InfluxDBClient
    from influxdb_client.domain.bucket_retention_rules import BucketRetentionRules

    cfg = load_influx_config(config_path)
    if not cfg['token']:
        logger.error("INFLUX_TOKEN is not set (see .env.example)")
        return False
    logger.info("InfluxDB %s | org=%s bucket=%s", cfg['url'], cfg['org'], cfg['bucket'])

    try:
        with InfluxDBClient(url=cfg['url'], token=cfg['token'], org=cfg['org']) as client:
            health = client.health()
            if health.status != 'pass':
                logger.error("InfluxDB health check failed: %s", health.message)
                return False

            buckets_api = client.buckets_api()
            bucket = buckets_api.find_bucket_by_name(cfg['bucket'])
            if bucket is None:
                logger.info("Bucket '%s' not found; creating (30-day retention)", cfg['bucket'])
                bucket = buckets_api.create_bucket(
                    bucket_name=cfg['bucket'], org=cfg['org'],
                    retention_rules=BucketRetentionRules(type='expire',
                                                         every_seconds=30 * 24 * 3600))
            logger.info("Bucket '%s' ready (id %s)", cfg['bucket'], bucket.id)

            if create_tokens:
                _create_scoped_tokens(client, bucket)
        return True
    except Exception as exc:
        logger.error("InfluxDB setup failed: %s (is it running? docker compose up -d)", exc)
        return False


def _create_scoped_tokens(client, bucket) -> None:
    from influxdb_client import Permission, PermissionResource

    org_id = bucket.org_id
    resource = PermissionResource(type='buckets', id=bucket.id, org_id=org_id)
    auth_api = client.authorizations_api()
    write = auth_api.create_authorization(
        org_id=org_id, permissions=[Permission(action='write', resource=resource)])
    read = auth_api.create_authorization(
        org_id=org_id, permissions=[Permission(action='read', resource=resource)])
    print("\nAdd these to your .env (shown once; store them securely):")
    print(f"INFLUX_WRITE_TOKEN={write.token}")
    print(f"INFLUX_READ_TOKEN={read.token}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    parser = argparse.ArgumentParser(description='Verify / prepare InfluxDB')
    parser.add_argument('--config', default='config/influxdb.ini')
    parser.add_argument('--create-tokens', action='store_true')
    args = parser.parse_args()
    sys.exit(0 if setup_influxdb(args.config, create_tokens=args.create_tokens) else 1)


if __name__ == '__main__':
    main()
