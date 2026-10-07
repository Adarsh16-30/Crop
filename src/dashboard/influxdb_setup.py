"""
InfluxDB Setup Script
======================
Initializes InfluxDB connection, verifies bucket exists,
and confirms connectivity for the crop disease detection system.
"""

import configparser
import os
import sys


def setup_influxdb(
    config_path: str = 'config/influxdb.ini'
) -> bool:
    """
    Verify InfluxDB connection and bucket setup.

    The Docker Compose auto-provisions the org, bucket, and token via
    environment variables. This script verifies everything is accessible.

    Parameters
    ----------
    config_path : str
        Path to InfluxDB configuration file.

    Returns
    -------
    bool
        True if setup is successful.
    """
    from influxdb_client import InfluxDBClient

    config = configparser.ConfigParser()
    config.read(config_path)

    url = config.get('influxdb', 'url', fallback='http://localhost:8086')
    token = config.get('influxdb', 'token', fallback='crop-disease-detection-token')
    org = config.get('influxdb', 'org', fallback='crop-org')
    bucket = config.get('influxdb', 'bucket', fallback='crop_data')

    print(f"InfluxDB Setup")
    print(f"  URL:    {url}")
    print(f"  Org:    {org}")
    print(f"  Bucket: {bucket}")
    print(f"  Token:  {token[:10]}...{token[-4:]}")

    try:
        client = InfluxDBClient(url=url, token=token, org=org)

        # Check health
        health = client.health()
        print(f"\n  Health: {health.status}")
        if health.status != 'pass':
            print(f"  ERROR: InfluxDB health check failed: {health.message}")
            return False

        # Check if bucket exists
        buckets_api = client.buckets_api()
        target_bucket = buckets_api.find_bucket_by_name(bucket)

        if target_bucket:
            print(f"  Bucket '{bucket}' exists (ID: {target_bucket.id})")
        else:
            print(f"  Bucket '{bucket}' not found. Creating...")
            from influxdb_client.domain.bucket_retention_rules import BucketRetentionRules
            retention = BucketRetentionRules(type='expire', every_seconds=0)  # infinite
            buckets_api.create_bucket(
                bucket_name=bucket,
                retention_rules=retention,
                org=org
            )
            print(f"  Bucket '{bucket}' created successfully")

        # Test write
        from influxdb_client import Point
        from influxdb_client.client.write_api import SYNCHRONOUS

        write_api = client.write_api(write_options=SYNCHRONOUS)
        test_point = Point("system_test").field("status", "ok")
        write_api.write(bucket=bucket, org=org, record=test_point)
        print(f"  Test write: SUCCESS")

        write_api.close()
        client.close()

        print(f"\n✅ InfluxDB setup verified successfully!")
        return True

    except Exception as e:
        print(f"\n❌ InfluxDB setup failed: {e}")
        print("\nMake sure InfluxDB is running:")
        print("  docker compose up -d")
        return False


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config/influxdb.ini'
    success = setup_influxdb(config_path)
    sys.exit(0 if success else 1)
