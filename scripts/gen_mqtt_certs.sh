#!/usr/bin/env bash
# Generate a private CA and a TLS server certificate for the Mosquitto broker.
#
#   scripts/gen_mqtt_certs.sh            # writes certs/mqtt/{ca,server}.{crt,key}
#   scripts/gen_mqtt_certs.sh --force    # overwrite existing certificates
#
# Extra hostnames / IPs the broker is reached by (comma separated), e.g. a
# field gateway's LAN address:
#   MQTT_CERT_HOSTS="broker.farm.lan,192.168.1.20" scripts/gen_mqtt_certs.sh
#
# The server certificate always covers: mosquitto (docker network name),
# localhost and 127.0.0.1. Clients trust certs/mqtt/ca.crt only.
# Private keys (*.key) never leave this machine: certs/ is git- and docker-ignored.
set -euo pipefail
export MSYS_NO_PATHCONV=1  # stop Git Bash on Windows rewriting "/CN=..." as a path

OUT_DIR="${MQTT_CERT_DIR:-certs/mqtt}"
CA_DAYS=3650
SERVER_DAYS=825

if [[ -f "$OUT_DIR/server.crt" && "${1:-}" != "--force" ]]; then
  echo "Certificates already exist in $OUT_DIR (use --force to replace them)." >&2
  exit 1
fi
command -v openssl >/dev/null || { echo "openssl is required" >&2; exit 1; }
mkdir -p "$OUT_DIR"

san="DNS:mosquitto,DNS:localhost,IP:127.0.0.1"
IFS=',' read -ra extra <<< "${MQTT_CERT_HOSTS:-}"
for host in "${extra[@]}"; do
  host="$(echo "$host" | xargs)"
  [[ -z "$host" ]] && continue
  if [[ "$host" =~ ^[0-9.]+$ || "$host" == *:* ]]; then san+=",IP:$host"; else san+=",DNS:$host"; fi
done

# 1. Certificate authority (EC P-256)
openssl ecparam -name prime256v1 -genkey -noout -out "$OUT_DIR/ca.key"
openssl req -x509 -new -key "$OUT_DIR/ca.key" -sha256 -days "$CA_DAYS" \
  -subj "/CN=Crop Disease Detection local MQTT CA" \
  -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
  -addext "keyUsage=critical,keyCertSign,cRLSign" \
  -out "$OUT_DIR/ca.crt"

# 2. Broker key + certificate signed by the CA
openssl ecparam -name prime256v1 -genkey -noout -out "$OUT_DIR/server.key"
openssl req -new -key "$OUT_DIR/server.key" -subj "/CN=mosquitto" -out "$OUT_DIR/server.csr"
cat > "$OUT_DIR/server.ext" <<EOF
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=$san
EOF
openssl x509 -req -in "$OUT_DIR/server.csr" -CA "$OUT_DIR/ca.crt" -CAkey "$OUT_DIR/ca.key" \
  -CAcreateserial -days "$SERVER_DAYS" -sha256 -extfile "$OUT_DIR/server.ext" \
  -out "$OUT_DIR/server.crt"
rm -f "$OUT_DIR/server.csr" "$OUT_DIR/server.ext" "$OUT_DIR/ca.srl"

chmod 600 "$OUT_DIR"/*.key
openssl verify -CAfile "$OUT_DIR/ca.crt" "$OUT_DIR/server.crt"
echo "Server certificate names: $san"
echo "Done. Keep $OUT_DIR/ca.key offline; clients only need $OUT_DIR/ca.crt."
