import os

SCALEOUT_ARCHIVE_DIR = os.environ.get("SCALEOUT_ARCHIVE_DIR", ".scaleout/archive")
SCALEOUT_AUTH_SCHEME = os.environ.get("SCALEOUT_AUTH_SCHEME", "Bearer")
SCALEOUT_AUTH_REFRESH_TOKEN_URI = os.environ.get("SCALEOUT_AUTH_REFRESH_TOKEN_URI", False)
SCALEOUT_AUTH_REFRESH_TOKEN = os.environ.get("SCALEOUT_AUTH_REFRESH_TOKEN", False)
SCALEOUT_CONNECT_API_SECURE = os.environ.get("SCALEOUT_CONNECT_API_SECURE", "true").lower() == "true"
# Use a TLS (secure) gRPC channel to the combiner. Secure by default; set to false
# for plaintext deployments (e.g. local dev, or a combiner not fronted by TLS).
SCALEOUT_GRPC_SECURE = os.environ.get("SCALEOUT_GRPC_SECURE", "true").lower() == "true"
SCALEOUT_CUSTOM_URL_PREFIX = os.environ.get("SCALEOUT_CUSTOM_URL_PREFIX", "")
SCALEOUT_PACKAGE_EXTRACT_DIR = os.environ.get("SCALEOUT_PACKAGE_EXTRACT_DIR", "package")
SCALEOUT_GRACEFUL_CLIENT_CONNECTION = os.environ.get("SCALEOUT_GRACEFUL_GRPC_HANDLING", "true").lower() == "true"
SCALEOUT_CHECK_COMPATIBILITY = os.environ.get("SCALEOUT_CHECK_COMPATIBILITY", "true").lower() == "true"
SCALEOUT_CLIENT_STATUS_REPORTING = os.environ.get("SCALEOUT_CLIENT_STATUS_REPORTING", "true").lower() == "true"
SCALEOUT_CLIENT_SEND_TELEMETRY = os.environ.get("SCALEOUT_CLIENT_SEND_TELEMETRY", "true").lower() == "true"
SCALEOUT_CLIENT_TASK_POLLING_INTERVAL = int(os.environ.get("SCALEOUT_CLIENT_TASK_POLLING_INTERVAL", "5"))
