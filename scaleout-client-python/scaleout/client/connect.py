"""Client options and URL helper used when starting an edge client."""

import uuid
from typing import Dict, Optional

from scaleoututil.config import SCALEOUT_CUSTOM_URL_PREFIX


def get_url(api_url: str) -> str:
    """Construct the URL for the API."""
    return f"{api_url}/{SCALEOUT_CUSTOM_URL_PREFIX}".rstrip("/")


class ClientOptions:
    """Options for configuring the client."""

    def __init__(self, name: str, package: str, preferred_combiner: Optional[str] = None, client_id: Optional[str] = None) -> None:
        """Initialize ClientOptions with validation."""
        self._validate(name, package)
        self.name = name
        self.package = package
        self.preferred_combiner = preferred_combiner
        self.client_id = client_id if client_id else str(uuid.uuid4())

    def _validate(self, name: str, package: str) -> None:
        """Validate the name and package."""
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError("Name must be a string")
        if not isinstance(package, str) or len(package) == 0 or package not in ["local", "remote"]:
            raise ValueError("Package must be either 'local' or 'remote'")

    def to_json(self) -> Dict[str, Optional[str]]:
        """Convert ClientOptions to JSON."""
        return {
            "name": self.name,
            "client_id": self.client_id,
            "preferred_combiner": self.preferred_combiner,
            "package": self.package,
        }
