"""Cryptographic signing utilities for ScaleoutModel.

Models are signed using Ed25519 asymmetric keys (via the ``cryptography`` package).
The signature covers a SHA-256 hash of the ordered model content entries inside the
ZIP (``inference_model.bin``, ``metadata.json``, ``training_model.bin``), in
alphabetical order.  Each entry contributes its name followed by a null-byte
separator and then its raw bytes.

Signatures are stored in the database (not inside the ZIP file), which allows
signatures to be added to pre-existing immutable models.

Typical usage::

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    sig_dict = model.sign(key, signer_id="node-a")
    # sig_dict = {"model_id": "...", "algorithm": "ed25519", "signer_id": "node-a", "signature": "<b64>"}
    # POST sig_dict to /api/v1/model-signatures/ to persist it.

    # Verification (signature_data comes from the API):
    is_valid = model.verify_signature(key.public_key(), signature_data)
"""

import base64
import hashlib
import zipfile
from cryptography.exceptions import InvalidSignature
from scaleoututil.logging import ScaleoutLogger


def compute_model_content_hash(zip_path: str) -> bytes:
    """Return the SHA-256 hash of the model's signable content.

    Hashes all entries in the ZIP in alphabetical order.  Each entry contributes
    its name followed by a null-byte separator and then its raw bytes.
    """
    hasher = hashlib.sha256()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(zf.namelist())
        for entry_name in names:
            hasher.update(entry_name.encode("utf-8"))
            hasher.update(b"\x00")
            with zf.open(entry_name) as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    hasher.update(chunk)
    return hasher.digest()


def compute_model_signature(zip_path: str, private_key, signer_id=None) -> dict:
    """Sign model content and return a dict ready to POST to the API.

    Args:
        zip_path: Path to the model ZIP file.
        private_key: An ``Ed25519PrivateKey`` from the ``cryptography`` package.
        signer_id: Optional free-form string identifying the signer.

    Returns:
        A dict with keys ``algorithm``, ``signature`` (base64-encoded), and
        optionally ``signer_id``.  The caller should attach ``model_id`` before
        posting to the API.
    """
    content_hash = compute_model_content_hash(zip_path)
    sig_bytes = private_key.sign(content_hash)
    entry = {
        "algorithm": "ed25519",
        "signature": base64.b64encode(sig_bytes).decode("ascii"),
    }
    if signer_id is not None:
        entry["signer_id"] = signer_id
    return entry


def verify_model_signature(zip_path: str, public_key, signature: str) -> bool:
    """Verify a base64-encoded signature string against model content.

    Args:
        zip_path: Path to the model ZIP file.
        public_key: An ``Ed25519PublicKey`` from the ``cryptography`` package.
        signature: Base64-encoded signature string — as returned by the API ``signature`` field.

    Returns:
        ``True`` if the signature is valid, ``False`` otherwise.
    """
    try:
        sig_bytes = base64.b64decode(signature)
    except Exception:
        ScaleoutLogger().warning(f"Malformed signature: {signature}")
        return False
    content_hash = compute_model_content_hash(zip_path)
    try:
        public_key.verify(sig_bytes, content_hash)
        return True
    except InvalidSignature:
        return False
