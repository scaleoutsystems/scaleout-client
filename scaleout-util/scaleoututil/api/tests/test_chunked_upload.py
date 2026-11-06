"""Unit tests for Scaleout._perform_chunked_upload.

These migrated from the CLI's old cli/upload_util.py (removed in issue #815): the chunked-upload
logic now lives solely in the Scaleout SDK, so its coverage lives here too.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import requests
import responses

from scaleoututil.api.client import Scaleout

BASE_URL = "http://example.com:80"


class TestChunkedUpload(unittest.TestCase):
    """Test the SDK chunked upload flow end to end (mocked HTTP)."""

    def setUp(self):
        # Avoid any real auth/token-cache interaction during construction.
        self.token_patch = patch("scaleoututil.api.client.Login")
        self.token_patch.start()
        self.cache_patch = patch("scaleoututil.api.client.TokenCache")
        mock_cache = self.cache_patch.start()
        mock_cache.return_value.exists.return_value = False

        self.client = Scaleout(host="example.com", secure=False)

    def tearDown(self):
        self.token_patch.stop()
        self.cache_patch.stop()

    def _write_temp(self, size: int) -> str:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"0" * size)
            return f.name

    @responses.activate
    def test_success(self):
        """A successful upload returns the file_token and issues init + chunks + complete."""
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/init", json={"upload_id": "u123"}, status=200)
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/u123/chunk", json={"message": "ok"}, status=200)
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/u123/complete", json={"file_token": "tok456"}, status=200)

        path = self._write_temp(11 * 1024 * 1024)  # 11 MB -> 13 chunks at 900 KB
        try:
            file_token = self.client._perform_chunked_upload(path)
            self.assertEqual(file_token, "tok456")
            # init + 13 chunks + complete
            self.assertEqual(len(responses.calls), 15)
            self.assertEqual(responses.calls[0].request.url, f"{BASE_URL}/api/v1/file-upload/init")
            for i in range(13):
                self.assertEqual(responses.calls[i + 1].request.headers["X-Chunk-Index"], str(i))
            self.assertEqual(responses.calls[14].request.url, f"{BASE_URL}/api/v1/file-upload/u123/complete")
        finally:
            os.unlink(path)

    @responses.activate
    def test_413_recovery(self):
        """A 413 on a chunk halves the chunk size, aborts the old session, and retries."""
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/init", json={"upload_id": "u900"}, status=200)
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/u900/chunk", status=413)
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/u900/abort", status=200)
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/init", json={"upload_id": "u450"}, status=200)
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/u450/chunk", json={"message": "ok"}, status=200)
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/u450/complete", json={"file_token": "recovered"}, status=200)

        path = self._write_temp(1024 * 1024)  # 1 MB
        try:
            file_token = self.client._perform_chunked_upload(path)
            self.assertEqual(file_token, "recovered")
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        """A missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.client._perform_chunked_upload("/tmp/does-not-exist-815.dummy")

    @responses.activate
    def test_init_fails(self):
        """A non-413 init error propagates as an HTTPError."""
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/init", json={"message": "File too large"}, status=400)
        path = self._write_temp(15)
        try:
            with self.assertRaises(requests.exceptions.HTTPError):
                self.client._perform_chunked_upload(path)
        finally:
            os.unlink(path)

    @responses.activate
    def test_chunk_fails(self):
        """A non-413 chunk error propagates as an HTTPError."""
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/init", json={"upload_id": "u123"}, status=200)
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/u123/chunk", json={"message": "boom"}, status=500)
        path = self._write_temp(15)
        try:
            with self.assertRaises(requests.exceptions.HTTPError):
                self.client._perform_chunked_upload(path)
        finally:
            os.unlink(path)

    @responses.activate
    def test_complete_fails(self):
        """A failure finalizing the upload propagates as an HTTPError."""
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/init", json={"upload_id": "u123"}, status=200)
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/u123/chunk", json={"message": "ok"}, status=200)
        responses.add(responses.POST, f"{BASE_URL}/api/v1/file-upload/u123/complete", json={"message": "Missing chunks"}, status=400)
        path = self._write_temp(15)
        try:
            with self.assertRaises(requests.exceptions.HTTPError):
                self.client._perform_chunked_upload(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
