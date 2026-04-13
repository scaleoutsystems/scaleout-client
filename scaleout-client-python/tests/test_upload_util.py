import os
import unittest
from unittest.mock import patch, MagicMock, call
import tempfile
import responses

from scaleout.cli.upload_util import perform_chunked_upload


class TestUploadUtil(unittest.TestCase):
    """Test the chunked upload utility."""

    @responses.activate
    def test_perform_chunked_upload_success(self):
        """Test a successful chunked upload flow."""
        base_url = "http://localhost:8092"
        token = "test-token"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Setup mock responses for the API endpoints
        responses.add(
            responses.POST, 
            f"{base_url}/api/v1/file-upload/init",
            json={"upload_id": "mock_upload_id_123"},
            status=200
        )
        
        responses.add(
            responses.POST, 
            f"{base_url}/api/v1/file-upload/mock_upload_id_123/chunk",
            json={"message": "Chunk uploaded"},
            status=200
        )
        
        responses.add(
            responses.POST, 
            f"{base_url}/api/v1/file-upload/mock_upload_id_123/complete",
            json={"file_token": "mock_file_token_456"},
            status=200
        )

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Create a file larger than 10MB to test multiple chunks natively
            temp_file.write(b"0" * (11 * 1024 * 1024))  # 11 MB
            temp_file_path = temp_file.name

        try:
            # Let the utility read the 11MB file and chunk it into slices
            file_token = perform_chunked_upload(base_url, token, temp_file_path, headers)
            
            self.assertEqual(file_token, "mock_file_token_456")
            
            # Assert the correct sequence of requests were made (Init + 13 Chunks + Complete)
            self.assertEqual(len(responses.calls), 15)
            
            # 1. Init
            self.assertEqual(responses.calls[0].request.url, f"{base_url}/api/v1/file-upload/init")
            
            # 2. Chunks (Indices 0..12)
            for i in range(13):
                self.assertEqual(responses.calls[i + 1].request.url, f"{base_url}/api/v1/file-upload/mock_upload_id_123/chunk")
                self.assertEqual(responses.calls[i + 1].request.headers["X-Chunk-Index"], str(i))
            
            # 3. Complete
            self.assertEqual(responses.calls[14].request.url, f"{base_url}/api/v1/file-upload/mock_upload_id_123/complete")
            
        finally:
            os.unlink(temp_file_path)

    @responses.activate
    @patch("scaleout.cli.upload_util.click.secho")
    def test_perform_chunked_upload_413_recovery(self, mock_secho):
        """Test that the client recovers from a 413 error by halving the chunk size."""
        base_url = "http://localhost:8092"
        token = "test-token"
        headers = {"Authorization": f"Bearer {token}"}

        # 1. First init (900 KB)
        responses.add(
            responses.POST,
            f"{base_url}/api/v1/file-upload/init",
            json={"upload_id": "upload_900kb", "chunk_size": 900*1024, "total_chunks": 2},
            status=200
        )

        # 2. First chunk fails with 413
        responses.add(
            responses.POST,
            f"{base_url}/api/v1/file-upload/upload_900kb/chunk",
            status=413
        )

        # 3. Abort called for old session
        responses.add(
            responses.POST,
            f"{base_url}/api/v1/file-upload/upload_900kb/abort",
            status=200
        )

        # 4. Second init (450 KB)
        responses.add(
            responses.POST,
            f"{base_url}/api/v1/file-upload/init",
            json={"upload_id": "upload_450kb", "chunk_size": 450*1024, "total_chunks": 3},
            status=200
        )

        # 5. Chunks for second session succeed
        responses.add(
            responses.POST,
            f"{base_url}/api/v1/file-upload/upload_450kb/chunk",
            json={"message": "Chunk uploaded"},
            status=200
        )

        # 6. Complete second session
        responses.add(
            responses.POST,
            f"{base_url}/api/v1/file-upload/upload_450kb/complete",
            json={"file_token": "recovered_token"},
            status=200
        )

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"0" * (1024 * 1024)) # 1 MB file
            temp_file_path = temp_file.name

        try:
            file_token = perform_chunked_upload(base_url, token, temp_file_path, headers)
            self.assertEqual(file_token, "recovered_token")
            mock_secho.assert_any_call("\nProxy rejected chunk size (413). Retrying with 450 KB chunks...", fg="yellow")
        finally:
            os.unlink(temp_file_path)

    def test_perform_chunked_upload_file_not_found(self):
        """Test trying to upload a non-existent file."""
        with self.assertRaises(SystemExit) as cm:
            perform_chunked_upload("http://localhost:8092", "test-token", "/tmp/does-not-exist.dummy", {})
        self.assertEqual(cm.exception.code, 1)

    @responses.activate
    @patch("scaleout.cli.upload_util.click.secho")
    def test_perform_chunked_upload_init_fails(self, mock_secho):
        """Test failure during initialization."""
        base_url = "http://localhost:8092"
        responses.add(
            responses.POST, 
            f"{base_url}/api/v1/file-upload/init",
            json={"message": "File too large"},
            status=400
        )
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"0" * 15)
            temp_file_path = temp_file.name
            
        try:
            with self.assertRaises(SystemExit) as cm:
                perform_chunked_upload(base_url, "test-token", temp_file_path, {})
            self.assertEqual(cm.exception.code, 1)
            mock_secho.assert_any_call("Details: {\"message\": \"File too large\"}", fg="red")
        finally:
            os.unlink(temp_file_path)

    @responses.activate
    @patch("scaleout.cli.upload_util.click.secho")
    def test_perform_chunked_upload_chunk_fails(self, mock_secho):
        """Test failure during chunk upload."""
        base_url = "http://localhost:8092"
        responses.add(
            responses.POST, 
            f"{base_url}/api/v1/file-upload/init",
            json={"upload_id": "mock_upload_id_123"},
            status=200
        )
        responses.add(
            responses.POST, 
            f"{base_url}/api/v1/file-upload/mock_upload_id_123/chunk",
            json={"message": "Internal Server Error"},
            status=500
        )

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"0" * 15)
            temp_file_path = temp_file.name

        try:
            with self.assertRaises(SystemExit) as cm:
                perform_chunked_upload(base_url, "test-token", temp_file_path, {})
            self.assertEqual(cm.exception.code, 1)
            mock_secho.assert_any_call("Details: {\"message\": \"Internal Server Error\"}", fg="red")
        finally:
            os.unlink(temp_file_path)

    @responses.activate
    @patch("scaleout.cli.upload_util.click.secho")
    def test_perform_chunked_upload_complete_fails(self, mock_secho):
        """Test failure during completion phase."""
        base_url = "http://localhost:8092"
        responses.add(
            responses.POST, 
            f"{base_url}/api/v1/file-upload/init",
            json={"upload_id": "mock_upload_id_123"},
            status=200
        )
        responses.add(
            responses.POST, 
            f"{base_url}/api/v1/file-upload/mock_upload_id_123/chunk",
            json={"message": "Chunk uploaded"},
            status=200
        )
        responses.add(
            responses.POST, 
            f"{base_url}/api/v1/file-upload/mock_upload_id_123/complete",
            json={"message": "Missing chunks"},
            status=400
        )

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"0" * 15)
            temp_file_path = temp_file.name
            
        try:
            with self.assertRaises(SystemExit) as cm:
                perform_chunked_upload(base_url, "test-token", temp_file_path, {})
            self.assertEqual(cm.exception.code, 1)
            mock_secho.assert_any_call("Details: {\"message\": \"Missing chunks\"}", fg="red")
        finally:
            os.unlink(temp_file_path)

if __name__ == "__main__":
    unittest.main()
