"""Unit tests for GrpcAuth."""

import unittest
from unittest.mock import Mock

from scaleout.client.grpc_handler import GrpcAuth


class TestGrpcAuth(unittest.TestCase):
    """Test cases for GrpcAuth class."""

    def test_grpc_auth_with_valid_token(self):
        """Test GrpcAuth adds authorization metadata with valid token."""
        token = "valid-token"
        auth = GrpcAuth(token)
        
        context = Mock()
        callback = Mock()
        
        auth(context, callback)
        
        # Should call callback with authorization header
        callback.assert_called_once()
        args = callback.call_args[0]
        self.assertEqual(len(args), 2)
        self.assertEqual(args[0], (("authorization", "Bearer valid-token"),))
        self.assertIsNone(args[1])

    def test_grpc_auth_with_none_token(self):
        """Test GrpcAuth skips authorization metadata when token is None."""
        token = None
        auth = GrpcAuth(token)
        
        context = Mock()
        callback = Mock()
        
        auth(context, callback)
        
        # Should call callback with empty metadata tuple
        callback.assert_called_once()
        args = callback.call_args[0]
        self.assertEqual(len(args), 2)
        self.assertEqual(args[0], ())
        self.assertIsNone(args[1])

    def test_grpc_auth_with_callable_valid_token(self):
        """Test GrpcAuth with callable that returns valid token."""
        token_callable = lambda: "callable-token"
        auth = GrpcAuth(token_callable)
        
        context = Mock()
        callback = Mock()
        
        auth(context, callback)
        
        # Should call callback with authorization header from callable
        callback.assert_called_once()
        args = callback.call_args[0]
        self.assertEqual(args[0], (("authorization", "Bearer callable-token"),))

    def test_grpc_auth_with_callable_none_token(self):
        """Test GrpcAuth with callable that returns None."""
        token_callable = lambda: None
        auth = GrpcAuth(token_callable)
        
        context = Mock()
        callback = Mock()
        
        auth(context, callback)
        
        # Should call callback with empty metadata tuple
        callback.assert_called_once()
        args = callback.call_args[0]
        self.assertEqual(args[0], ())


if __name__ == "__main__":
    unittest.main()
