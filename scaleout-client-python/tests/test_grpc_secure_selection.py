"""Unit tests for how the client picks a secure vs. insecure gRPC channel.

When SCALEOUT_GRPC_SECURE is set, the combiner-reported ``grpc_secure`` may only
upgrade to TLS, never downgrade. When unset, the combiner decides, defaulting to TLS
when the combiner doesn't report it. No channel is opened; the channel initializers are mocked.
"""

import unittest
from unittest.mock import MagicMock, patch

from scaleout.client.grpc_handler import GrpcConnectionOptions, GrpcHandler


class TestGrpcConnectionOptionsFromDict(unittest.TestCase):
    def test_grpc_secure_values(self):
        base = {"public_hostname": "h", "port": 1}
        self.assertIs(GrpcConnectionOptions.from_dict({**base, "grpc_secure": True}).secure, True)
        self.assertIs(GrpcConnectionOptions.from_dict({**base, "grpc_secure": False}).secure, False)
        self.assertIsNone(GrpcConnectionOptions.from_dict(base).secure)


class TestInitChannelSelection(unittest.TestCase):
    # (env SCALEOUT_GRPC_SECURE, combiner-reported secure) -> (expect secure channel, expect warning)
    CASES = {
        (True, True): (True, False),
        (True, False): (True, True),  # must not downgrade
        (True, None): (True, False),
        (False, True): (True, True),  # upgrade allowed
        (False, False): (False, False),
        (False, None): (False, False),
        (None, True): (True, False),  # unset: follow combiner
        (None, False): (False, False),
        (None, None): (True, False),  # unset, older combiner: TLS
    }

    def _run(self, env, secure):
        handler = GrpcHandler.__new__(GrpcHandler)
        handler.secure = secure
        handler._init_secure_channel = MagicMock()
        handler._init_insecure_channel = MagicMock()
        logger = MagicMock()
        with patch("scaleout.client.grpc_handler.SCALEOUT_GRPC_SECURE", env), patch("scaleout.client.grpc_handler.ScaleoutLogger", return_value=logger):
            handler._init_channel("host", 1234)
        return handler, logger

    def test_all_combinations(self):
        for (env, secure), (expect_secure, expect_warning) in self.CASES.items():
            with self.subTest(env=env, secure=secure):
                handler, logger = self._run(env, secure)
                if expect_secure:
                    handler._init_secure_channel.assert_called_once_with("host", 1234)
                    handler._init_insecure_channel.assert_not_called()
                else:
                    handler._init_insecure_channel.assert_called_once_with("host", 1234)
                    handler._init_secure_channel.assert_not_called()
                self.assertEqual(logger.warning.called, expect_warning)


if __name__ == "__main__":
    unittest.main()
