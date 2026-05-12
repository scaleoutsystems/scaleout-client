import pytest
import yaml
from click.testing import CliRunner
from scaleout.cli.main import context_cmd, remove_cmd, login
from unittest.mock import patch, MagicMock

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def mock_home(tmp_path):
    """Fixture to mock HOME_DIR to a temporary path."""
    with patch("scaleout.cli.shared.HOME_DIR", str(tmp_path)):
        yield tmp_path

def setup_config(mock_home, contexts=None, active_index=None):
    """Helper to initialize the .scaleout directory with context and active files."""
    context_dir = mock_home / ".scaleout"
    context_dir.mkdir(parents=True, exist_ok=True)

    if contexts is not None:
        with open(context_dir / "contexts.yaml", "w") as f:
            yaml.dump(contexts, f)

    if active_index is not None:
        with open(context_dir / "active.yaml", "w") as f:
            yaml.dump({"active_index": active_index}, f)

    return context_dir


def test_context_list_empty(runner, mock_home):
    """Test 'scaleout context' when no contexts exist."""
    result = runner.invoke(context_cmd, [])
    assert result.exit_code == 0
    assert "No contexts found" in result.output


def test_context_list_with_data(runner, mock_home):
    """Test 'scaleout context' listing with multiple contexts."""
    setup_config(
        mock_home,
        contexts=[
            {"name": "production", "host": "https://prod.example.com", "token": "prod-token"},
            {"name": "staging", "host": "https://stage.example.com", "token": "stage-token"}
        ],
        active_index=0
    )

    result = runner.invoke(context_cmd, [])

    assert result.exit_code == 0
    assert "Available contexts:" in result.output
    assert "★ [0] production" in result.output
    assert "  [1] staging" in result.output
    assert "Host: https://prod.example.com" in result.output
    assert "Host: https://stage.example.com" in result.output


def test_context_switch_by_index(runner, mock_home):
    """Test switching context using an index."""
    context_dir = setup_config(
        mock_home,
        contexts=[
            {"name": "ctx0", "host": "http://h0"},
            {"name": "ctx1", "host": "http://h1"}
        ]
    )

    # Switch to index 1
    result = runner.invoke(context_cmd, ["1"])

    assert result.exit_code == 0
    assert "Switched to context 'ctx1'" in result.output

    # Verify active.yaml was updated
    with open(context_dir / "active.yaml", "r") as f:
        active_data = yaml.safe_load(f)
    assert active_data["active_index"] == 1


def test_context_switch_by_name(runner, mock_home):
    """Test switching context using a name."""
    context_dir = setup_config(
        mock_home,
        contexts=[
            {"name": "production", "host": "http://h0"},
            {"name": "staging", "host": "http://h1"}
        ]
    )

    # Switch to 'staging'
    result = runner.invoke(context_cmd, ["staging"])

    assert result.exit_code == 0
    assert "Switched to context 'staging'" in result.output

    # Verify active.yaml was updated to index 1
    with open(context_dir / "active.yaml", "r") as f:
        active_data = yaml.safe_load(f)
    assert active_data["active_index"] == 1


def test_context_switch_invalid_index(runner, mock_home):
    """Test switching context with an out-of-bounds index."""
    setup_config(
        mock_home,
        contexts=[{"name": "ctx0", "host": "http://h0"}]
    )

    result = runner.invoke(context_cmd, ["5"])

    assert result.exit_code != 0
    assert "Invalid index 5" in result.output
    assert "Available range: 0-0" in result.output


def test_context_switch_invalid_name(runner, mock_home):
    """Test switching context with a non-existent name."""
    setup_config(
        mock_home,
        contexts=[{"name": "ctx0", "host": "http://h0"}]
    )

    result = runner.invoke(context_cmd, ["nonexistent"])

    assert result.exit_code != 0
    assert "Context 'nonexistent' not found" in result.output


def test_remove_context_by_index(runner, mock_home):
    """Test removing a context using its index."""
    context_dir = setup_config(
        mock_home,
        contexts=[
            {"name": "keep", "host": "http://keep"},
            {"name": "remove", "host": "http://remove"}
        ]
    )

    # Remove index 1 with -y
    result = runner.invoke(remove_cmd, ["1", "-y"])

    assert result.exit_code == 0
    assert "Removed context 'remove'" in result.output

    with open(context_dir / "contexts.yaml", "r") as f:
        remaining = yaml.safe_load(f)
    assert len(remaining) == 1
    assert remaining[0]["name"] == "keep"


def test_remove_context_by_name(runner, mock_home):
    """Test removing a context using its name."""
    context_dir = setup_config(
        mock_home,
        contexts=[
            {"name": "keep", "host": "http://keep"},
            {"name": "remove-me", "host": "http://remove"}
        ]
    )

    # Remove 'remove-me' with -y
    result = runner.invoke(remove_cmd, ["remove-me", "-y"])

    assert result.exit_code == 0
    assert "Removed context 'remove-me'" in result.output

    with open(context_dir / "contexts.yaml", "r") as f:
        remaining = yaml.safe_load(f)
    assert len(remaining) == 1
    assert remaining[0]["name"] == "keep"


def test_remove_context_with_confirmation(runner, mock_home):
    """Test removing a context with interactive confirmation."""
    context_dir = setup_config(
        mock_home,
        contexts=[{"name": "ctx0", "host": "http://h0"}]
    )

    # Invoke without -y, provide 'y' to prompt
    result = runner.invoke(remove_cmd, ["0"], input="y\n")

    assert "Are you sure you want to remove this context?" in result.output
    assert result.exit_code == 0
    assert "Removed context 'ctx0'" in result.output

    with open(context_dir / "contexts.yaml", "r") as f:
        remaining = yaml.safe_load(f)
    assert len(remaining) == 0


def test_remove_context_cancel_confirmation(runner, mock_home):
    """Test cancelling context removal at the confirmation prompt."""
    context_dir = setup_config(
        mock_home,
        contexts=[{"name": "ctx0", "host": "http://h0"}]
    )

    # Invoke without -y, provide 'n' to prompt
    result = runner.invoke(remove_cmd, ["0"], input="n\n")

    assert "Are you sure you want to remove this context?" in result.output
    assert "Cancelled" in result.output
    assert result.exit_code == 0 # click.confirm(abort=False) returns 0 if cancelled

    with open(context_dir / "contexts.yaml", "r") as f:
        remaining = yaml.safe_load(f)
    assert len(remaining) == 1


def test_remove_context_invalid_index(runner, mock_home):
    """Test removing a context with an out-of-bounds index."""
    setup_config(mock_home, contexts=[{"name": "ctx0"}])

    result = runner.invoke(remove_cmd, ["1", "-y"])

    assert result.exit_code != 0
    assert "Invalid index 1" in result.output


def test_remove_context_invalid_name(runner, mock_home):
    """Test removing a context with a non-existent name."""
    setup_config(mock_home, contexts=[{"name": "ctx0"}])

    result = runner.invoke(remove_cmd, ["nonexistent", "-y"])

    assert result.exit_code != 0
    assert "Context 'nonexistent' not found" in result.output


# ==============================================================================
# Test login command
# ==============================================================================

@patch("scaleout.cli.main.requests.post")
@patch("scaleout.cli.main.requests.get")
@patch("scaleout.cli.main.webbrowser.open")
@patch("scaleout.cli.main.time.sleep") # speed up tests
def test_login_success(mock_sleep, mock_browser, mock_get, mock_post, runner, mock_home):
    """Test the 'scaleout login' flow successfully completes and creates a context."""
    instance_url = "http://test-server"

    # Mock Step 1: Init login
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "requestId": "req-123",
            "pollToken": "poll-456",
            "loginUrl": "http://test-server/login"
        }
    )

    # Mock Step 2: Poll for status
    # First call: pending, Second call: done
    mock_get.side_effect = [
        MagicMock(status_code=200, json=lambda: {"status": "pending"}),
        MagicMock(status_code=200, json=lambda: {
            "status": "done",
            "access_token": "secret-token",
            "api_url": "http://api.test-server"
        })
    ]

    result = runner.invoke(login, [instance_url])

    assert result.exit_code == 0
    assert "Logged in successfully to http://api.test-server" in result.output

    # Verify context was created
    context_dir = mock_home / ".scaleout"
    with open(context_dir / "contexts.yaml", "r") as f:
        contexts = yaml.safe_load(f)

    assert len(contexts) == 1
    assert contexts[0]["name"] == "api.test-server"
    assert contexts[0]["host"] == "http://api.test-server"
    assert contexts[0]["token"] == "secret-token"


@patch("scaleout.cli.main.requests.post")
@patch("scaleout.cli.main.requests.get")
@patch("scaleout.cli.main.webbrowser.open")
@patch("scaleout.cli.main.time.sleep")
def test_login_updates_existing_context(mock_sleep, mock_browser, mock_get, mock_post, runner, mock_home):
    """Test that login updates an existing context if BOTH name and host match."""
    instance_url = "http://api.test-server"

    # Pre-setup an existing context with same name and same host
    setup_config(
        mock_home,
        contexts=[{"name": "api.test-server", "host": instance_url, "token": "old-token"}]
    )

    # Mock Login sequence
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"requestId": "r", "pollToken": "p", "loginUrl": "u"})
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {
        "status": "done",
        "access_token": "new-token",
        "api_url": instance_url
    })

    result = runner.invoke(login, [instance_url])

    assert result.exit_code == 0

    # Verify we still only have ONE context, but with the NEW token
    context_dir = mock_home / ".scaleout"
    with open(context_dir / "contexts.yaml", "r") as f:
        contexts = yaml.safe_load(f)

    assert len(contexts) == 1, f"Found {len(contexts)} contexts instead of 1. Identical host/name should update!"
    assert contexts[0]["token"] == "new-token"
