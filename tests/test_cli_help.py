from typer.testing import CliRunner

from rejuvbridge.cli import app


def test_cli_help_runs():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])  # typer auto-generates help
    assert result.exit_code == 0
    assert "RejuvBridge CLI" in result.stdout

