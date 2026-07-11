from ev_vision.cli import main


def test_validate_config_command() -> None:
    assert main(["validate-config", "--config", "config/default.yaml"]) == 0


def test_protocol_selftest_command() -> None:
    assert main(["protocol-selftest"]) == 0


def test_mock_run_finishes_laser_off(capsys) -> None:
    assert main(["mock-run", "--cycles", "10"]) == 0
    assert "laser=OFF" in capsys.readouterr().out
