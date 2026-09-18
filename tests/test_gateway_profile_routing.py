"""Regression tests for profile-scoped WebUI -> Gateway routing."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from api.gateway_chat import (
    _STREAM_RUN_IDS,
    _gateway_base_url_for_profile,
    stop_gateway_run,
)


def test_gateway_profile_url_uses_multiplex_prefix():
    env = {"HERMES_WEBUI_GATEWAY_BASE_URL": "http://127.0.0.1:8642/"}

    assert _gateway_base_url_for_profile("atlas", environ=env) == (
        "http://127.0.0.1:8642/p/atlas"
    )
    assert _gateway_base_url_for_profile("HOFFEE CMO", environ=env) == (
        "http://127.0.0.1:8642/p/HOFFEE%20CMO"
    )


def test_gateway_profile_url_preserves_owner_route_without_profile():
    env = {"HERMES_WEBUI_GATEWAY_BASE_URL": "http://127.0.0.1:8642/"}

    assert _gateway_base_url_for_profile(None, environ=env) == "http://127.0.0.1:8642"


def test_stop_gateway_run_targets_owning_profile():
    stream_id = "stream-profile-stop"
    _STREAM_RUN_IDS[stream_id] = "run-stop"
    response = MagicMock()
    response.status = 202
    response.code = 202
    response.geturl.return_value = (
        "http://127.0.0.1:8642/p/mentor/v1/runs/run-stop/stop"
    )
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    opener = MagicMock()
    opener.open.return_value = response

    try:
        with patch("urllib.request.build_opener", return_value=opener), patch(
            "api.gateway_chat._gateway_api_key", return_value="secret"
        ), patch(
            "api.gateway_chat.stream_owner_session_id", return_value="session-profile-stop"
        ), patch(
            "api.gateway_chat.get_session", return_value=SimpleNamespace(profile="mentor")
        ):
            assert stop_gateway_run("run-stop") is True
    finally:
        _STREAM_RUN_IDS.pop(stream_id, None)

    request = opener.open.call_args.args[0]
    assert request.full_url == (
        "http://127.0.0.1:8642/p/mentor/v1/runs/run-stop/stop"
    )
    assert request.get_header("Authorization") == "Bearer secret"
