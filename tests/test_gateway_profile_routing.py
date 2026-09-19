"""Regression tests for profile-scoped WebUI -> Gateway routing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from api.gateway_chat import (
    _clear_gateway_run_starting,
    _gateway_base_url_for_profile,
    _mark_gateway_run_starting,
    _publish_gateway_run_id,
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
    _mark_gateway_run_starting(stream_id, profile="mentor")
    _publish_gateway_run_id(stream_id, "run-stop")
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
        ):
            assert stop_gateway_run("run-stop") is True
    finally:
        _clear_gateway_run_starting(stream_id)

    request = opener.open.call_args.args[0]
    assert request.full_url == (
        "http://127.0.0.1:8642/p/mentor/v1/runs/run-stop/stop"
    )
    assert request.get_header("Authorization") == "Bearer secret"


def test_stop_gateway_run_fails_closed_without_atomic_profile_binding():
    with patch("urllib.request.build_opener") as build_opener:
        assert stop_gateway_run("unowned-run") is False
    build_opener.assert_not_called()


def test_stop_gateway_run_fails_closed_on_colliding_run_id():
    streams = ("stream-collision-a", "stream-collision-b")
    _mark_gateway_run_starting(streams[0], profile="atlas")
    _mark_gateway_run_starting(streams[1], profile="mentor")
    _publish_gateway_run_id(streams[0], "colliding-run")
    _publish_gateway_run_id(streams[1], "colliding-run")
    try:
        with patch("urllib.request.build_opener") as build_opener:
            assert stop_gateway_run("colliding-run") is False
        build_opener.assert_not_called()
    finally:
        for stream_id in streams:
            _clear_gateway_run_starting(stream_id)
