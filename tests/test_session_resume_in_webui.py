"""Focused tests for the safe backend ``POST /api/session/resume_in_webui``.

This is the *only* backend path that converts a foreign, read-only-projected
session (CLI / TUI / ACP / Desktop) into a writable WebUI sidecar while
``HERMES_WEBUI_EXTERNAL_STATE_READ_ONLY=1`` is in force. The tests below pin
the whole gate list:

  * disabled (allowlist unset)
  * invalid profile shape
  * wrong active profile
  * missing source row
  * bad (non-allowlisted) source
  * active/unended source (concurrent-writer guard)
  * lineage root/tip mismatch
  * happy path — and the source ``state.db`` bytes are unchanged
  * idempotent re-resume
  * cross-profile same-sid flat sidecar collision
  * existing read-only projection regression (ordinary import stays read-only)

All DBs are synthetic SQLite files under ``tmp_path``; no production state is
touched and every module-global path is monkeypatched per-test.
"""
import hashlib
import io
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class _Recorder:
    """Captures the (msg, status) of ``bad()`` or (payload, status) of ``j()``."""

    def __init__(self):
        self.bad: Any = None
        self.j: Any = None

    @property
    def status(self):
        if self.j is not None:
            return self.j[1]
        if self.bad is not None:
            return self.bad[1]
        return None

    def payload(self):
        return self.j[0] if self.j is not None else None

    def error(self):
        return self.bad[0] if self.bad is not None else None


def _install_response_recorder(routes, monkeypatch):
    """Swap ``j``/``bad`` for pure-python recorders so tests need no socket
    handler. Returns a fresh recorder dict passed as the fake handler."""
    rec = _Recorder()

    def fake_bad(_handler, msg, status=400):
        rec.bad = (msg, status)
        return None

    def fake_j(_handler, payload, status=200, extra_headers=None, *, pretty=True):
        rec.j = (payload, status)
        return None

    monkeypatch.setattr(routes, "bad", fake_bad)
    monkeypatch.setattr(routes, "j", fake_j)
    return rec


_STATE_DB_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT,
    session_source TEXT,
    model TEXT,
    parent_session_id TEXT,
    started_at REAL,
    ended_at REAL,
    end_reason TEXT,
    title TEXT,
    cwd TEXT,
    message_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,
    content TEXT,
    timestamp REAL
);
"""


def _make_state_db(
    path: Path,
    *,
    sid: str,
    source: str = "cli",
    session_source: str = "cli",
    model: str = "test-model",
    title: str = "Resumable session",
    parent_session_id=None,
    started_at: float = 1700000000.0,
    ended_at: "float | None" = 1700000100.0,
    end_reason: "str | None" = "cli_close",
    messages: int = 3,
    insert_row: bool = True,
) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_STATE_DB_SQL)
        if insert_row:
            conn.execute(
                "INSERT INTO sessions (id, source, session_source, model, "
                "parent_session_id, started_at, ended_at, end_reason, title, cwd, "
                "message_count) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid, source, session_source, model, parent_session_id,
                    started_at, ended_at, end_reason, title, "/tmp/ws", messages,
                ),
            )
        for i in range(messages):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?,?,?,?)",
                (sid, "user" if i % 2 == 0 else "assistant", f"msg {i}",
                 started_at + i),
            )
        conn.commit()
    finally:
        conn.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def routes_module():
    return pytest.importorskip("api.routes")


@pytest.fixture
def resume_env(routes_module, tmp_path, monkeypatch):
    """Wire every external path the endpoint touches onto ``tmp_path``.

    * ``get_hermes_home_for_profile`` -> per-profile dirs under tmp_path
    * active profile -> "alpha"
    * ``SESSION_DIR`` (routes + models) -> tmp_path/webui-state/sessions
    * operator allowlist -> "alpha"
    * response helpers -> pure recorders (returned as the fake handler)
    """
    import api.models as models
    import api.profiles as profiles

    profiles_root = tmp_path / "profiles"
    homes = {}
    for name in ("alpha", "beta"):
        home = profiles_root / name
        home.mkdir(parents=True)
        homes[name] = home

    sessions_dir = tmp_path / "webui-state" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "_index.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        profiles, "get_hermes_home_for_profile",
        lambda name: homes.get(str(name), homes["alpha"]),
    )
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "alpha")
    # Deterministic root-profile handling (no hermes_cli subprocess / cache).
    monkeypatch.setattr(
        profiles, "_is_root_profile",
        lambda name: (name or "default") == "default",
    )

    monkeypatch.setattr(routes_module, "SESSION_DIR", sessions_dir)
    monkeypatch.setattr(models, "SESSION_DIR", sessions_dir)
    monkeypatch.setattr(models, "_active_state_db_path", lambda: homes["alpha"] / "state.db")
    monkeypatch.setattr(routes_module, "publish_session_list_changed", lambda *a, **k: None)

    monkeypatch.setenv("HERMES_WEBUI_RESUME_ALLOW_PROFILES", "alpha")
    monkeypatch.delenv("HERMES_WEBUI_EXTERNAL_STATE_READ_ONLY", raising=False)

    rec = _install_response_recorder(routes_module, monkeypatch)
    return {"routes": routes_module, "models": models, "rec": rec,
            "homes": homes, "sessions_dir": sessions_dir}


def _body(sid="sess-alpha-1", profile="alpha", root=None, tip=None,
          confirm=True, **extra):
    payload = {
        "session_id": sid,
        "profile": profile,
        "lineage_root_id": sid if root is None else root,
        "lineage_tip_id": sid if tip is None else tip,
        "confirm": confirm,
    }
    payload.update(extra)
    return payload


def _alpha_db(env) -> Path:
    return env["homes"]["alpha"] / "state.db"


# ---------------------------------------------------------------------------
# Gate: allowlist disabled
# ---------------------------------------------------------------------------


def test_disabled_when_allowlist_unset(resume_env, monkeypatch):
    monkeypatch.delenv("HERMES_WEBUI_RESUME_ALLOW_PROFILES", raising=False)
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1")
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body())
    assert env["rec"].status == 403
    assert "disabled" in env["rec"].error()


def test_disabled_when_allowlist_blank(resume_env, monkeypatch):
    monkeypatch.setenv("HERMES_WEBUI_RESUME_ALLOW_PROFILES", "  ,  ")
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1")
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body())
    assert env["rec"].status == 403


def test_profile_not_in_allowlist(resume_env):
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1")
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(profile="beta"))
    # beta is a valid profile shape but not on the operator allowlist
    assert env["rec"].status == 403
    assert "not allowed" in env["rec"].error()


# ---------------------------------------------------------------------------
# Gate: profile validation / active profile
# ---------------------------------------------------------------------------


def test_invalid_profile_rejected(resume_env):
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(profile="Bad Name!"))
    assert env["rec"].status == 400
    assert "invalid profile" in env["rec"].error()


def test_missing_profile_rejected(resume_env):
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(profile=""))
    assert env["rec"].status == 400
    assert "invalid profile" in env["rec"].error()


def test_wrong_active_profile_rejected(resume_env, monkeypatch):
    import api.profiles as profiles

    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "beta")
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1")
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(profile="alpha"))
    assert env["rec"].status == 403
    assert "active profile" in env["rec"].error()


def test_confirm_required(resume_env):
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1")
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(confirm=False))
    assert env["rec"].status == 400


def test_lineage_ids_required(resume_env):
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1")
    env = resume_env
    body = _body()
    body.pop("lineage_root_id")
    env["routes"]._handle_session_resume_in_webui(env["rec"], body)
    assert env["rec"].status == 400


# ---------------------------------------------------------------------------
# Gate: source row / source allowlist / concurrent writer
# ---------------------------------------------------------------------------


def test_missing_source_row_404(resume_env):
    # DB exists (with schema) but no row for the requested sid.
    db = _alpha_db(resume_env)
    _make_state_db(db, sid="some-other-sid", insert_row=False)
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body())
    assert env["rec"].status == 404


def test_missing_source_db_404_no_fallback(resume_env):
    # No state.db at all for the profile -> hard 404, never the active store.
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body())
    assert env["rec"].status == 404


def test_source_store_errors_are_sanitized(resume_env, monkeypatch):
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1")
    env = resume_env

    def fail_read(*_args, **_kwargs):
        raise sqlite3.OperationalError("cannot open /private/secret/profile/state.db")

    monkeypatch.setattr(env["routes"], "_read_source_session_row", fail_read)
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body())

    assert env["rec"].status == 500
    assert "/private/secret" not in env["rec"].error()
    assert "<path>" in env["rec"].error()


def test_sidecar_store_errors_are_sanitized(resume_env, monkeypatch):
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1")
    env = resume_env

    def fail_store(*_args, **_kwargs):
        raise OSError("permission denied: /private/secret/webui/sess-alpha-1.json")

    monkeypatch.setattr(env["routes"], "import_cli_session", fail_store)
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body())

    assert env["rec"].status == 500
    assert "/private/secret" not in env["rec"].error()
    assert "<path>" in env["rec"].error()


def test_transcript_read_errors_fail_closed_and_are_sanitized(resume_env, monkeypatch):
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1")
    env = resume_env

    def fail_messages(*_args, **_kwargs):
        raise sqlite3.OperationalError("cannot read /private/secret/profile/state.db")

    monkeypatch.setattr(env["routes"], "get_state_db_session_messages", fail_messages)
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body())

    assert env["rec"].status == 500
    assert "/private/secret" not in env["rec"].error()
    assert "<path>" in env["rec"].error()
    assert not (env["sessions_dir"] / "sess-alpha-1.json").exists()


@pytest.mark.parametrize("replacement_sql", [
    "DROP TABLE messages",
    "DROP TABLE messages; CREATE TABLE messages (role TEXT, content TEXT)",
])
def test_missing_transcript_schema_fails_closed(resume_env, replacement_sql):
    db_path = _alpha_db(resume_env)
    _make_state_db(db_path, sid="sess-alpha-1")
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(replacement_sql)
        conn.commit()
    finally:
        conn.close()

    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body())

    assert env["rec"].status == 500
    assert not (env["sessions_dir"] / "sess-alpha-1.json").exists()


def test_bad_source_rejected(resume_env):
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1", source="telegram",
                   session_source="telegram")
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body())
    assert env["rec"].status == 403
    assert "not resumable" in env["rec"].error()


@pytest.mark.parametrize("source", ["cli", "tui", "acp", "desktop"])
def test_allowed_sources_pass_the_source_gate(resume_env, source):
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1", source=source,
                   session_source=source)
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body())
    assert env["rec"].status == 200, env["rec"].error()
    assert env["rec"].payload()["resumed"] is True


def test_active_unended_source_rejected(resume_env):
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1",
                   ended_at=None, end_reason=None)
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body())
    assert env["rec"].status == 409
    assert "appears active" in env["rec"].error()


# ---------------------------------------------------------------------------
# Gate: lineage
# ---------------------------------------------------------------------------


def test_lineage_tip_mismatch_rejected(resume_env):
    _make_state_db(_alpha_db(resume_env), sid="sess-alpha-1")
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(
        env["rec"], _body(root="sess-alpha-1", tip="some-other-tip"))
    assert env["rec"].status == 409
    assert "lineage" in env["rec"].error()


def test_lineage_root_mismatch_rejected(resume_env):
    _make_state_db(_alpha_db(resume_env), sid="sess-linear-tip",
                   parent_session_id="sess-linear-root")
    _make_state_db(_alpha_db(resume_env), sid="sess-linear-root",
                   ended_at=1700000000.0, end_reason="cli_close")
    env = resume_env
    # correct root/tip is (root, tip); send a bogus root.
    env["routes"]._handle_session_resume_in_webui(
        env["rec"],
        _body(sid="sess-linear-tip", root="wrong-root", tip="sess-linear-tip"),
    )
    assert env["rec"].status == 409


def test_lineage_continuation_happy_path(resume_env):
    db = _alpha_db(resume_env)
    _make_state_db(db, sid="sess-root", started_at=1700000000.0,
                   ended_at=1700000050.0, end_reason="compression")
    _make_state_db(db, sid="sess-tip", parent_session_id="sess-root",
                   started_at=1700000060.0, ended_at=1700000100.0,
                   end_reason="cli_close")
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(
        env["rec"], _body(sid="sess-tip", root="sess-root", tip="sess-tip"))
    assert env["rec"].status == 200, env["rec"].error()
    assert env["rec"].payload()["resumed"] is True


def test_obsolete_ancestor_with_continuation_is_rejected(resume_env):
    db = _alpha_db(resume_env)
    _make_state_db(db, sid="sess-root", started_at=1700000000.0,
                   ended_at=1700000050.0, end_reason="compression")
    _make_state_db(db, sid="sess-tip", parent_session_id="sess-root",
                   started_at=1700000060.0, ended_at=1700000100.0,
                   end_reason="cli_close")
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(
        env["rec"], _body(sid="sess-root", root="sess-root", tip="sess-root"))

    assert env["rec"].status == 409
    assert "current tip" in env["rec"].error()
    assert not (env["sessions_dir"] / "sess-root.json").exists()


# ---------------------------------------------------------------------------
# Happy path + source-db immutability
# ---------------------------------------------------------------------------


def test_happy_path_materialises_writable_sidecar_and_leaves_source_untouched(resume_env):
    sid = "sess-alpha-1"
    db = _alpha_db(resume_env)
    _make_state_db(db, sid=sid, source="cli", session_source="cli",
                   title="A finished CLI chat", model="m1", messages=4)

    before = _sha256(db)
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert env["rec"].status == 200, env["rec"].error()
    payload = env["rec"].payload()
    assert payload["ok"] is True
    assert payload["resumed"] is True
    assert payload["idempotent"] is False
    assert payload["session"]["session_id"] == sid
    assert payload["session"]["profile"] == "alpha"

    # Same sid, materialised as a writable sidecar bound to the profile with
    # the source metadata + messages preserved.
    sidecar = env["sessions_dir"] / f"{sid}.json"
    assert sidecar.exists()
    saved = env["models"].Session.load(sid)
    assert saved is not None
    assert saved.profile == "alpha"
    assert saved.read_only is False
    assert saved.is_cli_session is True
    assert saved.source_tag == "cli"
    assert saved.raw_source == "cli"
    assert saved.title == "A finished CLI chat"
    assert len(saved.messages) == 4
    assert saved.resume_source_profile == "alpha"
    assert saved.resume_source_state_db == str(db.resolve())
    assert saved.resume_lineage_root_id == sid
    assert saved.resume_lineage_tip_id == sid

    # The endpoint must not mutate the source state.db.
    assert _sha256(db) == before


def test_resume_source_snapshot_reuses_one_explicit_connection(resume_env, monkeypatch):
    env = resume_env
    sid = "snapshot-one-connection"
    db_path = _alpha_db(env)
    _make_state_db(db_path, sid=sid)

    routes = env["routes"]
    seen_connections = []
    real_row = routes._read_source_session_row
    real_lineage = routes.read_session_lineage_report
    real_messages = routes.get_state_db_session_messages

    def capture_row(*args, **kwargs):
        seen_connections.append(kwargs.get("connection"))
        return real_row(*args, **kwargs)

    def capture_lineage(*args, **kwargs):
        seen_connections.append(kwargs.get("connection"))
        return real_lineage(*args, **kwargs)

    def capture_messages(*args, **kwargs):
        seen_connections.append(kwargs.get("connection"))
        return real_messages(*args, **kwargs)

    monkeypatch.setattr(routes, "_read_source_session_row", capture_row)
    monkeypatch.setattr(routes, "read_session_lineage_report", capture_lineage)
    monkeypatch.setattr(routes, "get_state_db_session_messages", capture_messages)

    source_row, report, messages = routes._read_resume_source_snapshot(db_path, sid, "alpha")

    assert source_row["id"] == sid
    assert report["tip_session_id"] == sid
    assert messages
    assert seen_connections[0] is not None
    assert all(conn is seen_connections[0] for conn in seen_connections)


def test_source_change_before_publication_is_rejected(resume_env, monkeypatch):
    env = resume_env
    sid = "source-changed-before-publish"
    db_path = _alpha_db(env)
    _make_state_db(db_path, sid=sid)
    routes = env["routes"]
    real_snapshot = routes._read_resume_source_snapshot
    calls = 0

    def mutate_before_second_snapshot(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
                    (sid, "assistant", "late write", 1700000200.0),
                )
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                    (sid,),
                )
        return real_snapshot(*args, **kwargs)

    monkeypatch.setattr(routes, "_read_resume_source_snapshot", mutate_before_second_snapshot)
    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert env["rec"].status == 409
    assert "changed" in env["rec"].error()
    assert not (env["sessions_dir"] / f"{sid}.json").exists()


def test_failed_post_publish_verification_quarantines_new_sidecar(resume_env, monkeypatch):
    env = resume_env
    sid = "post-publish-verification-failure"
    _make_state_db(_alpha_db(env), sid=sid)
    routes = env["routes"]

    monkeypatch.setattr(routes, "_load_resume_sidecar_nonmutating", lambda _path: object())
    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    sidecar = env["sessions_dir"] / f"{sid}.json"
    quarantined = list((env["sessions_dir"] / ".resume-quarantine").glob(f"{sid}-*.json"))
    assert env["rec"].status == 500
    assert not sidecar.exists()
    assert len(quarantined) == 1


def test_happy_path_rejects_confirm_false_leaves_no_sidecar(resume_env):
    sid = "sess-alpha-1"
    _make_state_db(_alpha_db(resume_env), sid=sid)
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(sid=sid, confirm=False))
    assert env["rec"].status == 400
    assert not (env["sessions_dir"] / f"{sid}.json").exists()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotent_second_resume(resume_env):
    sid = "sess-alpha-1"
    db = _alpha_db(resume_env)
    _make_state_db(db, sid=sid, messages=3)
    env = resume_env
    routes = env["routes"]

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 200
    assert env["rec"].payload()["resumed"] is True

    sidecar = env["sessions_dir"] / f"{sid}.json"
    first_bytes = sidecar.read_bytes()
    db_before = _sha256(db)

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 200
    second = env["rec"].payload()
    assert second["resumed"] is False
    assert second["idempotent"] is True
    assert second["session"]["session_id"] == sid

    # A re-resume must not rewrite/duplicate the sidecar.
    assert sidecar.read_bytes() == first_bytes
    assert _sha256(db) == db_before


def test_controlled_post_resume_turn_continues_same_state_db_session_only(resume_env):
    """Exercise Hermes' production message persistence primitive after resume.

    The resume itself must leave the copied/synthetic state.db byte-identical.
    A subsequent controlled turn then appends two messages under the SAME source
    session id, creates no duplicate session row, and leaves a control session
    (its row and messages) byte-for-byte unchanged.
    """
    hermes_state = pytest.importorskip("hermes_state")
    env = resume_env
    db_path = _alpha_db(env)
    sid = "sess-controlled-resume"
    control_sid = "sess-control-untouched"

    db = hermes_state.SessionDB(db_path)
    try:
        db.create_session(sid, "cli", model="test-model", cwd="/tmp/ws")
        db.set_session_title(sid, "Controlled resume")
        db.append_message(sid, "user", "before resume")
        db.end_session(sid, "cli_close")

        db.create_session(control_sid, "cli", model="test-model", cwd="/tmp/ws")
        db.set_session_title(control_sid, "Control")
        db.append_message(control_sid, "user", "must remain unchanged")
        db.end_session(control_sid, "cli_close")
    finally:
        db.close()

    def snapshot_control():
        conn = sqlite3.connect(str(db_path))
        try:
            session_row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (control_sid,)
            ).fetchone()
            message_rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (control_sid,)
            ).fetchall()
            target_count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)
            ).fetchone()[0]
            return session_row, message_rows, target_count
        finally:
            conn.close()

    control_before, control_messages_before, target_before = snapshot_control()
    db_hash_before_resume = _sha256(db_path)

    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 200, env["rec"].error()
    assert env["rec"].payload()["session"]["session_id"] == sid
    assert _sha256(db_path) == db_hash_before_resume

    # Same primitive used by Hermes Agent for the resumed turn's persisted
    # user/assistant messages; there is deliberately no model/network call.
    db = hermes_state.SessionDB(db_path)
    try:
        db.append_message(sid, "user", "continued in WebUI")
        db.append_message(sid, "assistant", "controlled response")
    finally:
        db.close()

    control_after, control_messages_after, target_after = snapshot_control()
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ?", (sid,)
        ).fetchone()[0] == 1
        changed_session_ids = {
            row[0]
            for row in conn.execute(
                "SELECT session_id FROM messages GROUP BY session_id "
                "HAVING session_id = ? AND COUNT(*) = ?",
                (sid, target_before + 2),
            ).fetchall()
        }
    finally:
        conn.close()

    assert target_after == target_before + 2
    assert changed_session_ids == {sid}
    assert control_after == control_before
    assert control_messages_after == control_messages_before


# ---------------------------------------------------------------------------
# Flat sidecar collision (the WebUI store is not profile-qualified)
# ---------------------------------------------------------------------------


def test_cross_profile_same_sid_collision_refused(resume_env, monkeypatch):
    sid = "sess-shared-id"
    _make_state_db(_alpha_db(resume_env), sid=sid)
    env = resume_env
    # A sidecar for the SAME sid already exists, owned by another profile.
    other = env["models"].Session(session_id=sid, profile="beta", messages=[])
    other.save(touch_updated_at=False)
    sidecar = env["sessions_dir"] / f"{sid}.json"
    before = _sha256(sidecar)

    def mutating_load_must_not_run(*_args, **_kwargs):
        raise AssertionError("resume collision check called mutating Session.load")

    monkeypatch.setattr(env["models"].Session, "load", mutating_load_must_not_run)
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 409
    assert "already owns" in env["rec"].error()

    # The foreign sidecar is byte-identical; collision inspection is pure read.
    assert _sha256(sidecar) == before


def test_competing_sidecar_writer_cannot_overwrite_resumed_owner(resume_env):
    sid = "sess-competing-owner"
    _make_state_db(_alpha_db(resume_env), sid=sid)
    env = resume_env
    routes = env["routes"]
    sidecar = env["sessions_dir"] / f"{sid}.json"

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 200, env["rec"].error()
    resumed_bytes = sidecar.read_bytes()

    competing = env["models"].Session(
        session_id=sid,
        profile="beta",
        title="Competing owner",
        messages=[],
    )
    with pytest.raises(PermissionError, match="owned by profile"):
        competing.save(touch_updated_at=False)

    assert sidecar.read_bytes() == resumed_bytes
    persisted = json.loads(sidecar.read_text(encoding="utf-8"))
    assert persisted["profile"] == "alpha"
    assert persisted["resume_source_profile"] == "alpha"


def test_blank_profile_sidecar_collision_refused(resume_env):
    sid = "sess-blank-owner"
    _make_state_db(_alpha_db(resume_env), sid=sid)
    env = resume_env
    blank = env["models"].Session(session_id=sid, profile=None, messages=[])
    blank.save(touch_updated_at=False)

    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 409
    assert "already owns" in env["rec"].error()


def test_same_profile_unmarked_sidecar_is_not_treated_as_idempotent(resume_env):
    sid = "sess-existing-unmarked"
    _make_state_db(_alpha_db(resume_env), sid=sid)
    env = resume_env
    existing = env["models"].Session(session_id=sid, profile="alpha", messages=[])
    existing.save(touch_updated_at=False)

    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert env["rec"].status == 409
    assert "resume identity" in env["rec"].error()


# ---------------------------------------------------------------------------
# Regression: ordinary import / projection stays read-only
# ---------------------------------------------------------------------------


def test_existing_readonly_projection_regression(resume_env, monkeypatch):
    """With the operator read-only flag set, the ordinary claim path must still
    refuse to materialise a writable sidecar (i.e. resume_in_webui did not
    loosen the default projection)."""
    monkeypatch.setenv("HERMES_WEBUI_EXTERNAL_STATE_READ_ONLY", "1")
    sid = "sess-readonly-projection"
    _make_state_db(_alpha_db(resume_env), sid=sid, source="cli",
                   session_source="cli", messages=3)
    routes = resume_env["routes"]

    session, reason = routes._claim_or_synthesize_cli_session(sid)
    assert reason == "not_claimable"
    assert session is not None
    assert session.read_only is True
    # No writable sidecar may have been created by the projection.
    assert not (resume_env["sessions_dir"] / f"{sid}.json").exists()


def test_resume_flag_gate_independent_of_readonly_projection(resume_env, monkeypatch):
    """The resume endpoint works regardless of the read-only projection flag,
    but only through its explicit operator allowlist."""
    monkeypatch.setenv("HERMES_WEBUI_EXTERNAL_STATE_READ_ONLY", "1")
    sid = "sess-explicit-resume"
    _make_state_db(_alpha_db(resume_env), sid=sid, messages=2)
    env = resume_env
    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 200, env["rec"].error()
    assert env["rec"].payload()["resumed"] is True


# ---------------------------------------------------------------------------
# Route wiring (the endpoint is actually reachable through handle_post)
# ---------------------------------------------------------------------------


class _DispatchHandler:
    def __init__(self, path):
        self.status = None
        self.headers = {"Content-Type": "application/json", "Content-Length": "1"}
        self.rfile = io.BytesIO(b"")
        self.wfile = io.BytesIO()
        self.command = "POST"
        self.path = path
        self.client_address = ("127.0.0.1", 12345)

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers[key] = value

    def end_headers(self):
        pass


def test_endpoint_reachable_through_handle_post(resume_env, monkeypatch):
    sid = "sess-dispatch-1"
    _make_state_db(_alpha_db(resume_env), sid=sid)
    env = resume_env
    monkeypatch.setattr(env["routes"], "_check_csrf", lambda _h: True)
    monkeypatch.setattr(env["routes"], "read_body", lambda _h: _body(sid=sid))

    path = "/api/session/resume_in_webui"
    env["routes"].handle_post(_DispatchHandler(path), urlparse(path))

    assert env["rec"].status == 200, env["rec"].error()
    assert env["rec"].payload()["resumed"] is True
    assert (env["sessions_dir"] / f"{sid}.json").exists()


# ---------------------------------------------------------------------------
# Atomic first publication (#765 follow-up: no writer lock, exclusive claim)
# ---------------------------------------------------------------------------


def _install_thread_recorders(routes, monkeypatch):
    """Route ``j``/``bad`` to the *calling thread's* recorder.

    The shared ``resume_env`` recorder cannot observe two concurrent requests,
    so concurrency tests give each worker its own ``_Recorder`` via a
    thread-local. The worker must assign ``local.rec`` before calling.
    """
    local = threading.local()

    def fake_bad(_handler, msg, status=400):
        local.rec.bad = (msg, status)
        return None

    def fake_j(_handler, payload, status=200, extra_headers=None, *, pretty=True):
        local.rec.j = (payload, status)
        return None

    monkeypatch.setattr(routes, "bad", fake_bad)
    monkeypatch.setattr(routes, "j", fake_j)
    return local


def test_first_publication_claims_the_sidecar_with_an_exclusive_link(resume_env, monkeypatch):
    """First publication must be an exclusive claim, never a clobbering rename."""
    sid = "sess-exclusive-claim"
    _make_state_db(_alpha_db(resume_env), sid=sid, messages=3)
    env = resume_env
    models = env["models"]
    sidecar = env["sessions_dir"] / f"{sid}.json"

    links = []
    replaces = []
    real_link = models.os.link
    real_replace = models.os.replace

    def recording_link(src, dst):
        links.append((str(src), str(dst)))
        return real_link(src, dst)

    def recording_replace(src, dst):
        replaces.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(models.os, "link", recording_link)
    monkeypatch.setattr(models.os, "replace", recording_replace)

    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert env["rec"].status == 200, env["rec"].error()
    # Exactly one claim, from the private staging namespace onto the canonical id.
    assert [dst for _src, dst in links] == [str(sidecar)]
    staging_src = Path(links[0][0])
    assert staging_src.parent == env["sessions_dir"] / ".resume-staging"
    assert staging_src.name.startswith(f"{sid}.")
    # The canonical path is never *created* or clobbered by a rename: that
    # would be the check-then-replace race the exclusive claim replaces. F2/F3
    # add exactly one later in-place rewrite onto the canonical: the explicit
    # verified-marker commit, which happens only after the claim succeeded and
    # final source re-verification passed (never a foreign writer's rename).
    canonical_replaces = [(src, dst) for src, dst in replaces if dst == str(sidecar)]
    assert len(canonical_replaces) <= 1, canonical_replaces
    for src, _dst in canonical_replaces:
        assert Path(src).parent == env["sessions_dir"], src
    # The staged artifact is consumed by the claim, not left behind.
    assert not list((env["sessions_dir"] / ".resume-staging").glob("*.json"))
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["session_id"] == sid
    assert len(data["messages"]) == 3
    assert data["resume_source_profile"] == "alpha"


def test_readers_never_observe_a_partial_canonical_sidecar(resume_env, monkeypatch):
    """The canonical id stays invisible until the verified payload lands."""
    sid = "sess-atomic-visibility"
    _make_state_db(_alpha_db(resume_env), sid=sid, messages=3)
    env = resume_env
    models = env["models"]
    routes = env["routes"]
    sidecar = env["sessions_dir"] / f"{sid}.json"

    entered = threading.Event()
    release = threading.Event()
    observations = []
    real_link = models.os.link

    def gated_link(src, dst):
        observations.append(
            {
                "canonical_visible_before_claim": Path(dst).exists(),
                "staged_payload": json.loads(Path(src).read_text(encoding="utf-8")),
            }
        )
        entered.set()
        assert release.wait(timeout=5)
        return real_link(src, dst)

    monkeypatch.setattr(models.os, "link", gated_link)

    worker = threading.Thread(
        target=routes._handle_session_resume_in_webui,
        args=(env["rec"], _body(sid=sid)),
        daemon=True,
    )
    worker.start()
    assert entered.wait(timeout=5)
    try:
        # Parked at the claim: a concurrent reader still sees no writable sidecar
        # at the canonical path (the staged file lives only in the private
        # namespace), so a partially published artifact can never be observed.
        assert not sidecar.exists()
        assert models.Session.load(sid) is None
        assert observations[0]["canonical_visible_before_claim"] is False
        # The payload that is about to become visible is already complete.
        assert observations[0]["staged_payload"]["session_id"] == sid
        assert len(observations[0]["staged_payload"]["messages"]) == 3
    finally:
        release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert env["rec"].status == 200, env["rec"].error()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["session_id"] == sid
    assert len(data["messages"]) == 3


def test_concurrent_resume_requests_publish_exactly_one_sidecar(resume_env, monkeypatch):
    """Two simultaneous Resumes must publish once and stay free of writer locks."""
    sid = "sess-concurrent-resume"
    _make_state_db(_alpha_db(resume_env), sid=sid, messages=3)
    env = resume_env
    routes = env["routes"]
    sidecar = env["sessions_dir"] / f"{sid}.json"

    barrier = threading.Barrier(2)
    real_stage = routes.stage_session_sidecar

    def gated_stage(session, staging_path):
        staged = real_stage(session, staging_path)
        # Both requests reach the publication boundary together, so the
        # exclusive claim is the only thing that can pick the winner.
        barrier.wait(timeout=5)
        return staged

    monkeypatch.setattr(routes, "stage_session_sidecar", gated_stage)
    local = _install_thread_recorders(routes, monkeypatch)

    results = {}
    errors = []

    def worker(name):
        local.rec = _Recorder()
        try:
            routes._handle_session_resume_in_webui(local.rec, _body(sid=sid))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        results[name] = local.rec

    t1 = threading.Thread(target=worker, args=("a",), daemon=True)
    t2 = threading.Thread(target=worker, args=("b",), daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not t1.is_alive() and not t2.is_alive()
    assert not errors, errors
    assert sorted(rec.status for rec in results.values()) == [200, 200], [
        rec.error() for rec in results.values()
    ]
    payloads = [rec.payload() for rec in results.values()]
    # Exactly one request publishes; the other reconciles against the winner's
    # identical identity as an idempotent success instead of failing.
    assert sum(1 for payload in payloads if payload["resumed"] is True) == 1
    assert sum(1 for payload in payloads if payload["idempotent"] is True) == 1

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["session_id"] == sid
    assert len(data["messages"]) == 3
    assert not list((env["sessions_dir"] / ".resume-staging").glob("*.json"))
    quarantine = env["sessions_dir"] / ".resume-quarantine"
    assert not quarantine.exists() or not list(quarantine.glob("*.json"))


def test_parked_resume_publication_does_not_stall_unrelated_saves(resume_env, monkeypatch):
    """A Resume mid-claim must not block saves of other conversations (#765 F2)."""
    sid = "sess-parked-claim"
    _make_state_db(_alpha_db(resume_env), sid=sid, messages=3)
    env = resume_env
    models = env["models"]
    routes = env["routes"]

    entered = threading.Event()
    release = threading.Event()
    real_link = models.os.link

    def gated_link(src, dst):
        entered.set()
        assert release.wait(timeout=5)
        return real_link(src, dst)

    monkeypatch.setattr(models.os, "link", gated_link)

    worker = threading.Thread(
        target=routes._handle_session_resume_in_webui,
        args=(env["rec"], _body(sid=sid)),
        daemon=True,
    )
    worker.start()
    assert entered.wait(timeout=5)

    unrelated = models.Session(
        session_id="sess-unrelated-save",
        title="Unrelated",
        profile="alpha",
        messages=[{"role": "user", "content": "hello"}],
    )
    replaced = threading.Event()
    real_replace = models.os.replace

    def recording_replace(src, dst):
        if str(dst).endswith("sess-unrelated-save.json"):
            replaced.set()
        return real_replace(src, dst)

    monkeypatch.setattr(models.os, "replace", recording_replace)
    saver = threading.Thread(target=unrelated.save, kwargs={"skip_index": True}, daemon=True)
    saver.start()
    try:
        # The unrelated save must complete while the Resume claim is parked:
        # nothing on the Resume path may serialize unrelated writers.
        assert replaced.wait(timeout=5), (
            "an unrelated session save was blocked by a parked Resume publication"
        )
    finally:
        release.set()
    saver.join(timeout=5)
    worker.join(timeout=5)

    assert not saver.is_alive()
    assert env["rec"].status == 200, env["rec"].error()


def test_publish_refuses_to_clobber_an_occupied_id(resume_env):
    """An occupied canonical id is never overwritten by a lost claim."""
    sid = "sess-occupied-claim"
    _make_state_db(_alpha_db(resume_env), sid=sid)
    env = resume_env
    models = env["models"]
    sidecar = env["sessions_dir"] / f"{sid}.json"
    sidecar.write_text('{"session_id": "occupied"}\n', encoding="utf-8")

    candidate = models.import_cli_session(
        sid,
        "staged candidate",
        [{"role": "user", "content": "hello"}],
        "test-model",
        profile="alpha",
        persist=False,
    )
    staging = env["sessions_dir"] / ".resume-staging" / f"{sid}.stage.json"
    staged = models.stage_session_sidecar(candidate, staging)

    with pytest.raises(FileExistsError):
        models.publish_staged_session_sidecar(staged, staging)

    assert sidecar.read_text(encoding="utf-8") == '{"session_id": "occupied"}\n'
    # A lost claim leaves the staging file for the caller to clean up; nothing
    # was published and nothing was destroyed.
    assert staging.exists()


# ---------------------------------------------------------------------------
# D1 — first-ownership gate: a pre-claim ordinary save can never clobber a
# claimed Resume sidecar (audit probe: ``ordinary-save-race``), while ordinary
# same-id saves still stay lock-free against each other (#765).
# ---------------------------------------------------------------------------


def test_ordinary_save_in_flight_makes_resume_refuse_without_clobbering(resume_env, monkeypatch):
    """A save admitted before the claim must not be silently overwritten.

    Reproduces the audit's ``ordinary-save-race`` probe: an ordinary same-id
    save is parked immediately before ``os.replace`` when Resume arrives. The
    gate must fail closed (409) while a writer is active so that a Resume can
    never return 200 and then have its published sidecar overwritten.
    """
    sid = "sess-pre-claimed-writer"
    _make_state_db(_alpha_db(resume_env), sid=sid, messages=3)
    env = resume_env
    models = env["models"]
    routes = env["routes"]
    sidecar = env["sessions_dir"] / f"{sid}.json"

    parked = threading.Event()
    release = threading.Event()
    real_replace = models._safe_replace

    def gated_replace(src, dst):
        if str(dst).endswith(f"{sid}.json"):
            parked.set()
            assert release.wait(timeout=5)
        return real_replace(src, dst)

    monkeypatch.setattr(models, "_safe_replace", gated_replace)

    writer = models.Session(
        session_id=sid,
        title="Ordinary writer",
        profile="alpha",
        messages=[{"role": "user", "content": "writer"}],
    )
    writer_errors = []

    def run_writer():
        try:
            writer.save(skip_index=True)
        except Exception as exc:  # pragma: no cover - asserted below
            writer_errors.append(exc)

    saver = threading.Thread(target=run_writer, daemon=True)
    saver.start()
    assert parked.wait(timeout=5), "the ordinary save never reached os.replace"

    # Resume arrives while the writer is in flight: it must refuse, never
    # publish a sidecar that a stale writer can then overwrite.
    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 409, env["rec"].error()
    assert "in flight" in env["rec"].error()
    assert not sidecar.exists()

    release.set()
    saver.join(timeout=5)

    assert not saver.is_alive()
    assert not writer_errors, writer_errors
    published = json.loads(sidecar.read_text(encoding="utf-8"))
    assert published["session_id"] == sid
    assert published["profile"] == "alpha"
    # Nothing claims a resume identity for this sidecar, and no gate state leaks.
    assert not published.get("resume_source_profile")


def test_claim_gate_state_is_released_after_success(resume_env):
    """The gate must not leak per-session state once both sides finish."""
    sid = "sess-gate-release"
    _make_state_db(_alpha_db(resume_env), sid=sid, messages=2)
    env = resume_env
    models = env["models"]

    env["routes"]._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 200, env["rec"].error()

    assert sid not in models._SESSION_CLAIM_STATE
    # A later ordinary save of the claimed sidecar is admitted again.
    saved = models.Session.load(sid)
    assert saved is not None
    saved.messages.append({"role": "user", "content": "after resume"})
    saved.save(skip_index=True)
    assert sid not in models._SESSION_CLAIM_STATE


def test_active_resume_claim_blocks_a_new_same_id_save(resume_env, monkeypatch):
    """While a Resume claim is held, a new same-id save must refuse to write."""
    sid = "sess-claim-blocks-save"
    _make_state_db(_alpha_db(resume_env), sid=sid, messages=3)
    env = resume_env
    models = env["models"]
    routes = env["routes"]
    sidecar = env["sessions_dir"] / f"{sid}.json"

    entered = threading.Event()
    release = threading.Event()
    real_link = models.os.link

    def gated_link(src, dst):
        entered.set()
        assert release.wait(timeout=5)
        return real_link(src, dst)

    monkeypatch.setattr(models.os, "link", gated_link)

    worker = threading.Thread(
        target=routes._handle_session_resume_in_webui,
        args=(env["rec"], _body(sid=sid)),
        daemon=True,
    )
    worker.start()
    assert entered.wait(timeout=5)

    competing = models.Session(
        session_id=sid,
        title="late save",
        profile="alpha",
        messages=[{"role": "user", "content": "late"}],
    )
    with pytest.raises(PermissionError):
        competing.save(skip_index=True)

    release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert env["rec"].status == 200, env["rec"].error()
    published = json.loads(sidecar.read_text(encoding="utf-8"))
    assert published["resume_source_profile"] == "alpha"
    assert len(published["messages"]) == 3
    assert sid not in models._SESSION_CLAIM_STATE


def test_concurrent_same_id_ordinary_saves_still_reach_replace_in_parallel(resume_env, monkeypatch):
    """#765: the gate is a count, so two same-id saves still race in parallel."""
    sid = "sess-lock-free-pair"
    env = resume_env
    models = env["models"]

    barrier = threading.Barrier(2)
    both_replaced = []
    real_replace = models._safe_replace

    def gated_replace(src, dst):
        if str(dst).endswith(f"{sid}.json"):
            both_replaced.append(threading.get_ident())
            barrier.wait(timeout=5)
        return real_replace(src, dst)

    monkeypatch.setattr(models, "_safe_replace", gated_replace)

    errors = []

    def run_writer(n):
        session = models.Session(
            session_id=sid,
            title=f"writer {n}",
            profile="alpha",
            messages=[{"role": "user", "content": f"m{n}"}],
        )
        try:
            session.save(skip_index=True)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=run_writer, args=(n,), daemon=True) for n in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    # Both writers reached os.replace and rendezvoused there: the gate never
    # serialized them.
    assert len(both_replaced) == 2
    assert sid not in models._SESSION_CLAIM_STATE


# ---------------------------------------------------------------------------
# D2/D3/D4 — fail-closed post-publish verification, quarantine and cache/index
# eviction (audit probes: ``late-source-change``, ``unreadable-publication``,
# ``cached-quarantine``).
# ---------------------------------------------------------------------------


def test_source_change_after_final_snapshot_is_quarantined_fail_closed(resume_env, monkeypatch):
    """A source that changes during publication must not yield a 200 + stale sidecar."""
    env = resume_env
    sid = "late-source-change"
    db_path = _alpha_db(env)
    _make_state_db(db_path, sid=sid, messages=3)
    routes = env["routes"]
    sidecar = env["sessions_dir"] / f"{sid}.json"
    real_stage = routes.stage_session_sidecar
    mutated = []

    def mutate_during_stage(session, staging_path):
        staged = real_stage(session, staging_path)
        if not mutated:
            mutated.append(True)
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
                    (sid, "assistant", "late write", 1700000200.0),
                )
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                    (sid,),
                )
        return staged

    monkeypatch.setattr(routes, "stage_session_sidecar", mutate_during_stage)
    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert mutated, "the source mutation never ran"
    assert env["rec"].status == 500, env["rec"].error()
    assert not sidecar.exists()
    quarantined = list((env["sessions_dir"] / ".resume-quarantine").glob(f"{sid}-*.json"))
    assert len(quarantined) == 1
    assert sid not in (env["sessions_dir"] / "_index.json").read_text(encoding="utf-8")


def test_unreadable_published_sidecar_is_quarantined_and_deindexed(resume_env, monkeypatch):
    """A just-published sidecar that cannot be parsed must not stay live/indexed."""
    env = resume_env
    sid = "unreadable-publication"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    routes = env["routes"]
    sidecar = env["sessions_dir"] / f"{sid}.json"
    real_publish = routes.publish_staged_session_sidecar

    def publish_then_corrupt(session, staging_path):
        real_publish(session, staging_path)
        sidecar.write_text("{not valid json", encoding="utf-8")

    monkeypatch.setattr(routes, "publish_staged_session_sidecar", publish_then_corrupt)
    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert env["rec"].status == 500, env["rec"].error()
    assert not sidecar.exists()
    quarantined = list((env["sessions_dir"] / ".resume-quarantine").glob(f"{sid}-*.json"))
    assert len(quarantined) == 1
    index_text = (env["sessions_dir"] / "_index.json").read_text(encoding="utf-8")
    assert sid not in index_text


def test_quarantine_evicts_cached_writable_sidecar(resume_env, monkeypatch):
    """F3: a provisional publication is never exposed writable, even mid-publish.

    The pre-fix D5 probe asserted a *writable cached object* existed during the
    publication window and that quarantine evicted it. That assertion encoded
    the very F3 defect: a delayed reader could adopt an unverified publication
    and, if the quarantine move failed, keep writing through it. The invariant
    is now stronger and checked at the worst possible moment (inside publish):
    no reader may resolve a provisional publication, nothing is cached, and the
    quarantine denial is durable.
    """
    env = resume_env
    sid = "cached-quarantine"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    routes = env["routes"]
    models = env["models"]
    sidecar = env["sessions_dir"] / f"{sid}.json"
    real_publish = routes.publish_staged_session_sidecar
    probe = {}

    def publish_then_probe(session, staging_path):
        real_publish(session, staging_path)
        # The canonical is now PROVISIONAL: any reader must fail closed.
        try:
            models.get_session(sid)
            probe["writable"] = True
        except KeyError:
            probe["writable"] = False
        probe["cached"] = sid in models.SESSIONS
        probe["state"] = json.loads(sidecar.read_text(encoding="utf-8"))["resume_publication_state"]

    monkeypatch.setattr(routes, "publish_staged_session_sidecar", publish_then_probe)
    monkeypatch.setattr(routes, "_load_resume_sidecar_nonmutating", lambda _path: object())
    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert probe.get("state") == "provisional"
    assert probe.get("writable") is False, "provisional publication was served writable"
    assert probe.get("cached") is False, "provisional publication was cached as writable"
    assert env["rec"].status == 500, env["rec"].error()
    assert not sidecar.exists()
    # The stale writable object is gone from the cache and cannot be re-resolved.
    assert sid not in models.SESSIONS
    assert models.Session.load(sid) is None
    # Denial is durable on disk, so even a fresh process resolution fails closed.
    assert models.is_resume_publication_denied(sid)
    assert sid not in (env["sessions_dir"] / "_index.json").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# D4 (identity) — an idempotent re-resume must match the sidecar's own
# ``session_id``, not just its resume identity (audit probe: ``sid-identity``).
# ---------------------------------------------------------------------------


def test_idempotent_resume_rejects_sidecar_with_foreign_session_id(resume_env):
    """A hand-edited sidecar naming another session is never an idempotent match."""
    env = resume_env
    sid = "sid-identity-mismatch"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    routes = env["routes"]
    sidecar = env["sessions_dir"] / f"{sid}.json"

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 200, env["rec"].error()

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["session_id"] = "some-other-session"
    sidecar.write_text(json.dumps(data), encoding="utf-8")

    # The fixture's recorder is captured by the patched j()/bad(); reset it so
    # only the second request's outcome is observed.
    env["rec"].bad = None
    env["rec"].j = None
    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert env["rec"].status == 409, env["rec"].error()
    assert env["rec"].bad is not None
    assert "identity" in (env["rec"].error() or "")
    # The foreign payload is left untouched, not silently adopted.
    assert json.loads(sidecar.read_text(encoding="utf-8"))["session_id"] == "some-other-session"


# ---------------------------------------------------------------------------
# F1–F4 (Astra re-audit 94bfe8af) — explicit verified ownership and durable
# quarantine fencing.  Each test mirrors one independent probe schedule:
# index-error-with-source-change, quarantine-move-failure,
# unreadable-canonical-with-publish-error, second-Resume-while-provisional and
# delayed-cache-fill.
# ---------------------------------------------------------------------------


def _write_provisional_canonical(env, sid, *, messages=3):
    """Leave the exact on-disk state a first Resume holds before committing.

    Reproduces the PROVISIONAL canonical (and nothing else) without running the
    handler, which would keep the cross-process resume claim for the duration.
    """
    models = env["models"]
    session = models.Session(
        session_id=sid,
        profile="alpha",
        messages=[{"role": "user", "content": f"m{i}"} for i in range(messages)],
        read_only=False,
    )
    session.resume_source_profile = "alpha"
    session.resume_source_state_db = str(_alpha_db(env))
    session.resume_lineage_root_id = sid
    session.resume_lineage_tip_id = sid
    session.resume_publication_state = models.RESUME_PUBLICATION_PROVISIONAL
    session.save()
    return env["sessions_dir"] / f"{sid}.json"


def _mutate_source(db: Path, sid: str):
    """Mutate the source row after publication (the D2 "source moved" race)."""
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
            (sid,),
        )
        conn.commit()
    finally:
        conn.close()


def test_provisional_marker_precedes_verified_commit(resume_env, monkeypatch):
    """F2/F3: the canonical is PROVISIONAL until an explicit verified commit.

    Pins the two-phase ownership boundary: a reader observing the canonical
    before ``mark_resume_publication_verified`` must see ``provisional``, and
    only the explicit commit flips it to ``verified``.
    """
    env = resume_env
    sid = "phase-boundary"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    routes = env["routes"]
    models = env["models"]
    sidecar = env["sessions_dir"] / f"{sid}.json"
    seen = []

    real_mark = models.mark_resume_publication_verified

    def observe_then_mark(session):
        seen.append(json.loads(sidecar.read_text(encoding="utf-8"))["resume_publication_state"])
        return real_mark(session)

    monkeypatch.setattr(models, "mark_resume_publication_verified", observe_then_mark)
    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert env["rec"].status == 200, env["rec"].error()
    assert seen == ["provisional"]
    final = json.loads(sidecar.read_text(encoding="utf-8"))
    assert final["resume_publication_state"] == "verified"
    # Only a verified, non-denied publication is resolvable as writable.
    resolved = models.get_session(sid)
    assert resolved is not None
    assert models.session_publication_admissible(resolved)
    assert not models.is_resume_publication_denied(sid)


def test_second_resume_sees_provisional_publication_and_fails_closed(resume_env, monkeypatch):
    """F2: a competing Resume must never return idempotent success on a
    provisional first publication; it waits briefly then fails closed."""
    env = resume_env
    sid = "competing-provisional"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    routes = env["routes"]
    models = env["models"]
    sidecar = _write_provisional_canonical(env, sid)

    # A provisional publication is invisible to every reader.
    with pytest.raises(KeyError):
        models.get_session(sid)

    monkeypatch.setattr(routes, "_await_resume_publication_commit", lambda *a, **k: None)

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 409, env["rec"].error()
    assert env["rec"].bad is not None
    assert "still being verified" in (env["rec"].error() or "")
    # Never adopted as an idempotent re-resume, and the artifact is untouched.
    assert sidecar.exists()
    assert json.loads(sidecar.read_text(encoding="utf-8"))["resume_publication_state"] == "provisional"


def test_second_resume_waits_for_a_committed_first_publication(resume_env, monkeypatch):
    """F2: a committed (verified) first publication is still an idempotent match."""
    env = resume_env
    sid = "competing-committed"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    routes = env["routes"]
    models = env["models"]
    _write_provisional_canonical(env, sid)
    # Commit the verified marker directly (as the winning publisher would).
    session = models.Session.load(sid)
    models.mark_resume_publication_verified(session)

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))
    assert env["rec"].status == 200, env["rec"].error()
    assert json.loads((env["sessions_dir"] / f"{sid}.json").read_text(encoding="utf-8"))["resume_publication_state"] == "verified"


def test_parked_first_publication_fails_competing_resume_closed(resume_env, monkeypatch):
    """F2 (probe ``unverified-idempotent``): while a first publication is parked
    mid-window, a competing Resume must not report success and must not get a
    writable object."""
    env = resume_env
    sid = "unverified-idempotent"
    db = _alpha_db(env)
    _make_state_db(db, sid=sid, messages=3)
    routes = env["routes"]
    models = env["models"]

    real_publish = routes.publish_staged_session_sidecar
    released = threading.Event()
    parked = threading.Event()

    def parked_publish(session, staging_path):
        real_publish(session, staging_path)
        parked.set()
        released.wait(5.0)

    monkeypatch.setattr(routes, "publish_staged_session_sidecar", parked_publish)

    # Thread-safe outcome capture: each call's own handler object keys its result.
    outcomes = {}

    def dispatch_bad(handler, msg, status=400):
        outcomes[id(handler)] = ("bad", status, msg)
        return None

    def dispatch_j(handler, payload, status=200, extra_headers=None, *, pretty=True):
        outcomes[id(handler)] = ("ok", status, payload)
        return None

    monkeypatch.setattr(routes, "bad", dispatch_bad)
    monkeypatch.setattr(routes, "j", dispatch_j)

    first_handler = object()

    def first_call():
        routes._handle_session_resume_in_webui(first_handler, _body(sid=sid))

    thread = threading.Thread(target=first_call)
    thread.start()
    assert parked.wait(5.0), "first publication never reached the parked window"

    # The first publication is provisional and invisible; the competing Resume
    # must fail closed rather than adopt it, even after the source moves.
    _mutate_source(db, sid)
    second_handler = object()
    routes._handle_session_resume_in_webui(second_handler, _body(sid=sid))
    second = outcomes.get(id(second_handler))
    assert second is not None
    assert second[0] == "bad", second
    assert second[1] != 200, second
    assert second[1] in (409, 500), second

    released.set()
    thread.join(10.0)
    assert not thread.is_alive()
    # The first publication's source moved, so it must end non-200 with no
    # writable canonical left behind.
    first = outcomes.get(id(first_handler))
    assert first is not None and first[0] == "bad", first
    assert first[1] == 500, first
    assert not (env["sessions_dir"] / f"{sid}.json").exists()
    assert models.is_resume_publication_denied(sid)
    with pytest.raises(KeyError):
        models.get_session(sid)


def test_source_change_during_index_write_after_verified_commit_is_safe_degradation(resume_env, monkeypatch):
    """F1 (probe ``index-error-source-change``): the post-publish source re-read
    runs BEFORE the index write, so a source that moves in the index window is
    already covered by the verified commit. This schedule injects NO index
    exception (the real index writer runs and succeeds) — the name says exactly
    what it proves: a post-verification source change observed during the index
    write degrades safely to the committed verified artifact.

    The real index-exception schedules live in
    ``test_real_index_exception_after_verified_commit_keeps_verified_publication``
    and ``test_marker_loss_during_index_failure_is_not_adopted``.
    """
    env = resume_env
    sid = "index-error-source-change"
    db = _alpha_db(env)
    _make_state_db(db, sid=sid, messages=3)
    routes = env["routes"]
    models = env["models"]
    sidecar = env["sessions_dir"] / f"{sid}.json"
    snapshots = []

    real_load = routes._load_resume_sidecar_nonmutating
    real_index = models._write_session_index

    def counting_load(path):
        loaded = real_load(path)
        if loaded is not None:
            snapshots.append(len(getattr(loaded, "messages", []) or []))
        return loaded

    def index_then_mutate(updates=None):
        # The source moves exactly in the index window, after the canonical was
        # linked and read back. Only the post-commit index call carries updates.
        if updates:
            _mutate_source(db, sid)
        return real_index(updates=updates)

    monkeypatch.setattr(routes, "_load_resume_sidecar_nonmutating", counting_load)
    monkeypatch.setattr(models, "_write_session_index", index_then_mutate)

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    # The live re-read happened (3 messages observed from the published
    # canonical) — the source change was caught after publication.
    assert 3 in snapshots
    assert snapshots.count(3) >= 1
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["resume_publication_state"] == "verified"
    assert env["rec"].status == 200, env["rec"].error()


def test_quarantine_move_failure_still_denies_and_deindexes(resume_env, monkeypatch):
    """F1/F3 (probe ``quarantine-move-failure``): final-verification failure is
    terminal even when the quarantine move itself fails."""
    env = resume_env
    sid = "quarantine-move-failure"
    db = _alpha_db(env)
    _make_state_db(db, sid=sid, messages=3)
    routes = env["routes"]
    models = env["models"]
    sidecar = env["sessions_dir"] / f"{sid}.json"

    # Snapshot #1 proves consistency, #2 is the pre-publish boundary re-read,
    # #3 is the POST-publish final re-read: failing #3 is the F1 terminal case.
    real_snapshot = routes._read_resume_source_snapshot
    calls = {"n": 0}

    def flaky_snapshot(*a, **k):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise RuntimeError("source moved during publication")
        return real_snapshot(*a, **k)

    monkeypatch.setattr(routes, "_read_resume_source_snapshot", flaky_snapshot)

    real_replace = routes.os.replace

    def fail_quarantine_move(src, dst):
        if str(Path(dst).parent).endswith(".resume-quarantine"):
            raise OSError("quarantine move failed")
        return real_replace(src, dst)

    monkeypatch.setattr(routes.os, "replace", fail_quarantine_move)

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert calls["n"] >= 2, "the post-publish source re-read never ran"
    assert env["rec"].status == 500, env["rec"].error()
    # The denial tombstone is written BEFORE the (failing) move, so the
    # publication is still inaccessible even though the file remains on disk.
    assert models.is_resume_publication_denied(sid)
    assert sidecar.exists(), "the move was supposed to fail; file should remain"
    with pytest.raises(KeyError):
        models.get_session(sid)
    assert sid not in models.SESSIONS
    loaded = models.Session.load(sid)
    assert loaded is not None and not models.session_publication_admissible(loaded)
    assert sid not in (env["sessions_dir"] / "_index.json").read_text(encoding="utf-8")


def test_unreadable_canonical_with_publish_error_is_quarantined(resume_env, monkeypatch):
    """F1/F3 (probe ``unreadable-with-publish-error``): a post-link error that
    leaves an unreadable canonical must make it inaccessible, never adopt it."""
    env = resume_env
    sid = "resume-quarantine"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    routes = env["routes"]
    models = env["models"]
    sidecar = env["sessions_dir"] / f"{sid}.json"

    def index_corrupt_then_raise(updates=None):
        # Corrupt the canonical in the index window, then fail: this is the
        # schedule where the post-link error leaves an unreadable canonical.
        if sidecar.exists():
            sidecar.write_text("{ not json", encoding="utf-8")
        raise OSError("index write failed after publication")

    monkeypatch.setattr(models, "_write_session_index", index_corrupt_then_raise)

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert env["rec"].status == 500, env["rec"].error()
    assert models.is_resume_publication_denied(sid)
    # The unreadable canonical is gone from the live namespace.
    assert not sidecar.exists()
    assert not list(env["sessions_dir"].glob(f"{sid}.json"))
    with pytest.raises(KeyError):
        models.get_session(sid)
    assert models.Session.load(sid) is None
    assert sid not in (env["sessions_dir"] / "_index.json").read_text(encoding="utf-8")
    # The unreadable canonical was moved out of the live namespace (best
    # effort) and the denial tombstone keeps it inaccessible regardless.
    assert list((env["sessions_dir"] / ".resume-quarantine").glob(f"{sid}*.json"))


def test_provisional_read_during_first_publication_is_rejected(resume_env, monkeypatch):
    """F3 (probe ``cached``): a reader that looks at the canonical while the
    first publication is still PROVISIONAL is refused, and cannot restore a
    writable object or index entry after the eventual quarantine.

    The reader here is synchronous: it runs inside the publish wrapper, i.e. at
    the exact instant the provisional canonical is live, and it gets no
    writable object. (The delayed/barrier schedules that exercise a reader
    parked across a quarantine are
    ``test_reader_parked_after_affirmative_check_cannot_admit_after_rejection``
    and ``test_post_rejection_save_cannot_recreate_a_quarantined_canonical``.)
    """
    env = resume_env
    sid = "cached"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    routes = env["routes"]
    models = env["models"]

    real_publish = routes.publish_staged_session_sidecar
    reader_out = {}

    def publish_then_stall(session, staging_path):
        real_publish(session, staging_path)
        # A delayed reader loads the provisional canonical and is preempted.
        try:
            models.get_session(sid)
            reader_out["adopted"] = True
        except KeyError:
            reader_out["adopted"] = False

    monkeypatch.setattr(routes, "publish_staged_session_sidecar", publish_then_stall)
    monkeypatch.setattr(routes, "_load_resume_sidecar_nonmutating", lambda _path: object())

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert reader_out.get("adopted") is False
    assert env["rec"].status == 500, env["rec"].error()
    # After quarantine the reader still cannot rebuild a writable object.
    with pytest.raises(KeyError):
        models.get_session(sid)
    assert sid not in models.SESSIONS
    assert models.Session.load(sid) is None
    assert models.is_resume_publication_denied(sid)
    models.SESSIONS.clear()
    assert models.all_sessions() == [] or sid not in [
        s.get("session_id") for s in models.all_sessions()
    ]
    assert sid not in (env["sessions_dir"] / "_index.json").read_text(encoding="utf-8")


def test_ordinary_save_concurrency_is_not_serialized(resume_env, monkeypatch):
    """#765 guard: the ordinary save path must not serialize on Resume fencing.

    Two distinct sessions saving concurrently must both complete while an
    unrelated Resume publication is parked inside its (claim-held) window.
    """
    env = resume_env
    routes = env["routes"]
    models = env["models"]
    parked_sid = "parked-resume"
    _make_state_db(_alpha_db(env), sid=parked_sid, messages=3)

    real_publish = routes.publish_staged_session_sidecar
    released = threading.Event()
    parked = threading.Event()

    def parked_publish(session, staging_path):
        real_publish(session, staging_path)
        parked.set()
        released.wait(5.0)

    monkeypatch.setattr(routes, "publish_staged_session_sidecar", parked_publish)

    def first_call():
        routes._handle_session_resume_in_webui(object(), _body(sid=parked_sid))

    thread = threading.Thread(target=first_call)
    thread.start()
    assert parked.wait(5.0)

    barrier = threading.Barrier(2, timeout=10)
    done = []

    def save_session(sid):
        session = models.Session(session_id=sid, profile="alpha", messages=[{"role": "user", "content": sid}])
        barrier.wait()
        session.save()
        done.append(sid)

    savers = [threading.Thread(target=save_session, args=(f"concurrent-{i}",)) for i in range(2)]
    for t in savers:
        t.start()
    for t in savers:
        t.join(10.0)
    assert all(not t.is_alive() for t in savers)

    released.set()
    thread.join(10.0)
    assert sorted(done) == ["concurrent-0", "concurrent-1"], done
    for sid in done:
        assert (env["sessions_dir"] / f"{sid}.json").exists()


# ---------------------------------------------------------------------------
# A1-A4 (Astra 4f0ad5d8): adversarial regressions
#
# A1  recovery of the CURRENT attempt's publication must require an explicit
#     ``resume_publication_state == 'verified'`` marker; a missing, empty,
#     unknown or provisional marker is never proof.
# A2  admission/cache insertion and denial/eviction are atomic, and a tracked
#     Resume sidecar is revalidated at persistence, so a reader parked after an
#     affirmative admissibility check (or one holding an already-admitted
#     object) can neither cache nor save a revoked publication.
# A3  denial wins for marker-absent legacy Resume sidecars in direct lookup,
#     the cache, the fallback scan, the full index rebuild and incremental
#     index updates.
# A4  real index-exception and after-affirmative-check barriers, including a
#     real post-rejection save attempt.
# ---------------------------------------------------------------------------


def _isolate_resume_index(monkeypatch, models, env) -> Path:
    """Point the sidebar index at the test session dir for this test only."""
    index = env["sessions_dir"] / "_index.json"
    index.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", index)
    return index


def _make_tracked_resume_sidecar(env, sid, *, marker, messages=3, profile="alpha"):
    """Write a Resume sidecar directly, with an explicit publication marker.

    ``marker=None`` downgrades it to a pre-fix *legacy* Resume sidecar: it keeps
    every ownership field but carries no publication marker at all.
    """
    models = env["models"]
    session = models.Session(
        session_id=sid,
        profile=profile,
        messages=[{"role": "user", "content": f"m{i}"} for i in range(messages)],
        read_only=False,
    )
    session.resume_source_profile = profile
    session.resume_source_state_db = str(_alpha_db(env))
    session.resume_lineage_root_id = sid
    session.resume_lineage_tip_id = sid
    if marker is not None:
        session.resume_publication_state = marker
    session.save()
    return session, env["sessions_dir"] / f"{sid}.json"


def _strip_marker_from_disk(sidecar: Path) -> None:
    """Remove the publication marker from a canonical without re-saving it."""
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data.pop("resume_publication_state", None)
    sidecar.write_text(json.dumps(data), encoding="utf-8")


# ------------------------------- A1 ----------------------------------------


@pytest.mark.parametrize("marker", [None, "", "provisional", "unexpected"])
def test_post_link_recovery_requires_an_explicit_verified_marker(
    resume_env, monkeypatch, marker
):
    """A1: recovery over a post-link error never adopts an unverified marker.

    The publication attempt owns the canonical (it linked it itself), so the
    recovery path must demand ``resume_publication_state == 'verified'``. A
    missing, empty, provisional or unknown marker is rejected and the artifact
    is made inaccessible - the legacy marker-absent migration policy lives on a
    different, existing-sidecar-only path and must not leak in here.
    """
    env = resume_env
    models = env["models"]
    routes = env["routes"]
    tag = marker if marker else "absent"
    sid = f"a1-recovery-{tag}"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    sidecar = env["sessions_dir"] / f"{sid}.json"
    _isolate_resume_index(monkeypatch, models, env)

    real_publish = routes.publish_staged_session_sidecar

    def publish_then_break(session, staging_path):
        real_publish(session, staging_path)
        # The canonical is now linked and owned by THIS attempt. Simulate the
        # marker never reaching disk (or reaching it incorrectly) and a
        # post-link persistence error inside the same publication window.
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if marker is None:
            data.pop("resume_publication_state", None)
        else:
            data["resume_publication_state"] = marker
        sidecar.write_text(json.dumps(data), encoding="utf-8")
        raise OSError("post-link persist failure")

    monkeypatch.setattr(routes, "publish_staged_session_sidecar", publish_then_break)

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert env["rec"].status == 500, (tag, env["rec"].error())
    assert models.is_resume_publication_denied(sid)
    assert not sidecar.exists(), f"{tag}: rejected publication stayed live"
    with pytest.raises(KeyError):
        models.get_session(sid)
    assert sid not in models.SESSIONS


def test_marker_loss_during_index_failure_is_not_adopted(resume_env, monkeypatch):
    """A1: a verified marker lost during the index window is not proof.

    The verified commit succeeded, then the index write failed AND the on-disk
    marker was gone by the time the post-index integrity re-read ran. The
    attempt must not fall back to the legacy "no marker means committed"
    compatibility rule for a publication it owns: it fails closed, denies the
    id, and quarantines the artifact out of the live namespace.
    """
    env = resume_env
    models = env["models"]
    routes = env["routes"]
    sid = "a1b-marker-loss"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    sidecar = env["sessions_dir"] / f"{sid}.json"
    _isolate_resume_index(monkeypatch, models, env)

    real_index = models._write_session_index

    def index_strip_marker_and_raise(updates=None, **kwargs):
        if updates:
            _strip_marker_from_disk(sidecar)
            raise OSError("index write failed after verified commit")
        return real_index(updates=updates, **kwargs)

    monkeypatch.setattr(models, "_write_session_index", index_strip_marker_and_raise)

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert env["rec"].status == 500, env["rec"].error()
    assert models.is_resume_publication_denied(sid)
    assert not sidecar.exists(), "the marker-lost artifact stayed in the live namespace"
    with pytest.raises(KeyError):
        models.get_session(sid)
    assert models.Session.load(sid) is None
    assert sid not in models.SESSIONS


def test_real_index_exception_after_verified_commit_keeps_verified_publication(
    resume_env, monkeypatch
):
    """A4/A1 positive control: a REAL index exception over an intact verified
    artifact is the F1 safe-degradation path, and it still succeeds.

    This is the schedule the misleadingly-named older test claimed to cover but
    did not (it injected no index exception at all).
    """
    env = resume_env
    models = env["models"]
    routes = env["routes"]
    sid = "a4-real-index-exception"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    sidecar = env["sessions_dir"] / f"{sid}.json"
    _isolate_resume_index(monkeypatch, models, env)

    real_index = models._write_session_index
    raised = {"n": 0}

    def index_raise_once(updates=None, **kwargs):
        if updates:
            raised["n"] += 1
            raise OSError("index write failed after verified commit")
        return real_index(updates=updates, **kwargs)

    monkeypatch.setattr(models, "_write_session_index", index_raise_once)

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert raised["n"] == 1, "the real index exception was never injected"
    assert env["rec"].status == 200, env["rec"].error()
    assert not models.is_resume_publication_denied(sid)
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["resume_publication_state"] == "verified"
    models.SESSIONS.clear()
    assert models.get_session(sid) is not None


# ------------------------------- A2 ----------------------------------------


def test_reader_parked_after_affirmative_check_cannot_admit_after_rejection(
    resume_env, monkeypatch
):
    """A2/A4: the after-affirmative-check barrier.

    A reader is parked INSIDE the admissibility predicate, immediately after it
    has returned True for the live canonical. The attempt is then rejected and
    quarantined. When the reader resumes, the revocation generation recheck must
    refuse admission: no writable object is returned and the cache is not
    repopulated. (Before the fix the predicate ran outside the cache-insertion
    critical section, so the parked reader rewrote ``SESSIONS`` after the
    quarantine.)
    """
    env = resume_env
    models = env["models"]
    routes = env["routes"]
    sid = "a2-affirmative-barrier"
    _make_state_db(_alpha_db(env), sid=sid, messages=3)
    sidecar = env["sessions_dir"] / f"{sid}.json"
    _isolate_resume_index(monkeypatch, models, env)

    entered = threading.Event()
    release = threading.Event()
    reader = {}
    real_admissible = models.session_publication_admissible

    def parking_admissible(session, **kwargs):
        result = real_admissible(session, **kwargs)
        if (
            result
            and str(getattr(session, "session_id", "") or "") == sid
            and not entered.is_set()
        ):
            entered.set()
            release.wait(10.0)  # park AFTER the affirmative check
        return result

    monkeypatch.setattr(models, "session_publication_admissible", parking_admissible)

    def reader_body():
        try:
            obj = models.get_session(sid)
            reader["object"] = obj
            reader["writable"] = not bool(getattr(obj, "read_only", False))
        except KeyError:
            reader["refused"] = True
        except Exception as exc:  # pragma: no cover - diagnostic
            reader["error"] = repr(exc)

    real_index = models._write_session_index

    def index_start_reader_strip_and_raise(updates=None, **kwargs):
        if updates:
            thread = threading.Thread(target=reader_body, daemon=True)
            thread.start()
            assert entered.wait(10.0), "reader never reached the affirmative check"
            reader["thread"] = thread
            _strip_marker_from_disk(sidecar)
            raise OSError("index write failed after verified commit")
        return real_index(updates=updates, **kwargs)

    monkeypatch.setattr(models, "_write_session_index", index_start_reader_strip_and_raise)

    routes._handle_session_resume_in_webui(env["rec"], _body(sid=sid))

    assert env["rec"].status == 500, env["rec"].error()
    assert models.is_resume_publication_denied(sid)

    release.set()
    reader["thread"].join(10.0)
    assert not reader["thread"].is_alive()
    assert "error" not in reader, reader.get("error")
    assert reader.get("refused") is True, reader
    assert "object" not in reader, "a revoked publication was served writable"
    assert sid not in models.SESSIONS, "a revoked publication was re-cached"


def test_post_rejection_save_cannot_recreate_a_quarantined_canonical(
    resume_env, monkeypatch
):
    """A2/A4: a real post-rejection save attempt on an already-admitted object.

    The reader legitimately admitted and cached the verified publication BEFORE
    the rejection. After the denial and quarantine the canonical is gone, but
    the caller still holds the writable object. Its real ``save()`` must be
    refused, and the canonical must NOT reappear. (Before the fix the save
    recreated it: quarantine only evicted the cache and wrote a tombstone, and
    the save path never consulted either.)
    """
    env = resume_env
    models = env["models"]
    sid = "a2-post-rejection-save"
    sidecar = env["sessions_dir"] / f"{sid}.json"
    _isolate_resume_index(monkeypatch, models, env)

    _make_tracked_resume_sidecar(env, sid, marker=models.RESUME_PUBLICATION_VERIFIED)
    assert sidecar.exists()

    models.SESSIONS.clear()
    admitted = models.get_session(sid)
    assert admitted is not None and not bool(getattr(admitted, "read_only", False))
    assert sid in models.SESSIONS

    # Deny + evict (atomic in the fixed implementation), then simulate the
    # best-effort move out of the live namespace succeeding.
    models.mark_resume_publication_denied(sid, reason="post-rejection save probe")
    models.evict_session_from_cache(sid)
    assert models.is_resume_publication_denied(sid)
    sidecar.unlink()
    assert not sidecar.exists()

    with pytest.raises(PermissionError):
        admitted.save(skip_index=True)

    assert not sidecar.exists(), "a post-rejection save recreated the canonical"
    assert sid not in models.SESSIONS
    with pytest.raises(KeyError):
        models.get_session(sid)


# ------------------------------- A3 ----------------------------------------


def test_denied_legacy_resume_sidecar_fails_closed_in_lookup_cache_and_scan(
    resume_env, monkeypatch
):
    """A3: denial wins for a marker-absent legacy Resume sidecar.

    The sidecar keeps every resume ownership field but no publication marker
    (a pre-fix artifact), its canonical stays on disk (the quarantine move
    failed), and it IS denied. Direct lookup, the cache and the fallback scan
    must all refuse it.
    """
    env = resume_env
    models = env["models"]
    sid = "a3-legacy-denied"
    _isolate_resume_index(monkeypatch, models, env)
    _, sidecar = _make_tracked_resume_sidecar(env, sid, marker=None)

    legacy = models.Session.load(sid)
    assert legacy is not None
    # It really is a pre-fix legacy Resume sidecar: ownership fields present,
    # publication marker absent.
    assert models.resume_publication_state(legacy) is None
    assert getattr(legacy, "resume_source_state_db", None)
    assert getattr(legacy, "resume_lineage_root_id", None) == sid
    # Control: a legacy sidecar with no denial is still admissible (the
    # supported pre-fix migration policy must not regress).
    assert models.session_publication_admissible(legacy)

    models.mark_resume_publication_denied(sid, reason="legacy denial probe")
    assert models.is_resume_publication_denied(sid)
    assert sidecar.exists(), "the move was supposed to fail; the file stays"

    models.SESSIONS.clear()
    with pytest.raises(KeyError):
        models.get_session(sid)
    assert sid not in models.SESSIONS

    loaded = models.Session.load(sid)
    assert loaded is not None and not models.session_publication_admissible(loaded)

    models.SESSIONS.clear()
    rows = models.all_sessions()
    assert sid not in [row.get("session_id") for row in rows]
    assert not models.SESSIONS


def test_denied_legacy_resume_sidecar_is_dropped_by_full_index_rebuild(
    resume_env, monkeypatch
):
    """A3: a full ``_write_session_index()`` rebuild drops the denied legacy row
    while keeping an un-denied legacy sibling (so the fix is not a blanket
    "hide every Resume sidecar")."""
    env = resume_env
    models = env["models"]
    denied_sid = "a3-rebuild-denied"
    control_sid = "a3-rebuild-control"
    index = _isolate_resume_index(monkeypatch, models, env)
    _make_tracked_resume_sidecar(env, denied_sid, marker=None)
    _make_tracked_resume_sidecar(env, control_sid, marker=None)

    models.mark_resume_publication_denied(denied_sid, reason="rebuild probe")
    models.SESSIONS.clear()

    models._write_session_index()

    rows = json.loads(index.read_text(encoding="utf-8"))
    ids = [row.get("session_id") for row in rows]
    assert denied_sid not in ids, "the full rebuild re-indexed a denied legacy row"
    assert control_sid in ids, "an un-denied legacy sidecar was wrongly dropped"


def test_denied_legacy_resume_sidecar_is_dropped_by_incremental_index_update(
    resume_env, monkeypatch
):
    """A3: an incremental ``_write_session_index(updates=...)`` drops a denied
    legacy row that is already in the index."""
    env = resume_env
    models = env["models"]
    denied_sid = "a3-incremental-denied"
    index = _isolate_resume_index(monkeypatch, models, env)
    _, sidecar = _make_tracked_resume_sidecar(env, denied_sid, marker=None)
    models.mark_resume_publication_denied(denied_sid, reason="incremental probe")

    # Seed the index exactly as a pre-fix deployment would have left it.
    stale_row = models.Session.load(denied_sid).compact()
    index.write_text(json.dumps([stale_row]), encoding="utf-8")

    control = models.Session(
        session_id="a3-incremental-control",
        profile="alpha",
        messages=[{"role": "user", "content": "control"}],
    )
    models._write_session_index(updates=[control])

    rows = json.loads(index.read_text(encoding="utf-8"))
    ids = [row.get("session_id") for row in rows]
    assert denied_sid not in ids, "the incremental update kept a denied legacy row"
    assert "a3-incremental-control" in ids
    assert sidecar.exists()


def test_all_sessions_index_path_drops_a_denied_legacy_resume_sidecar(
    resume_env, monkeypatch
):
    """A3: the sidebar index fast path must deny the legacy sid, not just a
    marked one, when its canonical is still on disk."""
    env = resume_env
    models = env["models"]
    denied_sid = "a3-indexpath-denied"
    control_sid = "a3-indexpath-control"
    index = _isolate_resume_index(monkeypatch, models, env)
    _make_tracked_resume_sidecar(env, denied_sid, marker=None)
    _make_tracked_resume_sidecar(env, control_sid, marker=None)
    models.mark_resume_publication_denied(denied_sid, reason="index-path probe")

    # A valid index (both rows present) so all_sessions() takes the fast path.
    rows = [
        models.Session.load(denied_sid).compact(),
        models.Session.load(control_sid).compact(),
    ]
    index.write_text(json.dumps(rows), encoding="utf-8")

    models.SESSIONS.clear()
    listed = [row.get("session_id") for row in models.all_sessions()]
    assert denied_sid not in listed
    assert control_sid in listed


def test_denial_snapshot_failure_fails_closed_for_tracked_resume_identity(
    resume_env, monkeypatch
):
    """A3: an unavailable denial *snapshot* is "unknown", never "no denials".

    ``_denied_resume_publication_ids`` returns None when the denial directory
    cannot be listed. The per-id tombstone check must then be used instead of
    affirmative absence, for a marker-absent legacy Resume identity as much as
    for a marked one - direct lookup, enumeration and the index rebuild.
    """
    env = resume_env
    models = env["models"]
    denied_sid = "a3-snapshot-denied"
    control_sid = "a3-snapshot-control"
    index = _isolate_resume_index(monkeypatch, models, env)
    _make_tracked_resume_sidecar(env, denied_sid, marker=None)
    _make_tracked_resume_sidecar(env, control_sid, marker=None)
    models.mark_resume_publication_denied(denied_sid, reason="snapshot probe")

    monkeypatch.setattr(models, "_denied_resume_publication_ids", lambda *a, **k: None)

    models.SESSIONS.clear()
    with pytest.raises(KeyError):
        models.get_session(denied_sid)
    assert not models.session_publication_admissible(models.Session.load(denied_sid))

    listed = [row.get("session_id") for row in models.all_sessions()]
    assert denied_sid not in listed
    assert control_sid in listed, "an un-denied legacy sidecar was wrongly dropped"

    models._write_session_index()
    ids = [row.get("session_id") for row in json.loads(index.read_text(encoding="utf-8"))]
    assert denied_sid not in ids

    # Ordinary (non-Resume) sessions are unaffected by an unavailable snapshot.
    plain = models.Session(session_id="a3-plain", profile="alpha")
    assert models.session_publication_admissible(plain)
