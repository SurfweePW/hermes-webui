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
