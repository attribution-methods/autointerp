"""Tests for the model/API-key selection flow (autointerp_agent.model_select).

The prompt_toolkit UI itself needs a TTY; everything around it (provider
detection, live-catalog fetching/filtering, env persistence, validation, the
orchestrated flow, REPL command dispatch) is exercised here with the UI and
network primitives monkeypatched.
"""

from __future__ import annotations

import asyncio
import io
import os
import stat
from pathlib import Path

from rich.console import Console

from autointerp_agent import model_select as ms
from autointerp_agent.config import AgentConfig

_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
)


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False)


def _clear_keys(monkeypatch) -> None:
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def _seq_select(responses):
    """Fake _select_async asserting menu order: [(title_substring, value), ...]."""
    calls: list[str] = []

    async def fake(title, options, default_index=0):
        assert len(calls) < len(responses), f"unexpected extra menu: {title}"
        expect, value = responses[len(calls)]
        assert expect in title, f"expected menu {expect!r}, got {title!r}"
        calls.append(title)
        return value

    return fake


# ---------------------------------------------------------------------------
# provider detection
# ---------------------------------------------------------------------------


def test_provider_for_model_prefixes() -> None:
    assert ms.provider_for_model("anthropic/claude-sonnet-4-5").key == "anthropic"
    assert ms.provider_for_model("openai/gpt-5.1").key == "openai"
    assert ms.provider_for_model("openrouter/anthropic/claude-sonnet-4.5").key == "openrouter"


def test_provider_for_model_bare_ids() -> None:
    assert ms.provider_for_model("claude-haiku-4-5").key == "anthropic"
    assert ms.provider_for_model("gpt-4o").key == "openai"


def test_provider_for_model_unknown_is_none() -> None:
    assert ms.provider_for_model("ollama/llama3") is None
    assert ms.provider_for_model("") is None


def test_missing_key_env(monkeypatch) -> None:
    _clear_keys(monkeypatch)
    assert ms.missing_key_env("anthropic/claude-sonnet-4-5") == "ANTHROPIC_API_KEY"
    # Any of the provider's env vars satisfies the requirement.
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    assert ms.missing_key_env("anthropic/claude-sonnet-4-5") is None
    # Unknown providers are not gated.
    assert ms.missing_key_env("ollama/llama3") is None


# ---------------------------------------------------------------------------
# live model catalogs
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


def test_is_openai_chat_id_filter() -> None:
    assert ms._is_openai_chat_id("gpt-5.1")
    assert ms._is_openai_chat_id("o3-mini")
    assert ms._is_openai_chat_id("chatgpt-4o-latest")
    for noise in (
        "whisper-1", "text-embedding-3-large", "gpt-4o-audio-preview",
        "dall-e-3", "omni-moderation-latest", "gpt-5.1-codex", "tts-1",
        "gpt-4o-realtime-preview", "davinci-002",
    ):
        assert not ms._is_openai_chat_id(noise), noise


def test_dedupe_dated_prefers_alias() -> None:
    ids = ["gpt-9", "gpt-9-2026-05-01", "o7-2026-04-16", "claude-x-20260101"]
    # Dated snapshots drop only when their undated alias is present.
    assert ms._dedupe_dated(ids) == ["gpt-9", "o7-2026-04-16", "claude-x-20260101"]


def test_fetch_anthropic_models_sorts_and_prefixes(monkeypatch) -> None:
    import httpx

    payload = {
        "data": [
            {"id": "claude-b", "display_name": "Claude B",
             "created_at": "2025-01-01T00:00:00Z"},
            {"id": "claude-a", "display_name": "Claude A",
             "created_at": "2026-01-01T00:00:00Z"},
        ]
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(payload))
    models = ms._fetch_anthropic_models("sk-x")
    assert models == [
        ("anthropic/claude-a", "Claude A"),
        ("anthropic/claude-b", "Claude B"),
    ]


def test_fetch_openai_models_filters_sorts_dedupes(monkeypatch) -> None:
    import httpx

    payload = {
        "data": [
            {"id": "whisper-1", "created": 999},
            {"id": "gpt-9-2026-05-01", "created": 500},
            {"id": "gpt-9", "created": 400},
            {"id": "o7", "created": 300},
            {"id": "text-embedding-3-large", "created": 998},
        ]
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(payload))
    models = ms._fetch_openai_models("sk-x")
    # Non-chat ids gone; dated snapshot collapsed into its alias; created desc.
    assert models == [("openai/gpt-9", ""), ("openai/o7", "")]


def test_fetch_openrouter_models_families_tilde_and_tool_support(monkeypatch) -> None:
    import httpx

    payload = {
        "data": [
            {"id": "weirdvendor/foo", "name": "Weird", "created": 900,
             "supported_parameters": ["tools"]},
            {"id": "~anthropic/claude-zeta-latest", "name": "Zeta", "created": 800,
             "supported_parameters": ["tools", "tool_choice"]},
            {"id": "anthropic/claude-x", "name": "X", "created": 700,
             "supported_parameters": ["tools"]},
            # The agent needs tool calling — image/classifier models are hidden.
            {"id": "openai/gpt-image-9", "name": "Img", "created": 950,
             "supported_parameters": ["max_tokens"]},
            {"id": "anthropic/claude-no-params", "name": "NP", "created": 940},
        ]
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(payload))
    models = ms._fetch_openrouter_models(None)
    assert models == [
        ("openrouter/~anthropic/claude-zeta-latest", "Zeta"),
        ("openrouter/anthropic/claude-x", "X"),
    ]


def test_fetch_with_fallback_uses_registry_then_builtin(monkeypatch) -> None:
    provider = ms.PROVIDERS["anthropic"]

    def boom(_provider):
        raise RuntimeError("offline")

    monkeypatch.setattr(ms, "_fetch_live_models", boom)
    monkeypatch.setattr(
        ms, "_registry_models", lambda p: [("anthropic/claude-from-registry", "")]
    )
    models, source = asyncio.run(ms.fetch_models_with_fallback(provider))
    assert source == "litellm registry"
    assert models == [("anthropic/claude-from-registry", "")]

    monkeypatch.setattr(ms, "_registry_models", lambda p: [])
    models, source = asyncio.run(ms.fetch_models_with_fallback(provider))
    assert source == "built-in fallback"
    assert models  # repo defaults survive as last resort
    assert all(ms.provider_for_model(mid).key == "anthropic" for mid, _ in models)


def test_annotate_marks_current_and_default() -> None:
    out = ms._annotate(
        [("anthropic/claude-sonnet-4-5", "Sonnet"), ("openai/gpt-9", "")],
        current="openai/gpt-9",
    )
    assert "(repo default)" in out[0][1]
    assert "(current)" in out[1][1]


# ---------------------------------------------------------------------------
# .env persistence
# ---------------------------------------------------------------------------


def test_persist_creates_env_file_with_restrictive_perms(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    try:
        ms.persist_to_env_file({"AUTOINTERP_TEST_VAR": "abc"}, env_path)
        assert env_path.read_text() == "AUTOINTERP_TEST_VAR=abc\n"
        mode = stat.S_IMODE(env_path.stat().st_mode)
        assert mode == 0o600
        assert os.environ["AUTOINTERP_TEST_VAR"] == "abc"
    finally:
        os.environ.pop("AUTOINTERP_TEST_VAR", None)


def test_persist_replaces_in_place_and_preserves_other_lines(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("# comment\nOTHER=1\nAUTOINTERP_TEST_VAR=old\n")
    env_path.chmod(0o664)  # e.g. hand-created with a loose umask
    try:
        ms.persist_to_env_file(
            {"AUTOINTERP_TEST_VAR": "new", "AUTOINTERP_MODEL": "openai/gpt-5.1"},
            env_path,
        )
        lines = env_path.read_text().splitlines()
        assert lines == [
            "# comment",
            "OTHER=1",
            "AUTOINTERP_TEST_VAR=new",
            "AUTOINTERP_MODEL=openai/gpt-5.1",
        ]
        # Perms are re-tightened on update, not only on creation.
        assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    finally:
        os.environ.pop("AUTOINTERP_TEST_VAR", None)
        os.environ.pop("AUTOINTERP_MODEL", None)


def test_env_file_is_gitignored(tmp_path: Path) -> None:
    assert not ms.env_file_is_gitignored(tmp_path)
    (tmp_path / ".gitignore").write_text("__pycache__/\n.env\n")
    assert ms.env_file_is_gitignored(tmp_path)


def test_repo_gitignore_covers_env_file() -> None:
    # The setup flow offers to write API keys to .env; that file must never
    # be committable from the repo root.
    assert ms.env_file_is_gitignored(Path(__file__).resolve().parents[1])


# ---------------------------------------------------------------------------
# validation + error formatting
# ---------------------------------------------------------------------------


class _FakeAuthError(Exception):
    pass


def test_format_llm_error_first_line_and_hint() -> None:
    exc = _FakeAuthError(
        "_FakeAuthError: Missing Anthropic API Key - details\nlong traceback line"
    )
    msg = ms.format_llm_error(exc, model_name="anthropic/claude-sonnet-4-5")
    assert "traceback" not in msg
    assert "_FakeAuthError" in msg
    assert "[anthropic/claude-sonnet-4-5]" in msg
    # Auth-flavored errors point the user at /model; others don't.
    assert "/model" in msg
    other = ms.format_llm_error(ValueError("boom"))
    assert "/model" not in other


def test_validate_model_success_and_failure(monkeypatch) -> None:
    import litellm

    seen: dict = {}

    async def ok(**kwargs):
        seen.update(kwargs)
        return {"choices": []}

    monkeypatch.setattr(litellm, "acompletion", ok)
    assert asyncio.run(ms.validate_model("anthropic/claude-sonnet-4-5")) == (True, "")
    # The ping must exercise tool calling — the agent sends tools on every
    # real call, so a tools-incapable model has to fail at setup time.
    assert seen.get("tools") == [ms._PING_TOOL]
    assert seen.get("tool_choice") == "auto"

    async def boom(**kwargs):
        raise _FakeAuthError("bad key")

    monkeypatch.setattr(litellm, "acompletion", boom)
    ok_flag, err = asyncio.run(ms.validate_model("anthropic/claude-sonnet-4-5"))
    assert not ok_flag
    assert "bad key" in err


class BadRequestError(Exception):
    """Name-matched stand-in for litellm.BadRequestError."""


def test_validate_model_reasoning_budget_error_is_benign(monkeypatch) -> None:
    # gpt-5-nano spent the whole ping budget on reasoning tokens — the
    # request authenticated and the model ran, so the key is valid.
    import litellm

    async def reasoning_budget(**kwargs):
        raise BadRequestError(
            "OpenAIException - Could not finish the message because max_tokens "
            "or model output limit was reached. Please try again with higher "
            "max_tokens."
        )

    monkeypatch.setattr(litellm, "acompletion", reasoning_budget)
    assert asyncio.run(ms.validate_model("openai/gpt-5-nano")) == (True, "")

    # A BadRequest that is NOT about token budget is still a real failure.
    async def bad_model(**kwargs):
        raise BadRequestError("OpenAIException - The model `gpt-99` does not exist")

    monkeypatch.setattr(litellm, "acompletion", bad_model)
    ok_flag, err = asyncio.run(ms.validate_model("openai/gpt-99"))
    assert not ok_flag and "gpt-99" in err


# ---------------------------------------------------------------------------
# ensure_model_ready gating
# ---------------------------------------------------------------------------


def test_ensure_ready_passes_through_when_key_present(monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    out = asyncio.run(ms.ensure_model_ready(config, _console(), interactive=True))
    assert out is config


def test_ensure_ready_noninteractive_missing_key_returns_none(monkeypatch) -> None:
    _clear_keys(monkeypatch)
    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    console = _console()
    out = asyncio.run(ms.ensure_model_ready(config, console, interactive=False))
    assert out is None
    text = console.file.getvalue()
    assert "ANTHROPIC_API_KEY" in text


def test_ensure_ready_interactive_without_tty_degrades_to_instructions(monkeypatch) -> None:
    # Under pytest stdin is not a TTY, so even interactive=True must not try
    # to launch the prompt_toolkit UI.
    _clear_keys(monkeypatch)
    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    out = asyncio.run(ms.ensure_model_ready(config, _console(), interactive=True))
    assert out is None


# ---------------------------------------------------------------------------
# run_setup_flow (UI + network primitives monkeypatched)
# ---------------------------------------------------------------------------


def _wire_happy_flow(monkeypatch, *, save_choice: int, key: str = "sk-test-123") -> None:
    async def fake_select_provider(current):
        return ms.PROVIDERS["anthropic"]

    async def fake_prompt_text(message, password=False):
        assert password
        return key

    async def fake_fetch(provider):
        return [("anthropic/claude-sonnet-4-5", "Claude Sonnet 4.5")], "live"

    async def fake_validate(model):
        return True, ""

    monkeypatch.setattr(ms, "select_provider", fake_select_provider)
    monkeypatch.setattr(ms, "_prompt_text", fake_prompt_text)
    monkeypatch.setattr(ms, "fetch_models_with_fallback", fake_fetch)
    monkeypatch.setattr(ms, "validate_model", fake_validate)
    monkeypatch.setattr(
        ms, "_select_async", _seq_select([("Select a model", 0), ("Save", save_choice)])
    )


def test_setup_flow_saves_user_level_credentials(monkeypatch, tmp_path: Path) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.chdir(tmp_path)
    creds = tmp_path / "userhome" / "credentials"
    monkeypatch.setattr(ms, "USER_ENV_PATH", creds)
    _wire_happy_flow(monkeypatch, save_choice=0)  # "Yes — for this user"

    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    try:
        out = asyncio.run(ms.run_setup_flow(config, _console()))
        assert out is not None and out.model_name == "anthropic/claude-sonnet-4-5"
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-123"
        text = creds.read_text()
        assert "ANTHROPIC_API_KEY=sk-test-123" in text
        assert "AUTOINTERP_MODEL=anthropic/claude-sonnet-4-5" in text
        assert not (tmp_path / ".env").exists()  # project file untouched
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("AUTOINTERP_MODEL", None)


def test_setup_flow_saves_project_env(monkeypatch, tmp_path: Path) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text(".env\n")
    monkeypatch.setattr(ms, "USER_ENV_PATH", tmp_path / "userhome" / "credentials")
    _wire_happy_flow(monkeypatch, save_choice=1)  # "Yes — for this project only"

    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    try:
        out = asyncio.run(ms.run_setup_flow(config, _console()))
        assert out is not None
        env_text = (tmp_path / ".env").read_text()
        assert "ANTHROPIC_API_KEY=sk-test-123" in env_text
        assert not (tmp_path / "userhome" / "credentials").exists()
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("AUTOINTERP_MODEL", None)


def test_setup_flow_validation_failure_quit_pops_bad_key(monkeypatch) -> None:
    _clear_keys(monkeypatch)

    async def fake_select_provider(current):
        return ms.PROVIDERS["anthropic"]

    async def fake_prompt_text(message, password=False):
        return "sk-bad"

    async def fake_fetch(provider):
        return [("anthropic/claude-x", "")], "live"

    async def fake_validate(model):
        return False, "AuthenticationError: bad key"

    monkeypatch.setattr(ms, "select_provider", fake_select_provider)
    monkeypatch.setattr(ms, "_prompt_text", fake_prompt_text)
    monkeypatch.setattr(ms, "fetch_models_with_fallback", fake_fetch)
    monkeypatch.setattr(ms, "validate_model", fake_validate)
    monkeypatch.setattr(
        ms,
        "_select_async",
        _seq_select([("Select a model", 0), ("Validation failed", 2)]),
    )

    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    out = asyncio.run(ms.run_setup_flow(config, _console()))
    assert out is None
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_setup_flow_cancel_at_provider_picker_returns_none(monkeypatch) -> None:
    _clear_keys(monkeypatch)

    async def fake_select_provider(current):
        return None

    monkeypatch.setattr(ms, "select_provider", fake_select_provider)
    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    assert asyncio.run(ms.run_setup_flow(config, _console())) is None


def test_setup_flow_skips_key_prompt_when_env_present(monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    async def fake_select_provider(current):
        return ms.PROVIDERS["openai"]

    async def fail_prompt(message, password=False):
        raise AssertionError("key prompt must not run when the env var is set")

    async def fake_fetch(provider):
        return [("openai/gpt-9", "")], "live"

    async def fake_validate(model):
        return True, ""

    monkeypatch.setattr(ms, "select_provider", fake_select_provider)
    monkeypatch.setattr(ms, "_prompt_text", fail_prompt)
    monkeypatch.setattr(ms, "fetch_models_with_fallback", fake_fetch)
    monkeypatch.setattr(ms, "validate_model", fake_validate)
    # Existing key → use-existing menu; model changed → save still offered.
    monkeypatch.setattr(
        ms,
        "_select_async",
        _seq_select(
            [("already set", 0), ("Select a model", 0), ("Save", 2)]
        ),
    )

    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    out = asyncio.run(ms.run_setup_flow(config, _console()))
    assert out is not None
    assert out.model_name == "openai/gpt-9"


def test_setup_flow_replace_existing_key(monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-old-key-1234")

    async def fake_select_provider(current):
        return ms.PROVIDERS["openai"]

    async def fake_prompt_text(message, password=False):
        assert "OPENAI_API_KEY" in message and password
        return "sk-new-key-5678"

    async def fake_fetch(provider):
        return [("openai/gpt-9", "")], "live"

    async def fake_validate(model):
        return True, ""

    monkeypatch.setattr(ms, "select_provider", fake_select_provider)
    monkeypatch.setattr(ms, "_prompt_text", fake_prompt_text)
    monkeypatch.setattr(ms, "fetch_models_with_fallback", fake_fetch)
    monkeypatch.setattr(ms, "validate_model", fake_validate)
    monkeypatch.setattr(
        ms,
        "_select_async",
        _seq_select(
            [("already set", 1), ("Select a model", 0), ("Save", 2)]
        ),
    )

    config = AgentConfig(model_name="openai/gpt-9")
    out = asyncio.run(ms.run_setup_flow(config, _console()))
    assert out is not None
    assert os.environ["OPENAI_API_KEY"] == "sk-new-key-5678"


def test_setup_flow_custom_model_path(monkeypatch) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    async def fake_select_provider(current):
        return "custom"

    async def fake_prompt_text(message, password=False):
        assert not password
        return "openrouter/~anthropic/claude-fable-latest"

    async def fake_validate(model):
        return True, ""

    monkeypatch.setattr(ms, "select_provider", fake_select_provider)
    monkeypatch.setattr(ms, "_prompt_text", fake_prompt_text)
    monkeypatch.setattr(ms, "validate_model", fake_validate)
    monkeypatch.setattr(ms, "_select_async", _seq_select([("Save", 2)]))

    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    out = asyncio.run(ms.run_setup_flow(config, _console()))
    assert out is not None
    assert out.model_name == "openrouter/~anthropic/claude-fable-latest"


# ---------------------------------------------------------------------------
# env-file precedence (process env > project .env > user credentials)
# ---------------------------------------------------------------------------


def test_load_env_files_precedence(monkeypatch, tmp_path: Path) -> None:
    from autointerp_agent.config import load_env_files

    project = tmp_path / ".env"
    user = tmp_path / "credentials"
    project.write_text(
        "AUTOINTERP_TEST_SHARED=project\n"
        "AUTOINTERP_TEST_PROJECT=1\n"
        "AUTOINTERP_TEST_ENVWINS=project\n"
    )
    user.write_text(
        "AUTOINTERP_TEST_SHARED=user\n"
        "AUTOINTERP_TEST_USER=1\n"
        "AUTOINTERP_TEST_ENVWINS=user\n"
    )
    file_vars = (
        "AUTOINTERP_TEST_SHARED", "AUTOINTERP_TEST_PROJECT", "AUTOINTERP_TEST_USER"
    )
    for var in file_vars:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTOINTERP_TEST_ENVWINS", "env")

    try:
        load_env_files(project_env=project, user_env=user)
        assert os.environ["AUTOINTERP_TEST_SHARED"] == "project"  # project > user
        assert os.environ["AUTOINTERP_TEST_PROJECT"] == "1"
        assert os.environ["AUTOINTERP_TEST_USER"] == "1"
        assert os.environ["AUTOINTERP_TEST_ENVWINS"] == "env"  # real env > files
    finally:
        for var in file_vars:
            os.environ.pop(var, None)


def test_cli_config_fallback_honors_saved_credentials(monkeypatch, tmp_path: Path) -> None:
    """From a directory with no configs/agent.yaml, a saved model + key in
    ~/.autointerp/credentials still apply (the 'works in any directory' claim)."""
    import argparse

    import autointerp_agent.config as cfg
    from autointerp_agent.cli import _load_cli_config

    creds = tmp_path / "credentials"
    creds.write_text("AUTOINTERP_MODEL=openai/gpt-test\nAUTOINTERP_TEST_K=1\n")
    monkeypatch.delenv("AUTOINTERP_MODEL", raising=False)
    monkeypatch.delenv("AUTOINTERP_TEST_K", raising=False)
    monkeypatch.setattr(cfg, "USER_ENV_PATH", creds)
    monkeypatch.chdir(tmp_path)  # no config file here

    # Isolate from any real repo .env (a developer's saved credentials would
    # otherwise win via the project>user precedence and pollute the test).
    real_load_dotenv = cfg.load_dotenv

    def isolated_load_dotenv(path=None, **kwargs):
        if path is None:
            return False  # skip the repo-.env stack-walk discovery
        return real_load_dotenv(path, **kwargs)

    monkeypatch.setattr(cfg, "load_dotenv", isolated_load_dotenv)

    args = argparse.Namespace(
        config="configs/agent.yaml", model=None, max_iterations=None,
        skills_dir=None, auto_approve=False,
    )
    try:
        config = _load_cli_config(args)
        assert config.model_name == "openai/gpt-test"
        assert os.environ["AUTOINTERP_TEST_K"] == "1"
    finally:
        os.environ.pop("AUTOINTERP_MODEL", None)
        os.environ.pop("AUTOINTERP_TEST_K", None)


# REPL command dispatch lives in autointerp_agent.repl — see test_repl_ui.py.


# ---------------------------------------------------------------------------
# local (HuggingFace) models
# ---------------------------------------------------------------------------


def test_local_model_helpers(monkeypatch) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    m = "hosted_vllm/Qwen/Qwen2.5-7B-Instruct"
    assert ms.is_local_model(m)
    assert not ms.is_local_model("anthropic/claude-sonnet-4-5")
    # Local models need no API key gate.
    assert ms.missing_key_env(m) is None
    assert ms.local_api_base() == ms.DEFAULT_LOCAL_API_BASE
    # Curated list: 8 models, each with a parameter size shown.
    assert len(ms.LOCAL_MODELS) == 8
    assert all("B ·" in desc for _, desc in ms.LOCAL_MODELS)


def test_select_local_model_curated_and_custom(monkeypatch) -> None:
    async def pick_first(title, options, default_index=0):
        assert "open-weights" in title
        return 0

    monkeypatch.setattr(ms, "_select_async", pick_first)
    assert asyncio.run(ms.select_local_model(None)) == ms.LOCAL_MODELS[0][0]

    async def pick_last(title, options, default_index=0):
        return len(options) - 1

    async def fake_prompt(message, password=False):
        return "my-org/my-model"

    monkeypatch.setattr(ms, "_select_async", pick_last)
    monkeypatch.setattr(ms, "_prompt_text", fake_prompt)
    assert asyncio.run(ms.select_local_model(None)) == "my-org/my-model"


def test_setup_flow_local_model_saves_base_url(monkeypatch, tmp_path: Path) -> None:
    _clear_keys(monkeypatch)
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    monkeypatch.delenv("AUTOINTERP_MODEL", raising=False)
    monkeypatch.setattr(ms, "USER_ENV_PATH", tmp_path / "creds")

    async def fake_select_provider(current):
        return "local"

    async def fake_select_local(current):
        return "Qwen/Qwen2.5-7B-Instruct"

    async def fake_prompt(message, password=False):
        return "http://localhost:9000/v1"  # the base-URL prompt

    async def fake_validate(model):
        assert model == "hosted_vllm/Qwen/Qwen2.5-7B-Instruct"
        return True, ""

    monkeypatch.setattr(ms, "select_provider", fake_select_provider)
    monkeypatch.setattr(ms, "select_local_model", fake_select_local)
    monkeypatch.setattr(ms, "_prompt_text", fake_prompt)
    monkeypatch.setattr(ms, "validate_model", fake_validate)
    monkeypatch.setattr(ms, "_select_async", _seq_select([("Save", 0)]))

    config = AgentConfig(model_name="anthropic/claude-sonnet-4-5")
    try:
        out = asyncio.run(ms.run_setup_flow(config, _console()))
        assert out is not None
        assert out.model_name == "hosted_vllm/Qwen/Qwen2.5-7B-Instruct"
        assert os.environ["HOSTED_VLLM_API_BASE"] == "http://localhost:9000/v1"
        creds = (tmp_path / "creds").read_text()
        assert "HOSTED_VLLM_API_BASE=http://localhost:9000/v1" in creds
        assert "AUTOINTERP_MODEL=hosted_vllm/Qwen/Qwen2.5-7B-Instruct" in creds
    finally:
        os.environ.pop("HOSTED_VLLM_API_BASE", None)
        os.environ.pop("AUTOINTERP_MODEL", None)


def test_setup_flow_switch_provider_prompts_for_new_key(monkeypatch) -> None:
    # Mid-session OpenAI -> Anthropic with no Anthropic key: must prompt for it.
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")  # current provider has a key

    async def fake_select_provider(current):
        return ms.PROVIDERS["anthropic"]  # switch to Anthropic

    prompted = {"for": None}

    async def fake_prompt(message, password=False):
        prompted["for"] = message
        return "sk-ant-new"

    async def fake_fetch(provider):
        return [("anthropic/claude-sonnet-4-5", "")], "live"

    async def fake_validate(model):
        return True, ""

    monkeypatch.setattr(ms, "select_provider", fake_select_provider)
    monkeypatch.setattr(ms, "_prompt_text", fake_prompt)
    monkeypatch.setattr(ms, "fetch_models_with_fallback", fake_fetch)
    monkeypatch.setattr(ms, "validate_model", fake_validate)
    monkeypatch.setattr(ms, "_select_async", _seq_select([("Select a model", 0), ("Save", 2)]))

    config = AgentConfig(model_name="openai/gpt-9")
    try:
        out = asyncio.run(ms.run_setup_flow(config, _console()))
        assert out is not None and out.model_name == "anthropic/claude-sonnet-4-5"
        assert "ANTHROPIC_API_KEY" in (prompted["for"] or "")  # asked for the key
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-new"
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("AUTOINTERP_MODEL", None)
