"""Unit tests for the optional LLM-based closet regeneration.

These tests don't hit the network. They mock urllib to verify:
- LLMConfig correctly reads env vars and CLI overrides
- missing config is reported cleanly
- the OpenAI-compatible request shape is correct
- response parsing handles the standard chat-completions payload
"""

import io
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from mempalace.closet_llm import (
    LLMConfig,
    _call_llm,
    _parsed_to_closet_lines,
    regenerate_closets,
)


# ── LLMConfig ─────────────────────────────────────────────────────────────


class TestLLMConfig:
    def test_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("LLM_ENDPOINT", "http://localhost:11434/v1")
        monkeypatch.setenv("LLM_KEY", "sk-abc")
        monkeypatch.setenv("LLM_MODEL", "llama3:8b")
        c = LLMConfig()
        assert c.endpoint == "http://localhost:11434/v1"
        assert c.key == "sk-abc"
        assert c.model == "llama3:8b"

    def test_cli_flags_override_env(self, monkeypatch):
        monkeypatch.setenv("LLM_ENDPOINT", "http://env-endpoint/v1")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        c = LLMConfig(endpoint="http://flag-endpoint/v1", model="flag-model")
        assert c.endpoint == "http://flag-endpoint/v1"
        assert c.model == "flag-model"

    def test_trailing_slash_stripped(self):
        c = LLMConfig(endpoint="http://foo/v1/", model="m")
        assert c.endpoint == "http://foo/v1"

    def test_missing_reports_required(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT", raising=False)
        monkeypatch.delenv("LLM_KEY", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        c = LLMConfig()
        missing = c.missing()
        assert any("ENDPOINT" in m for m in missing)
        assert any("MODEL" in m for m in missing)
        # key is optional
        assert not any("KEY" in m for m in missing)

    def test_key_is_optional(self, monkeypatch):
        monkeypatch.delenv("LLM_KEY", raising=False)
        c = LLMConfig(endpoint="http://local/v1", model="m")
        assert c.missing() == []


# ── _parsed_to_closet_lines ──────────────────────────────────────────────


class TestParsedToLines:
    def test_topics_become_pointers(self):
        parsed = {"topics": ["authentication", "jwt tokens"], "quotes": [], "summary": ""}
        lines = _parsed_to_closet_lines(parsed, ["d1", "d2"], "Alice;Bob")
        assert len(lines) == 2
        assert "authentication|Alice;Bob|→d1,d2" in lines
        assert "jwt tokens|Alice;Bob|→d1,d2" in lines

    def test_quotes_and_summary_included(self):
        parsed = {
            "topics": ["t1"],
            "quotes": ["[Igor] we ship Friday"],
            "summary": "Release planning discussion",
        }
        lines = _parsed_to_closet_lines(parsed, ["d1"], "")
        joined = "\n".join(lines)
        assert "we ship Friday" in joined
        assert "Release planning discussion" in joined

    def test_caps_topics_at_15(self):
        parsed = {"topics": [f"t{i}" for i in range(20)], "quotes": [], "summary": ""}
        lines = _parsed_to_closet_lines(parsed, ["d1"], "")
        assert len(lines) == 15


# ── _call_llm (HTTP mocked) ──────────────────────────────────────────────


class _FakeResp:
    """Mimics urlopen's context-manager response."""

    def __init__(self, payload: dict, status: int = 200):
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


class TestCallLLM:
    def _make_cfg(self):
        return LLMConfig(
            endpoint="http://localhost:11434/v1", key="sk-test", model="llama3:8b"
        )

    def test_request_shape_and_parsing(self):
        cfg = self._make_cfg()
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeResp(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "topics": ["postgres"],
                                        "quotes": ["[Igor] migrate now"],
                                        "summary": "db migration",
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 42, "completion_tokens": 17},
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            parsed, usage = _call_llm(cfg, "/tmp/test.md", "w", "r", "content body")

        assert parsed["topics"] == ["postgres"]
        assert usage["prompt_tokens"] == 42
        assert captured["url"] == "http://localhost:11434/v1/chat/completions"
        # Authorization header is stored capitalized-then-lowercase depending on urllib version
        auth_vals = {v for k, v in captured["headers"].items() if k.lower() == "authorization"}
        assert "Bearer sk-test" in auth_vals
        assert captured["body"]["model"] == "llama3:8b"
        assert captured["body"]["messages"][0]["role"] == "user"

    def test_omits_auth_header_when_no_key(self):
        cfg = LLMConfig(endpoint="http://localhost:11434/v1", model="llama3:8b")
        captured_headers = {}

        def fake_urlopen(req, timeout=None):
            captured_headers.update({k.lower(): v for k, v in req.header_items()})
            return _FakeResp(
                {
                    "choices": [
                        {"message": {"content": '{"topics":[],"quotes":[],"summary":""}'}}
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _call_llm(cfg, "/tmp/x", "w", "r", "c")

        assert "authorization" not in captured_headers

    def test_strips_code_fences(self):
        cfg = self._make_cfg()
        fenced = '```json\n{"topics":["t1"],"quotes":[],"summary":""}\n```'

        def fake_urlopen(req, timeout=None):
            return _FakeResp(
                {
                    "choices": [{"message": {"content": fenced}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            parsed, _ = _call_llm(cfg, "/tmp/x", "w", "r", "c")
        assert parsed == {"topics": ["t1"], "quotes": [], "summary": ""}

    def test_returns_none_on_invalid_json(self):
        cfg = self._make_cfg()

        def fake_urlopen(req, timeout=None):
            return _FakeResp(
                {
                    "choices": [{"message": {"content": "not json at all"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            parsed, usage = _call_llm(cfg, "/tmp/x", "w", "r", "c")
        assert parsed is None


# ── regenerate_closets error paths ───────────────────────────────────────


class TestRegenerateClosets:
    def test_missing_config_returns_error(self, monkeypatch):
        monkeypatch.delenv("LLM_ENDPOINT", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        with tempfile.TemporaryDirectory() as palace:
            result = regenerate_closets(palace)
            assert result["error"] == "missing-config"
            assert any("ENDPOINT" in m for m in result["missing"])
