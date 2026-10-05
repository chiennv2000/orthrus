# SPDX-License-Identifier: Apache-2.0
import pytest

from orthrus_vllm.speculative import activate


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(activate.ENV_FLAG, raising=False)
    assert not activate.enabled()


def test_enabled_only_when_flag_is_one(monkeypatch):
    monkeypatch.setenv(activate.ENV_FLAG, "1")
    assert activate.enabled()
    monkeypatch.setenv(activate.ENV_FLAG, "0")
    assert not activate.enabled()


def test_refuses_unsupported_vllm_version(monkeypatch):
    import vllm

    monkeypatch.setattr(vllm, "__version__", "0.1.0")
    with pytest.raises(RuntimeError, match="only supported"):
        activate.activate()
