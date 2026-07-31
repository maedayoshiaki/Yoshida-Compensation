"""Tests for the packaged GrayCode ``sys.path`` bootstrap.

GrayCode は git submodule でありインストール済みパッケージではないため、
``external/GrayCode/src`` を ``sys.path`` に載せる処理が必要になる。この
処理は以前 4 箇所に複製されていた。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ensure_on_syspath_returns_the_vendored_source_dir() -> None:
    from yoshidacomp.geometry.graycode_path import ensure_on_syspath

    found = ensure_on_syspath()

    assert found == _REPO_ROOT / "external" / "GrayCode" / "src"
    assert (found / "graycode").is_dir()
    assert str(found) in sys.path


def test_ensure_on_syspath_is_idempotent() -> None:
    """複数回呼んでも sys.path に重複エントリを作らないこと。

    Four separate modules bootstrap this path; with an ``examples`` script
    imported from a test that already bootstrapped, the helper runs more
    than once per interpreter.
    """
    from yoshidacomp.geometry.graycode_path import ensure_on_syspath

    first = ensure_on_syspath()
    snapshot = list(sys.path)

    second = ensure_on_syspath()
    third = ensure_on_syspath()

    assert second == first
    assert third == first
    assert sys.path == snapshot
    assert sys.path.count(str(first)) == 1


def test_find_graycode_src_reports_what_it_looked_for(monkeypatch, tmp_path) -> None:
    """見つからない場合は探索内容を明示した例外を出すこと。

    Failing here beats failing later with a bare ``ModuleNotFoundError:
    No module named 'graycode'`` from an unrelated call site.
    """
    from yoshidacomp.geometry import graycode_path

    monkeypatch.delenv(graycode_path.GRAYCODE_SRC_ENV_VAR, raising=False)
    monkeypatch.setattr(graycode_path, "_candidate_start_dirs", lambda: [tmp_path])

    with pytest.raises(FileNotFoundError) as excinfo:
        graycode_path.find_graycode_src()

    message = str(excinfo.value)
    assert "external/GrayCode/src" in message.replace("\\", "/")
    assert str(tmp_path) in message
    assert graycode_path.GRAYCODE_SRC_ENV_VAR in message


def test_env_var_overrides_the_search(monkeypatch, tmp_path) -> None:
    """editable 配置以外でも明示指定で解決できること。"""
    from yoshidacomp.geometry import graycode_path

    override = tmp_path / "vendored" / "src"
    (override / "graycode").mkdir(parents=True)
    monkeypatch.setenv(graycode_path.GRAYCODE_SRC_ENV_VAR, str(override))

    assert graycode_path.find_graycode_src() == override.resolve()


def test_module_imports_with_graycode_blocked() -> None:
    """``graycode`` を遮断しても helper が import・実行できること。

    ``geometry/warp.py`` defers every GrayCode import into its functions so
    that ``import yoshidacomp.geometry.warp`` stays GrayCode-free (guarded by
    ``tests/test_package_boundaries.py``). A helper that imported
    ``graycode`` to find its path would undo exactly that.
    """
    script = """
import sys


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "graycode":
            raise ImportError(f"blocked: {name}")
        return None


sys.meta_path.insert(0, Blocker())

from yoshidacomp.geometry.graycode_path import ensure_on_syspath

found = ensure_on_syspath()
assert found.is_dir(), found
assert "graycode" not in sys.modules, sorted(sys.modules)

print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
