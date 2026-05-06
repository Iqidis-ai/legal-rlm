"""Repository-wide pytest harness fixes.

Windows + OneDrive on this workstation can make pytest's default global temp
root (`AppData/Local/Temp/pytest-of-*`) and cache teardown inaccessible. Keep
test temp paths inside the repo with normal directory creation so tmp_path
fixtures remain usable and pytest exits cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path


def pytest_configure(config):
    if os.name != "nt":
        return

    import _pytest.pathlib as pytest_pathlib
    from _pytest.tmpdir import TempPathFactory

    temp_root = Path(config.rootpath) / ".pytest-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    def _repo_local_basetemp(self):
        basetemp = getattr(self, "_basetemp", None)
        if basetemp is not None:
            return basetemp
        temp_root.mkdir(parents=True, exist_ok=True)
        self._basetemp = temp_root
        return temp_root

    def _repo_local_mktemp(self, basename, numbered=True):
        base = self.getbasetemp()
        name = str(basename)
        if not numbered:
            path = base / name
            path.mkdir(parents=True, exist_ok=False)
            return path
        for idx in range(10_000):
            path = base / f"{name}{idx}"
            try:
                path.mkdir(parents=True, exist_ok=False)
                return path
            except FileExistsError:
                continue
        raise FileExistsError(f"could not allocate pytest temp path for {name!r}")

    TempPathFactory.getbasetemp = _repo_local_basetemp
    TempPathFactory.mktemp = _repo_local_mktemp
    pytest_pathlib.cleanup_dead_symlinks = lambda root: None
