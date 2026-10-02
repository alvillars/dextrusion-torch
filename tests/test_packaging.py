import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(),
                    reason="needs a git checkout")
def test_no_source_file_is_ignored_by_gitignore():
    """Hatchling leaves files matched by .gitignore out of the wheel: an unanchored `data/` entry
    once dropped the `dextrusion.data` subpackage from every non-editable install."""
    out = subprocess.run(["git", "ls-files", "--cached", "--ignored", "--exclude-standard", "src"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert out.split() == []
