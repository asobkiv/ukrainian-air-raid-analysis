"""Create a source/report ZIP and SHA256 manifest; exclude local Telegram HTML."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    ignored = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv", "venv", "work"}
    paths = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or any(p in ignored or p.endswith(".egg-info") for p in rel.parts):
            continue
        if rel.parts[:3] == ("data", "telegram", "raw") or path.name.startswith(".env"):
            continue
        if path.name != "artifact_manifest.json":
            paths.append(path)
    manifest = root / "artifact_manifest.json"
    manifest.write_text(json.dumps({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2))
    archive = root.parent / (root.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in paths + [manifest]:
            z.write(path, str(Path(root.name) / path.relative_to(root)))
    with zipfile.ZipFile(archive) as z:
        assert not any("/data/telegram/raw/" in n or "/.git/" in n for n in z.namelist())
        assert z.testzip() is None
    archive.with_suffix(".zip.sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "  " + archive.name + "\n")
    print(f"Packaged {len(paths) + 1} files, {archive.stat().st_size:,} bytes; local raw HTML excluded")


if __name__ == "__main__":
    main()
