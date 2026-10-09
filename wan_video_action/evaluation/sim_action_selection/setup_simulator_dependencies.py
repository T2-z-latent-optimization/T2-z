"""Load the original collectors, assets and isolated native Python environments.

This is an explicit installation entry point, never called by evaluation.
Existing working trees, source YAMLs, environments and datasets are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import zipfile


HERE = Path(__file__).resolve().parent


def run(*args, cwd=None, capture=False):
    env = os.environ.copy()
    # Do not inherit a user's mirror configuration for package installation.
    for key in ("UV_INDEX", "UV_EXTRA_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX",
                "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "UV_FIND_LINKS", "PIP_FIND_LINKS"):
        env.pop(key, None)
    env["UV_NO_CONFIG"] = "1"
    result = subprocess.run([str(arg) for arg in args], cwd=cwd, env=env,
                            check=True, text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def require_paths(root, paths):
    missing = [str(root / path) for path in paths if not (root / path).exists()]
    if missing:
        raise RuntimeError("Missing original simulation dependencies:\n" + "\n".join(missing))


def ensure_repository(root, spec):
    directory = root / spec["directory"]
    revision = spec["revision"]
    if not directory.exists():
        run("git", "clone", "--filter=blob:none", "--no-checkout", spec["url"], directory)
        run("git", "fetch", "origin", revision, cwd=directory)
        run("git", "checkout", "--detach", revision, cwd=directory)
    else:
        if not (directory / ".git").exists():
            raise RuntimeError(f"Preserving existing non-Git directory: {directory}")
        head = run("git", "rev-parse", "HEAD", cwd=directory, capture=True)
        if head != revision:
            run("git", "fetch", "origin", revision, cwd=directory)
            ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=directory)
            if ancestor.returncode not in (0, 1):
                raise RuntimeError(f"Cannot determine ancestry in {directory}")
            if ancestor.returncode == 1:
                dirty = run("git", "status", "--porcelain", "--untracked-files=no", cwd=directory, capture=True)
                if dirty:
                    raise RuntimeError(f"Refusing to update modified source files in {directory}")
                # Never reset, force-checkout or merge divergent physics sources.
                run("git", "merge", "--ff-only", revision, cwd=directory)
            else:
                print(f"Keeping existing descendant revision in {directory}: {head}", flush=True)
    require_paths(directory, spec["required"])


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_missing_assets(archive, destination):
    with zipfile.ZipFile(archive) as bundle:
        members = []
        asset_roots = set()
        for member in bundle.infolist():
            name = PurePosixPath(member.filename)
            if name.parts and name.parts[0] == "__MACOSX":
                continue
            if (name.is_absolute() or ".." in name.parts or not name.parts
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise RuntimeError(f"Unexpected official asset archive path: {member.filename}")
            if "assets" not in name.parts:
                if member.is_dir():
                    continue
                raise RuntimeError(f"Unexpected official asset archive path: {member.filename}")
            asset_index = name.parts.index("assets")
            asset_roots.add(name.parts[:asset_index + 1])
            relative = Path(*name.parts[asset_index:])
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise RuntimeError(f"Asset path escapes installation directory: {member.filename}")
            members.append((member, relative))
        if len(asset_roots) != 1:
            raise RuntimeError("Official archive must contain exactly one assets root")
        # Preserve any existing local asset, including local visual overlays.
        for member, relative in members:
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)


def ensure_plus_assets(root, cache, spec):
    package = root / "LIBERO-plus" / "libero" / "libero"
    assets = package / "assets"
    if any(not (assets / path).is_file() for path in spec["required"]):
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / f"libero-plus-{spec['sha256']}.zip"
        if not archive.exists() or archive.stat().st_size != spec["archive_size_bytes"]:
            print("Downloading official LIBERO-plus assets (6.40 GB archive).", flush=True)
            run("curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                "--output", archive, spec["url"])
        if file_sha256(archive) != spec["sha256"]:
            raise RuntimeError(f"Official asset checksum mismatch; preserving archive for diagnosis: {archive}")
        extract_missing_assets(archive, package)
    overlay = root / spec["overlay_source"]
    require_paths(overlay, ["scenes"])
    for source in sorted(overlay.rglob("*.xml")):
        target = assets / source.relative_to(overlay)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_sha256(target) != file_sha256(source):
                raise RuntimeError(f"Preserving different local scene overlay: {target}")
        else:
            shutil.copyfile(source, target)
    require_paths(assets, spec["required"])


def ensure_environment(root, spec):
    venv = root / spec["directory"] / ".venv"
    python = venv / "bin" / "python"
    if venv.exists():
        if not python.exists():
            raise RuntimeError(f"Preserving incomplete environment: {venv}")
        print(f"Reusing existing environment without package changes: {venv}", flush=True)
        return
    if shutil.which("uv") is None:
        raise RuntimeError("Install uv from https://docs.astral.sh/uv/ before creating native environments")
    run("uv", "venv", "--python", spec["python"], venv)
    # Native collector environments contain intentional upstream dependency
    # mismatches. Reproduce the collected versions, do not re-resolve a new
    # physics stack through FastWAM's unrelated training dependencies.
    run("uv", "pip", "install", "--python", python,
        "--default-index", "https://pypi.org/simple", "--no-deps",
        "-r", HERE / spec["requirements"])
    for repository in spec["editable_sources"]:
        run("uv", "pip", "install", "--python", python,
            "--default-index", "https://pypi.org/simple", "--no-deps", "-e", root / repository)
    # Do not editable-install LIBERO-plus into this environment: the ordinary
    # LIBERO package stays intact, and Plus workers explicitly pin its namespace.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos-root", type=Path, default=HERE.parents[2].parent)
    parser.add_argument("--cache-dir", type=Path, default=HERE.parents[2] / ".cache" / "simulator_assets")
    parser.add_argument("--skip-assets", action="store_true", help="Synchronize sources without downloading Plus assets")
    parser.add_argument("--skip-environments", action="store_true", help="Keep environment creation separate")
    args = parser.parse_args()
    root = args.repos_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "simulator_dependencies.json").read_text(encoding="utf-8"))
    for spec in manifest["repositories"]:
        ensure_repository(root, spec)
    if not args.skip_assets:
        ensure_plus_assets(root, args.cache_dir.expanduser().resolve(), manifest["plus_assets"])
    if not args.skip_environments:
        for spec in manifest["environments"]:
            ensure_environment(root, spec)
    print("Original simulator dependencies loaded. No datasets or checkpoints were modified.")


if __name__ == "__main__":
    main()
