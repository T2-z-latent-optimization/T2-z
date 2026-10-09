import stat
import zipfile

import pytest

from ..setup_simulator_dependencies import extract_missing_assets


@pytest.mark.parametrize("prefix", ["", "release/bundle/"])
def test_asset_root_is_normalized_without_changing_contents(tmp_path, prefix):
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        if prefix:
            bundle.writestr("release/", b"")
            bundle.writestr(prefix, b"")
        bundle.writestr(prefix + "assets/scenes/table.xml", b"original scene")
    destination = tmp_path / "installed"
    extract_missing_assets(archive, destination)
    assert (destination / "assets/scenes/table.xml").read_bytes() == b"original scene"
    assert not (destination / "release").exists()


def test_existing_local_asset_is_preserved(tmp_path):
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("release/assets/scenes/table.xml", b"downloaded scene")
    destination = tmp_path / "installed"
    target = destination / "assets/scenes/table.xml"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"local scene")
    extract_missing_assets(archive, destination)
    assert target.read_bytes() == b"local scene"


@pytest.mark.parametrize("path", ["../assets/escape.xml", "/assets/escape.xml"])
def test_unsafe_archive_path_is_rejected_before_extraction(tmp_path, path):
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("release/assets/scenes/table.xml", b"scene")
        bundle.writestr(path, b"unsafe")
    destination = tmp_path / "installed"
    with pytest.raises(RuntimeError, match="Unexpected official asset archive path"):
        extract_missing_assets(archive, destination)
    assert not destination.exists()


def test_archive_symlink_is_rejected(tmp_path):
    archive = tmp_path / "assets.zip"
    link = zipfile.ZipInfo("release/assets/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(link, b"../../outside")
    with pytest.raises(RuntimeError, match="Unexpected official asset archive path"):
        extract_missing_assets(archive, tmp_path / "installed")


def test_multiple_asset_roots_are_rejected_before_extraction(tmp_path):
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("first/assets/scenes/a.xml", b"a")
        bundle.writestr("second/assets/scenes/b.xml", b"b")
    destination = tmp_path / "installed"
    with pytest.raises(RuntimeError, match="exactly one assets root"):
        extract_missing_assets(archive, destination)
    assert not destination.exists()
