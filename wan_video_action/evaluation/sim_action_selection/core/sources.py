from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types


class Sources:
    """Import immutable collectors; generated BDDL lives only in our own tmp dir."""

    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        if not (self.root / "ttt4dynamics" / "push_box_libero.py").is_file():
            raise FileNotFoundError(f"Not a TTT4dynamics source checkout: {self.root}")
        self.repos = self.root.parent
        for path in (self.root, self.root / "scripts", self.repos / "FastWAM-TTT" / "src"):
            sys.path.insert(0, str(path))
        os.environ.setdefault("MUJOCO_GL", "egl")
        scratch = self.root / "tmp" / "sim_action_selection"
        scratch.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="decision_", dir=scratch)
        self.work = Path(self._temporary.name)
        self.provenance: dict[str, str] = {}

    def load(self, relative: str, name: str):
        path = self.root / relative
        return self.load_path(path, name)

    def load_path(self, path: Path, name: str):
        path = path.resolve()
        self.provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        existing = sys.modules.get(name)
        if (existing is not None
                and getattr(existing, "__file__", None) == str(path)
                and getattr(existing, "_t2z_source_sha256", None) == self.provenance[str(path)]):
            return existing
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import original collector: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            if sys.modules.get(name) is module:
                del sys.modules[name]
            raise
        module._t2z_source_sha256 = self.provenance[str(path)]
        return module

    def config(self, relative: str) -> dict:
        path = self.root / relative
        raw = path.read_bytes()
        self.provenance[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def pin_libero(self, plus: bool = False):
        root = self.repos / ("LIBERO-plus" if plus else "LIBERO")
        root = (root / "libero").resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        if "libero.libero.envs" in sys.modules:
            current = sys.modules.get("libero")
            if root in [Path(path).resolve() for path in getattr(current, "__path__", ())]:
                return
            raise RuntimeError("Cannot switch LIBERO source after importing a simulator; use a separate worker")
        namespace = types.ModuleType("libero")
        namespace.__path__ = [str(root)]
        namespace.__package__ = "libero"
        namespace.__spec__ = importlib.util.spec_from_loader("libero", loader=None, is_package=True)
        namespace.__spec__.submodule_search_locations = [str(root)]
        sys.modules["libero"] = namespace

    @staticmethod
    def disable_optional_wand():
        """Same optional-import workaround as the original Plus launcher.

        This is not a physics mock. ImageMagick augmentation is not used here;
        invoking it is an error rather than silently fabricating an image.
        """
        class Unavailable:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("ImageMagick augmentation is unavailable in this worker")

        package = types.ModuleType("wand")
        package.__path__ = []
        api = types.ModuleType("wand.api")
        api.library = types.SimpleNamespace(MagickMotionBlurImage=Unavailable())
        image = types.ModuleType("wand.image")
        image.Image = Unavailable
        package.api, package.image = api, image
        sys.modules.update({"wand": package, "wand.api": api, "wand.image": image})

    def close(self):
        self._temporary.cleanup()
