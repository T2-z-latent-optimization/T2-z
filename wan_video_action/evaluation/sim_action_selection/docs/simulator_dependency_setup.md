# Simulator dependencies

```bash
python wan_video_action/evaluation/sim_action_selection/setup_simulator_dependencies.py
bash wan_video_action/evaluation/sim_action_selection/setup_model_environment.sh
```

`simulator_dependencies.json` records source revisions, required files and asset
checksums. Existing source changes and assets are preserved. Sources, assets and
virtual environments are installed outside this package.

The simulator environments use `requirements-simulator-libero.txt` and
`requirements-simulator-robomme.txt`; the model overlay uses
`requirements-model.txt`. Installation uses uv and the official PyPI index.

Native rendering requires compatible GPU drivers and EGL for MuJoCo or Vulkan
for SAPIEN. LIBERO-plus additionally requires its asset bundle, scene overlays
and MagickWand. Host drivers and system packages are not changed by the installer.

Pass the installed source root and the respective Python executables through
`--source-root`, `--simulator-python` and `--model-python`.
