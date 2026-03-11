# reaper-stem-separation-plugin

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/kn0ll/reaper-stem-separation-plugin)

[![CI](https://github.com/kn0ll/reaper-stem-separation-plugin/actions/workflows/ci.yml/badge.svg)](https://github.com/kn0ll/reaper-stem-separation-plugin/actions/workflows/ci.yml)

AI-powered stem separation plugin for REAPER — isolate vocals, drums, bass, and more with Demucs and RoFormer, directly in your DAW.

## Table of contents

- [Install](#install)
  - [Linux](#linux)
  - [macOS](#macos)
  - [Windows](#windows)
- [Models](#models)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Building from source](#building-from-source)
- [Running from source](#running-from-source)
  - [Converting models locally](#converting-models-locally)

## Install

Download the archive for your platform from the [latest release](https://github.com/kn0ll/reaper-stem-separation-plugin/releases/latest), extract it into your REAPER `UserPlugins` folder, and restart REAPER.

### Linux

```bash
curl -fSL https://github.com/kn0ll/reaper-stem-separation-plugin/releases/latest/download/reaper-stem-separation-plugin-linux-x64-cuda.tar.gz | tar xz -C ~/.config/REAPER/UserPlugins/
```

For GPU acceleration with an NVIDIA GPU, add the [CUDA repository](https://developer.nvidia.com/cuda-downloads) and install the runtime libraries:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12 libcudnn9-cuda-12
```

Without these, the plugin still works but runs on CPU.

### macOS

```bash
curl -fSL https://github.com/kn0ll/reaper-stem-separation-plugin/releases/latest/download/reaper-stem-separation-plugin-macos-arm64-cpu.tar.gz | tar xz -C ~/Library/Application\ Support/REAPER/UserPlugins/
```

GPU acceleration is not available on macOS. The plugin runs on CPU, which is still plenty fast for most tracks.

### Windows

```powershell
Invoke-WebRequest https://github.com/kn0ll/reaper-stem-separation-plugin/releases/latest/download/reaper-stem-separation-plugin-windows-x64-cuda.zip -OutFile $env:TEMP\rss.zip; Expand-Archive $env:TEMP\rss.zip "$env:APPDATA\REAPER\UserPlugins" -Force; Remove-Item $env:TEMP\rss.zip
```

For GPU acceleration with an NVIDIA GPU, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn). Without these, the plugin still works but runs on CPU.

## Models

| Model | Stems | Best for |
|-------|-------|----------|
| **Vocals (Best quality)** — BS-RoFormer HyperACE | Vocals + Instrumental | Cleanest vocal isolation, highest overall fidelity |
| **Vocals (Fast)** — MelBand-RoFormer | Vocals + Instrumental | Quick vocal isolation, great for previewing |
| **All Stems** — HTDemucs FT | Drums, Bass, Other, Vocals | Full 4-stem separation; strong drum isolation |
| **All Stems + Guitar & Piano** — HTDemucs 6s | Drums, Bass, Other, Vocals, Guitar, Piano | Only option for guitar and piano as separate stems |

Models are downloaded automatically on first use and cached in `UserPlugins/reaper-stem-separation-plugin/models/`.

## Usage

1. Select an audio item in REAPER
2. Right-click and choose **Separate stems**
3. Pick a model -- a short description explains what each one does
4. Click **Separate** -- a progress bar tracks the work
5. Stem tracks appear in your project when complete

## How it works

The plugin supports two inference backends: **Demucs** (via the [demucs.onnx](https://github.com/sevagh/demucs.onnx) library) and **RoFormer** (direct ONNX Runtime chunked inference). Audio from the selected item is decoded, fed through the neural network, and the resulting stems are written as WAV files and imported as new tracks. All processing happens in a background thread so REAPER stays responsive.

## Building from source

Requires CMake 3.18+, Ninja, a C++20 compiler, and Eigen3.

```bash
git clone --recurse-submodules https://github.com/kn0ll/reaper-stem-separation-plugin.git
cd reaper-stem-separation-plugin
make dist
```

This produces a platform archive (e.g. `reaper-stem-separation-plugin-linux-x64-cpu.tar.gz`) ready to extract into `UserPlugins/`.

Set `ORT_PREFIX` if ONNX Runtime is installed somewhere non-standard:

```bash
make dist ORT_PREFIX=/path/to/onnxruntime
```

## Running from source

For faster iteration during development, you can symlink the built plugin instead of copying it each time:

```bash
make plugin
ln -sf "$(pwd)/build/reaper_stem_separation_plugin.so" ~/.config/REAPER/UserPlugins/reaper_stem_separation_plugin.so
ln -sfn "$(pwd)/build/reaper-stem-separation-plugin" ~/.config/REAPER/UserPlugins/reaper-stem-separation-plugin
```

Place model files in the repo's `models/` directory and the plugin will find them there before attempting a download.

### Converting models locally

Model conversion is handled automatically by CI during releases. To run it locally (requires Docker):

```bash
make models
```

This builds a self-contained Docker image with all Python dependencies, downloads checkpoints from HuggingFace, converts Demucs models to ORT format, and exports RoFormer models to ONNX. Output files land in `models/`.
