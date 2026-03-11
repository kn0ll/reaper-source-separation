# REAPER Stem Separation Plugin

**Download:** [macOS](https://github.com/kn0ll/reaper-stem-separation-plugin/releases/latest/download/reaper_stem_separation_plugin.dylib) | [Windows](https://github.com/kn0ll/reaper-stem-separation-plugin/releases/latest/download/reaper_stem_separation_plugin.dll) | [Linux](https://github.com/kn0ll/reaper-stem-separation-plugin/releases/latest/download/reaper_stem_separation_plugin.so)

Right click any audio item, click "Separate stems", and get individual tracks for vocals, drums, bass, guitar, piano, and more.

![Screenshot](screenshot.png)

## Contents

- [Installation](#installation)
- [GPU Acceleration](#gpu-acceleration)
- [Models](#models)
- [Development](#development)
- [License](#license)

## Installation

Download the plugin binary for your platform from the [latest release](https://github.com/kn0ll/reaper-stem-separation-plugin/releases/latest), place it in your REAPER `UserPlugins` folder, and restart REAPER.

<details>
<summary>ReaPack</summary>

1. **Extensions > ReaPack > Import repositories...**
2. Paste: `https://raw.githubusercontent.com/kn0ll/reaper-stem-separation-plugin/master/index.xml`
3. **Extensions > ReaPack > Browse packages**, search "Stem Separation", click Install
4. Restart REAPER

</details>

<details>
<summary>macOS</summary>

```bash
curl -fSL -o ~/Library/Application\ Support/REAPER/UserPlugins/reaper_stem_separation_plugin.dylib \
  https://github.com/kn0ll/reaper-stem-separation-plugin/releases/latest/download/reaper_stem_separation_plugin.dylib
```

</details>

<details>
<summary>Windows</summary>

```powershell
Invoke-WebRequest https://github.com/kn0ll/reaper-stem-separation-plugin/releases/latest/download/reaper_stem_separation_plugin.dll -OutFile "$env:APPDATA\REAPER\UserPlugins\reaper_stem_separation_plugin.dll"
```

</details>

<details>
<summary>Linux</summary>

```bash
curl -fSL -o ~/.config/REAPER/UserPlugins/reaper_stem_separation_plugin.so \
  https://github.com/kn0ll/reaper-stem-separation-plugin/releases/latest/download/reaper_stem_separation_plugin.so
```

</details>


## GPU Acceleration

The plugin runs on CPU by default. For faster processing with an NVIDIA GPU, install the CUDA runtime for your platform.

<details>
<summary>macOS</summary>

GPU acceleration is not available on macOS. The plugin runs on CPU, which is still plenty fast for most tracks.

</details>

<details>
<summary>Windows</summary>

Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn), then restart REAPER.

```powershell
# After installing both, verify with:
nvidia-smi
```

</details>

<details>
<summary>Linux</summary>

Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/installation/linux.html), then restart REAPER.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12 libcudnn9-cuda-12
```

</details>

## Models

| Label | Model |
|-------|-------|
| **Vocals (Best quality)** | [BS-RoFormer HyperACE](https://huggingface.co/pcunwa/BS-Roformer-HyperACE) |
| **Vocals (Fast)** | [MelBand-RoFormer](https://huggingface.co/KimberleyJSN/melbandroformer) |
| **Vocals, Drums, Bass, Other** | [HTDemucs](https://github.com/facebookresearch/demucs) |
| **Vocals, Drums, Bass, Other, Guitar, Piano** | [HTDemucs 6s](https://github.com/facebookresearch/demucs) |

ONNX Runtime and models are downloaded automatically on first use, then cached in `UserPlugins` for future use.

## Development

[![Build](https://github.com/kn0ll/reaper-stem-separation-plugin/actions/workflows/build.yml/badge.svg)](https://github.com/kn0ll/reaper-stem-separation-plugin/actions/workflows/build.yml)

### Devcontainer

The fastest way to get started. Open the included [devcontainer](https://containers.dev/supporting) in your preferred IDE, all build dependencies are pre-installed.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/kn0ll/reaper-stem-separation-plugin)

### Build From Source

Requires a C++20 compiler, CMake 3.18+, Ninja, Eigen3. Optionally requires [Docker](https://docs.docker.com/get-docker/) for model conversion.

<details>
<summary>macOS</summary>

```bash
# Required dependencies (if not using Devcontainer)
brew install cmake ninja eigen

# Build the plugin and symlink to REAPER installation
make plugin
ln -sf "$(pwd)/build/reaper_stem_separation_plugin.dylib" ~/Library/Application\ Support/REAPER/UserPlugins/reaper_stem_separation_plugin.dylib

# Build the models and symlink to REAPER installation (optional, downloaded on first use)
make models
mkdir -p ~/Library/Application\ Support/REAPER/UserPlugins/reaper-stem-separation-plugin
ln -sfn "$(pwd)/models" ~/Library/Application\ Support/REAPER/UserPlugins/reaper-stem-separation-plugin/models
```

</details>

<details>
<summary>Windows</summary>

Install [Visual Studio](https://visualstudio.microsoft.com/) (for MSVC), then from an elevated prompt:

```powershell
choco install cmake ninja make -y
```

Then from a **Developer Command Prompt**:

```powershell
make plugin

# Copy into REAPER installation
copy build\reaper_stem_separation_plugin.dll "$env:APPDATA\REAPER\UserPlugins\"

# Build models (optional, requires Docker, downloaded on first use)
make models
xcopy /E /I models "$env:APPDATA\REAPER\UserPlugins\reaper-stem-separation-plugin\models\"
```

</details>

<details>
<summary>Linux</summary>

```bash
# Required dependencies (if not using Devcontainer)
sudo apt-get install cmake ninja-build libeigen3-dev

# Build the plugin and symlink to REAPER installation
make plugin
ln -sf "$(pwd)/build/reaper_stem_separation_plugin.so" ~/.config/REAPER/UserPlugins/reaper_stem_separation_plugin.so

# Build the models and symlink to REAPER installation (optional, downloaded on first use)
make models
mkdir -p ~/.config/REAPER/UserPlugins/reaper-stem-separation-plugin
ln -sfn "$(pwd)/models" ~/.config/REAPER/UserPlugins/reaper-stem-separation-plugin/models
```

</details>

## License

MIT License. See [LICENSE](LICENSE) for full license and third-party notices.
