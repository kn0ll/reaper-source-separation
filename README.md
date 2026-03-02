# reaper-source-separation

A native REAPER extension for AI source separation. Right-click any audio item, select **Separate Sources**, and get individual stems (vocals, drums, bass, etc.) as new tracks in your project.

Models are downloaded automatically the first time you use them -- no manual setup required.

## Install

1. Go to the [Releases](https://github.com/kn0ll/reaper-source-separation/releases) page
2. Download the archive for your platform:
   - **Linux**: `reaper-source-separation-linux-x64.tar.gz`
   - **macOS**: `reaper-source-separation-macos-arm64.tar.gz`
   - **Windows**: `reaper-source-separation-windows-x64.zip`
3. Extract into your REAPER `UserPlugins` folder:
   - **Linux**: `~/.config/REAPER/UserPlugins/`
   - **macOS**: `~/Library/Application Support/REAPER/UserPlugins/`
   - **Windows**: `%APPDATA%\REAPER\UserPlugins\`
4. Restart REAPER

After extracting, your `UserPlugins` folder should look like:

```
UserPlugins/
  reaper_source_separation.so          (or .dylib / .dll)
  reaper-source-separation/
    libonnxruntime.so*      (or .dylib / .dll)
```

## Usage

1. Select an audio item in REAPER
2. Right-click and choose **Separate Sources**
3. Pick a model and number of threads
4. Click **Separate** -- a progress bar tracks the work
5. Stem tracks appear in your project when complete

## How it works

The plugin runs source separation models directly inside REAPER using [ONNX Runtime](https://onnxruntime.ai/). Audio from the selected item is decoded, fed through the neural network, and the resulting stems are written as WAV files and imported as new tracks. All processing happens in a background thread so REAPER stays responsive.

Models are stored in `UserPlugins/reaper-source-separation/models/` and are downloaded from the GitHub Release on first use.

## Building from source

Requires CMake 3.18+, Ninja, a C++20 compiler, and Eigen3.

```bash
git clone --recurse-submodules https://github.com/kn0ll/reaper-source-separation.git
cd reaper-source-separation
make dist
```

This produces a platform archive (e.g. `reaper-source-separation-linux-x64.tar.gz`) ready to extract into `UserPlugins/`.

Set `ORT_PREFIX` if ONNX Runtime is installed somewhere non-standard:

```bash
make dist ORT_PREFIX=/path/to/onnxruntime
```

## Running from source

For faster iteration during development, you can symlink the built plugin instead of copying it each time:

```bash
make plugin
ln -sf "$(pwd)/build/reaper_source_separation.so" ~/.config/REAPER/UserPlugins/reaper_source_separation.so
ln -sfn "$(pwd)/build/reaper-source-separation" ~/.config/REAPER/UserPlugins/reaper-source-separation
```

Place model files in the repo's `models/` directory and the plugin will find them there before attempting a download.

### Converting models locally

Only needed if you want to test with local model files instead of downloaded ones:

```bash
pip install torch onnxruntime onnx
pip install -e vendor/demucs.onnx/demucs-for-onnx
make models
```
