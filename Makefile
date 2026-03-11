BUILD_DIR   := build
DIST_DIR    := dist
VENDOR_DIR  := vendor/demucs.onnx
MODELS_DIR  := models
STAGING_DIR := models/staging
PYTHON      ?= python3
ORT_VERSION ?= 1.19.2

# HuggingFace model URLs
HF_BS_ROFORMER_CKPT  := https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/v2_voc/bs_roformer_voc_hyperacev2.ckpt
HF_BS_ROFORMER_CFG   := https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/v2_voc/config.yaml
HF_BS_ROFORMER_PY    := https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/v2_voc/bs_roformer.py
HF_MELBAND_CKPT      := https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt
HF_MELBAND_CFG       := https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml
HF_MELBAND_PY        := https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/models/bs_roformer/mel_band_roformer.py

# Auto-detect platform
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
UNAME_M := $(shell uname -m 2>/dev/null || echo x86_64)

ifeq ($(UNAME_S),Darwin)
    PLATFORM      := macos
    ARCH          := $(if $(filter arm64,$(UNAME_M)),arm64,x64)
    PLUGIN_EXT    := .dylib
    ORT_ASSET     := onnxruntime-osx-$(ARCH)-$(ORT_VERSION).tgz
else ifeq ($(UNAME_S),Linux)
    PLATFORM      := linux
    ARCH          := x64
    PLUGIN_EXT    := .so
    ORT_ASSET     := onnxruntime-linux-x64-$(ORT_VERSION).tgz
else
    PLATFORM      := windows
    ARCH          := x64
    PLUGIN_EXT    := .dll
    ORT_ASSET     := onnxruntime-win-x64-$(ORT_VERSION).zip
endif

ifeq ($(PLATFORM),linux)
    INSTALL_PATH := ~/.config/REAPER/UserPlugins/
else ifeq ($(PLATFORM),macos)
    INSTALL_PATH := ~/Library/Application Support/REAPER/UserPlugins/
else
    INSTALL_PATH := %APPDATA%\\REAPER\\UserPlugins\\
endif
ORT_EXTRACT_DIR := $(subst .tgz,,$(subst .zip,,$(ORT_ASSET)))

# Auto-download ORT into ort/ if ORT_PREFIX not explicitly set
ORT_PREFIX ?= $(shell ls -d ort/$(ORT_EXTRACT_DIR) 2>/dev/null)
ifeq ($(ORT_PREFIX),)
    ORT_PREFIX := /usr/local
endif

CMAKE_EXTRA ?=

.PHONY: all plugin models _models _models-demucs-4s _models-demucs-6s _models-roformer ort dist _dist clean help

MODELS_IMAGE := reaper-stem-separation-models

all: plugin

plugin:
	cmake -S . -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Release -DORT_PREFIX=$(ORT_PREFIX) $(CMAKE_EXTRA)
	cmake --build $(BUILD_DIR) --config Release
	@echo "Built: $(BUILD_DIR)/reaper_stem_separation_plugin$(PLUGIN_EXT)"

# -- Model conversion (Docker) --

models:
	docker build -f Dockerfile.models -t $(MODELS_IMAGE) .
	docker run --rm -v $(CURDIR)/models:/workspace/models -v $(CURDIR)/scripts:/workspace/scripts:ro $(MODELS_IMAGE) _models

# -- Internal targets (run inside container) --

_models: _models-demucs-4s _models-demucs-6s _models-roformer

_models-demucs-4s:
	@mkdir -p $(MODELS_DIR)
	$(PYTHON) $(VENDOR_DIR)/scripts/convert-pth-to-onnx.py $(MODELS_DIR)
	$(PYTHON) -m onnxruntime.tools.convert_onnx_models_to_ort $(MODELS_DIR)
	@if [ -f "$(MODELS_DIR)/htdemucs.with_runtime_opt.ort" ]; then \
		cp "$(MODELS_DIR)/htdemucs.with_runtime_opt.ort" "$(MODELS_DIR)/htdemucs.ort"; \
	fi

_models-demucs-6s:
	@mkdir -p $(MODELS_DIR)
	$(PYTHON) $(VENDOR_DIR)/scripts/convert-pth-to-onnx.py $(MODELS_DIR) --six-source
	$(PYTHON) -m onnxruntime.tools.convert_onnx_models_to_ort $(MODELS_DIR)
	@if [ -f "$(MODELS_DIR)/htdemucs_6s.with_runtime_opt.ort" ]; then \
		cp "$(MODELS_DIR)/htdemucs_6s.with_runtime_opt.ort" "$(MODELS_DIR)/htdemucs_6s.ort"; \
	fi

_models-roformer: _models-roformer-bs _models-roformer-mb

_models-roformer-bs:
	@mkdir -p $(STAGING_DIR)/bs_hyperace $(STAGING_DIR)/models/bs_roformer $(MODELS_DIR)
	curl -fSL -o $(STAGING_DIR)/bs_hyperace/checkpoint.ckpt "$(HF_BS_ROFORMER_CKPT)"
	curl -fSL -o $(STAGING_DIR)/bs_hyperace/config.yaml "$(HF_BS_ROFORMER_CFG)"
	curl -fSL -o $(STAGING_DIR)/bs_hyperace/bs_roformer.py "$(HF_BS_ROFORMER_PY)"
	curl -fSL -o $(STAGING_DIR)/models/bs_roformer/attend.py \
		"https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/models/bs_roformer/attend.py"
	touch $(STAGING_DIR)/models/__init__.py $(STAGING_DIR)/models/bs_roformer/__init__.py
	PYTHONPATH=$(STAGING_DIR):$$PYTHONPATH $(PYTHON) scripts/export_onnx.py \
		--model-type bs_roformer \
		--checkpoint $(STAGING_DIR)/bs_hyperace/checkpoint.ckpt \
		--config $(STAGING_DIR)/bs_hyperace/config.yaml \
		--model-code $(STAGING_DIR)/bs_hyperace/bs_roformer.py \
		--chunk-size 352800 \
		--output $(MODELS_DIR)/bs_roformer_vocals.onnx

_models-roformer-mb:
	@mkdir -p $(STAGING_DIR)/melband $(STAGING_DIR)/models/bs_roformer $(MODELS_DIR)
	curl -fSL -o $(STAGING_DIR)/melband/checkpoint.ckpt "$(HF_MELBAND_CKPT)"
	curl -fSL -o $(STAGING_DIR)/melband/config.yaml "$(HF_MELBAND_CFG)"
	curl -fSL -o $(STAGING_DIR)/melband/mel_band_roformer.py "$(HF_MELBAND_PY)"
	curl -fSL -o $(STAGING_DIR)/models/bs_roformer/attend.py \
		"https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/models/bs_roformer/attend.py"
	touch $(STAGING_DIR)/models/__init__.py $(STAGING_DIR)/models/bs_roformer/__init__.py
	PYTHONPATH=$(STAGING_DIR):$$PYTHONPATH $(PYTHON) scripts/export_onnx.py \
		--model-type mel_band_roformer \
		--checkpoint $(STAGING_DIR)/melband/checkpoint.ckpt \
		--config $(STAGING_DIR)/melband/config.yaml \
		--model-code $(STAGING_DIR)/melband/mel_band_roformer.py \
		--chunk-size 352800 \
		--output $(MODELS_DIR)/melband_roformer_vocals.onnx

ort:
	@if [ "$(ORT_PREFIX)" = "/usr/local" ] && [ ! -d "ort/$(ORT_EXTRACT_DIR)" ]; then \
		echo "Downloading $(ORT_ASSET)..."; \
		mkdir -p ort && cd ort \
		&& curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/$(ORT_ASSET)" -o ort-dl \
		&& if echo "$(ORT_ASSET)" | grep -q '\.zip$$'; then unzip -q ort-dl; else tar xzf ort-dl; fi \
		&& rm ort-dl; \
		echo "ORT downloaded to ort/$(ORT_EXTRACT_DIR)"; \
	fi

dist: ort
	$(MAKE) _dist ORT_PREFIX=$(ORT_PREFIX) CMAKE_EXTRA='$(CMAKE_EXTRA)'

_dist: plugin
	rm -rf $(DIST_DIR)
	mkdir -p $(DIST_DIR)
	cp $(BUILD_DIR)/reaper_stem_separation_plugin$(PLUGIN_EXT) $(DIST_DIR)/
	cp LICENSE $(DIST_DIR)/
	@echo "Output: $(DIST_DIR)/reaper_stem_separation_plugin$(PLUGIN_EXT)"

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR) ort $(STAGING_DIR)

help:
	@echo "Targets:"
	@echo "  plugin    Build reaper_stem_separation_plugin (default)"
	@echo "  models    Convert all models via Docker (Demucs + RoFormer)"
	@echo "  ort       Download ONNX Runtime headers for compilation"
	@echo "  dist      Build plugin binary into dist/"
	@echo "  clean     Remove build/dist/staging directories"
	@echo ""
	@echo "Variables:"
	@echo "  ORT_PREFIX    Path to ONNX Runtime install (default: /usr/local)"
