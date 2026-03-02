BUILD_DIR   := build
DIST_DIR    := dist
VENDOR_DIR  := vendor/demucs.onnx
MODELS_DIR  := models
PYTHON      ?= python3
ORT_VERSION ?= 1.19.2
PROVIDER    ?= cpu

# Auto-detect platform
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
UNAME_M := $(shell uname -m 2>/dev/null || echo x86_64)

ifeq ($(UNAME_S),Darwin)
    PLATFORM      := macos
    ARCH          := $(if $(filter arm64,$(UNAME_M)),arm64,x64)
    PLUGIN_EXT    := .dylib
    ORT_LIB_GLOB  := libonnxruntime*.dylib
    ARCHIVE_FMT   := tar.gz
    ORT_ASSET     := onnxruntime-osx-$(ARCH)-$(ORT_VERSION).tgz
else ifeq ($(UNAME_S),Linux)
    PLATFORM      := linux
    ARCH          := x64
    PLUGIN_EXT    := .so
    ORT_LIB_GLOB  := libonnxruntime*.so*
    ARCHIVE_FMT   := tar.gz
    ifeq ($(PROVIDER),cuda)
        ORT_ASSET := onnxruntime-linux-x64-gpu-$(ORT_VERSION).tgz
    else
        ORT_ASSET := onnxruntime-linux-x64-$(ORT_VERSION).tgz
    endif
else
    PLATFORM      := windows
    ARCH          := x64
    PLUGIN_EXT    := .dll
    ORT_LIB_GLOB  := onnxruntime*.dll
    ARCHIVE_FMT   := zip
    ifeq ($(PROVIDER),cuda)
        ORT_ASSET := onnxruntime-win-x64-gpu-$(ORT_VERSION).zip
    else
        ORT_ASSET := onnxruntime-win-x64-$(ORT_VERSION).zip
    endif
endif

DIST_NAME := reaper-source-separation-$(PLATFORM)-$(ARCH)-$(PROVIDER)
ORT_EXTRACT_DIR := $(subst .tgz,,$(subst .zip,,$(ORT_ASSET)))

# Auto-download ORT into ort/ if ORT_PREFIX not explicitly set
ORT_PREFIX ?= $(shell ls -d ort/$(ORT_EXTRACT_DIR) 2>/dev/null)
ifeq ($(ORT_PREFIX),)
    ORT_PREFIX := /usr/local
endif

CMAKE_EXTRA ?=

.PHONY: all plugin models models-4s models-6s ort dist _dist clean help

all: plugin

plugin:
	cmake -S . -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Release -DORT_PREFIX=$(ORT_PREFIX) $(CMAKE_EXTRA)
	cmake --build $(BUILD_DIR) --config Release
	@echo "Built: $(BUILD_DIR)/reaper_source_separation$(PLUGIN_EXT)"

models: models-4s models-6s

models-4s:
	@mkdir -p $(MODELS_DIR)
	$(PYTHON) $(VENDOR_DIR)/scripts/convert-pth-to-onnx.py $(MODELS_DIR)
	$(PYTHON) -m onnxruntime.tools.convert_onnx_models_to_ort $(MODELS_DIR)

models-6s:
	@mkdir -p $(MODELS_DIR)
	$(PYTHON) $(VENDOR_DIR)/scripts/convert-pth-to-onnx.py $(MODELS_DIR) --six-source
	$(PYTHON) -m onnxruntime.tools.convert_onnx_models_to_ort $(MODELS_DIR)

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
	$(MAKE) _dist ORT_PREFIX=$(ORT_PREFIX) PROVIDER=$(PROVIDER) CMAKE_EXTRA='$(CMAKE_EXTRA)'

_dist: plugin
	rm -rf $(DIST_DIR)
	mkdir -p $(DIST_DIR)/reaper-source-separation/models
	cp $(BUILD_DIR)/reaper_source_separation$(PLUGIN_EXT) $(DIST_DIR)/
	cp $(ORT_PREFIX)/lib/$(ORT_LIB_GLOB) $(DIST_DIR)/reaper-source-separation/ 2>/dev/null || true
	@# Include local models if they exist (local dev); CI has none, so this is a no-op there
	cp $(MODELS_DIR)/*.ort $(DIST_DIR)/reaper-source-separation/models/ 2>/dev/null || true
ifeq ($(UNAME_S),Linux)
	cd $(DIST_DIR)/reaper-source-separation && \
		for f in libonnxruntime.so.*.*.*; do \
			major=$$(echo $$f | sed 's/libonnxruntime\.so\.//;s/\..*//' ); \
			ln -sf $$f libonnxruntime.so.$$major; \
			ln -sf libonnxruntime.so.$$major libonnxruntime.so; \
		done 2>/dev/null || true
endif
ifeq ($(ARCHIVE_FMT),tar.gz)
	cd $(DIST_DIR) && tar czf ../$(DIST_NAME).tar.gz .
else
	cd $(DIST_DIR) && 7z a -tzip ../$(DIST_NAME).zip .
endif
	@echo "Packaged: $(DIST_NAME).$(ARCHIVE_FMT)"

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR) ort reaper-source-separation-*.tar.gz reaper-source-separation-*.zip

help:
	@echo "Targets:"
	@echo "  plugin        Build reaper_source_separation (default)"
	@echo "  models        Convert all PyTorch models to ORT format (requires Python)"
	@echo "  models-4s     Convert 4-stem model only"
	@echo "  models-6s     Convert 6-stem model only"
	@echo "  dist          Build + package tarball/zip with ORT libs (+ local models if present)"
	@echo "  clean         Remove build/dist directories"
	@echo ""
	@echo "Variables:"
	@echo "  ORT_PREFIX    Path to ONNX Runtime install (default: /usr/local)"
	@echo "  PROVIDER      Execution provider tag for archive name: cpu or cuda (default: cpu)"
	@echo "  PYTHON        Python interpreter (default: python3)"
