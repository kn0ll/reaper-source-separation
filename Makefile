BUILD_DIR   := build
DIST_DIR    := dist
VENDOR_DIR  := vendor/demucs.onnx
MODELS_DIR  := models
PYTHON      ?= python3
ORT_PREFIX  ?= /usr/local

# Auto-detect platform for dist
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
UNAME_M := $(shell uname -m 2>/dev/null || echo x86_64)

ifeq ($(UNAME_S),Darwin)
    PLATFORM      := macos
    ARCH          := $(if $(filter arm64,$(UNAME_M)),arm64,x64)
    PLUGIN_EXT    := .dylib
    ORT_LIB_GLOB  := libonnxruntime*.dylib
    ARCHIVE_FMT   := tar.gz
else ifeq ($(UNAME_S),Linux)
    PLATFORM      := linux
    ARCH          := x64
    PLUGIN_EXT    := .so
    ORT_LIB_GLOB  := libonnxruntime*.so*
    ARCHIVE_FMT   := tar.gz
else
    PLATFORM      := windows
    ARCH          := x64
    PLUGIN_EXT    := .dll
    ORT_LIB_GLOB  := onnxruntime*.dll
    ARCHIVE_FMT   := zip
endif

DIST_NAME := reaper-source-separation-$(PLATFORM)-$(ARCH)

CMAKE_EXTRA ?=
ifneq ($(ORT_PREFIX),/usr/local)
    CMAKE_EXTRA += -DORT_PREFIX=$(ORT_PREFIX)
endif

.PHONY: all plugin models models-4s models-6s dist clean help

all: plugin

plugin:
	cmake -S . -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Release $(CMAKE_EXTRA)
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

dist: plugin
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
	rm -rf $(BUILD_DIR) $(DIST_DIR) reaper-source-separation-*.tar.gz reaper-source-separation-*.zip

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
	@echo "  PYTHON        Python interpreter (default: python3)"
