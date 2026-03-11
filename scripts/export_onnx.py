#!/usr/bin/env python3
"""Export BS-RoFormer and MelBand-RoFormer models to full-model ONNX files.

The exported ONNX models accept raw waveform input (1, 2, samples) and output
separated stems (1, num_stems, 2, samples), with STFT/ISTFT baked in.

Usage:
    # BS-RoFormer HyperACE (vocals) -- uses custom model code from HuggingFace
    python scripts/export_onnx.py \
        --model-type bs_roformer \
        --checkpoint models/staging/bs_hyperace/checkpoint.ckpt \
        --config models/staging/bs_hyperace/config.yaml \
        --model-code models/staging/bs_hyperace/bs_roformer.py \
        --output models/bs_roformer_vocals.onnx

    # MelBand-RoFormer (vocals) -- uses pip bs-roformer package
    python scripts/export_onnx.py \
        --model-type mel_band_roformer \
        --checkpoint models/staging/melband/checkpoint.ckpt \
        --config models/staging/melband/config.yaml \
        --output models/melband_roformer_vocals.onnx
"""

import argparse
import importlib.util
import sys

import torch
import yaml


def load_model(model_type: str, config_path: str, checkpoint_path: str,
               model_code_path: str | None = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", config)

    if model_code_path:
        spec = importlib.util.spec_from_file_location("custom_model", model_code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if model_type == "bs_roformer":
            model = module.BSRoformer(**model_config)
        elif model_type == "mel_band_roformer":
            model = module.MelBandRoformer(**model_config)
        else:
            print(f"Unknown model type: {model_type}", file=sys.stderr)
            sys.exit(1)
    else:
        if model_type == "bs_roformer":
            from bs_roformer import BSRoformer
            model = BSRoformer(**model_config)
        elif model_type == "mel_band_roformer":
            from bs_roformer import MelBandRoformer
            model = MelBandRoformer(**model_config)
        else:
            print(f"Unknown model type: {model_type}", file=sys.stderr)
            sys.exit(1)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(model, output_path: str, sample_rate: int = 44100,
                duration_secs: float = 8.0):
    num_samples = int(sample_rate * duration_secs)
    dummy_input = torch.randn(1, 2, num_samples)

    print(f"Tracing model with input shape {list(dummy_input.shape)}...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            input_names=["audio"],
            output_names=["stems"],
            dynamic_axes={
                "audio": {2: "samples"},
                "stems": {3: "samples"},
            },
        )

    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export RoFormer model to ONNX")
    parser.add_argument("--model-type", required=True,
                        choices=["bs_roformer", "mel_band_roformer"])
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--model-code", default=None,
                        help="Path to custom model .py file (overrides pip package)")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    model = load_model(args.model_type, args.config, args.checkpoint, args.model_code)
    export_onnx(model, args.output, sample_rate=args.sample_rate)


if __name__ == "__main__":
    main()
