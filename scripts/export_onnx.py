#!/usr/bin/env python3
"""Export RoFormer models to ONNX with STFT/ISTFT stripped out.

The exported ONNX models operate on real-valued STFT representations:

  Input:  stft_real   (batch, channels, freq_bins, time_frames, 2)
  Output: masked_stft (batch, num_stems, channels, freq_bins, time_frames, 2)

STFT and ISTFT are handled by the C++ runtime.  This avoids the well-known
ONNX limitation that complex tensors and torch.stft/istft cannot be exported.

Usage:
    python scripts/export_onnx.py \
        --model-type bs_roformer \
        --checkpoint path/to/checkpoint.ckpt \
        --config path/to/config.yaml \
        --model-code path/to/bs_roformer.py \
        --chunk-size 352800 \
        --output models/bs_roformer_vocals.onnx

    python scripts/export_onnx.py \
        --model-type mel_band_roformer \
        --checkpoint path/to/checkpoint.ckpt \
        --config path/to/config.yaml \
        --chunk-size 352800 \
        --output models/melband_roformer_vocals.onnx
"""

import argparse
import importlib.util
import sys

import torch
import torch.nn as nn
import yaml
from einops import rearrange, pack, unpack


# ---------------------------------------------------------------------------
# Core wrappers: strip STFT/ISTFT, keep only the neural-network core
# ---------------------------------------------------------------------------


class BSRoformerCore(nn.Module):
    """Wraps a BSRoformer model (custom HyperACE variant).

    The custom HyperACE transformers have a simple forward(x) signature
    (no value_residual / hyper-connections).
    """

    def __init__(self, model):
        super().__init__()
        self.band_split = model.band_split
        self.layers = model.layers
        self.final_norm = model.final_norm
        self.mask_estimators = model.mask_estimators
        self.audio_channels = model.audio_channels

    def forward(self, stft_real):
        # stft_real: (batch, channels, freq, time, 2)
        stft_repr = rearrange(stft_real, "b s f t c -> b (f s) t c")

        x = rearrange(stft_repr, "b f t c -> b t (f c)")
        x = self.band_split(x)

        for transformer_block in self.layers:
            time_transformer, freq_transformer = transformer_block

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)
            x, = unpack(x, ps, "* t d")

            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            x, = unpack(x, ps, "* f d")

        x = self.final_norm(x)

        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)

        # Complex multiply in real arithmetic:  (a+bi)(c+di) = (ac-bd)+(ad+bc)i
        stft_5d = rearrange(stft_repr, "b f t c -> b 1 f t c")
        sr, si = stft_5d[..., 0], stft_5d[..., 1]
        mr, mi = mask[..., 0], mask[..., 1]
        real = sr * mr - si * mi
        imag = sr * mi + si * mr
        masked = torch.stack([real, imag], dim=-1)

        # (batch, stems, freq*ch, time, 2) -> (batch, stems, ch, freq, time, 2)
        masked = rearrange(
            masked, "b n (f s) t c -> b n s f t c", s=self.audio_channels
        )
        return masked


class MelBandRoformerCore(nn.Module):
    """Wraps a ZFTurbo-style MelBandRoformer model (strips STFT/ISTFT).

    ZFTurbo's Transformer uses a simple forward(x) -> x signature
    (no value_residual / hyper-connections).  Layer blocks may be either
    2-tuple (time, freq) or 3-tuple (linear, time, freq).
    """

    def __init__(self, model):
        super().__init__()
        self.band_split = model.band_split
        self.layers = model.layers
        self.mask_estimators = model.mask_estimators
        self.audio_channels = model.audio_channels
        self.register_buffer("freq_indices", model.freq_indices)
        self.register_buffer("num_bands_per_freq", model.num_bands_per_freq)

    def forward(self, stft_real):
        # stft_real: (batch, channels, freq, time, 2)
        batch = stft_real.shape[0]
        channels = self.audio_channels

        stft_repr = rearrange(stft_real, "b s f t c -> b (f s) t c")

        batch_arange = torch.arange(batch, device=stft_real.device).unsqueeze(1)
        x = stft_repr[batch_arange, self.freq_indices]

        x = rearrange(x, "b f t c -> b t (f c)")
        x = self.band_split(x)

        for layer_block in self.layers:
            if len(layer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = layer_block
                x, ft_ps = pack([x], "b * d")
                x = linear_transformer(x)
                x, = unpack(x, ft_ps, "b * d")
            else:
                time_transformer, freq_transformer = layer_block

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)
            x, = unpack(x, ps, "* t d")

            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            x, = unpack(x, ps, "* f d")

        num_stems = len(self.mask_estimators)
        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        # Scatter masks from mel-band space back to full frequency space
        stft_5d = rearrange(stft_repr, "b f t c -> b 1 f t c")
        stft_expanded = stft_5d.expand(-1, num_stems, -1, -1, -1)

        t = stft_repr.shape[2]
        scatter_idx = self.freq_indices.reshape(1, 1, -1, 1, 1).expand(
            batch, num_stems, -1, t, 2
        )

        masks_summed = torch.zeros_like(stft_expanded).scatter_add(
            2, scatter_idx, masks
        )

        denom = self.num_bands_per_freq.float()
        denom = (
            denom.unsqueeze(1)
            .expand(-1, channels)
            .reshape(1, 1, -1, 1, 1)
            .clamp(min=1e-8)
        )
        masks_averaged = masks_summed / denom

        # Complex multiply in real arithmetic
        sr, si = stft_expanded[..., 0], stft_expanded[..., 1]
        mr, mi = masks_averaged[..., 0], masks_averaged[..., 1]
        real = sr * mr - si * mi
        imag = sr * mi + si * mr
        masked = torch.stack([real, imag], dim=-1)

        masked = rearrange(
            masked, "b n (f s) t c -> b n s f t c", s=channels
        )
        return masked


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_type, config_path, checkpoint_path, model_code_path=None):
    with open(config_path) as f:
        config = yaml.full_load(f)

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


def read_stft_params(config_path):
    """Extract STFT parameters from the model config."""
    with open(config_path) as f:
        config = yaml.full_load(f)
    mc = config.get("model", config)
    return {
        "n_fft": mc.get("stft_n_fft", 2048),
        "hop_length": mc.get("stft_hop_length", 512),
        "win_length": mc.get("stft_win_length", 2048),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_onnx(core_model, output_path, stft_params, chunk_size):
    n_fft = stft_params["n_fft"]
    hop = stft_params["hop_length"]
    freq_bins = n_fft // 2 + 1
    # time_frames for a chunk with center-padding
    time_frames = chunk_size // hop + 1

    dummy = torch.randn(1, 2, freq_bins, time_frames, 2)

    print(
        f"Tracing core model with stft_real shape {list(dummy.shape)} "
        f"(chunk_size={chunk_size}, freq_bins={freq_bins}, time_frames={time_frames})..."
    )

    with torch.no_grad():
        torch.onnx.export(
            core_model,
            dummy,
            output_path,
            export_params=True,
            opset_version=17,
            input_names=["stft_real"],
            output_names=["masked_stft"],
            dynamic_axes={
                "stft_real": {3: "time_frames"},
                "masked_stft": {4: "time_frames"},
            },
            dynamo=False,
        )

    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export RoFormer model core (no STFT/ISTFT) to ONNX"
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["bs_roformer", "mel_band_roformer"],
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--model-code",
        default=None,
        help="Path to custom model .py file (overrides pip package)",
    )
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
        help="Audio chunk size in samples (for computing trace dimensions)",
    )
    args = parser.parse_args()

    model = load_model(args.model_type, args.config, args.checkpoint, args.model_code)
    stft_params = read_stft_params(args.config)

    if args.model_type == "bs_roformer":
        core = BSRoformerCore(model)
    else:
        core = MelBandRoformerCore(model)
    core.eval()

    export_onnx(core, args.output, stft_params, args.chunk_size)


if __name__ == "__main__":
    main()
