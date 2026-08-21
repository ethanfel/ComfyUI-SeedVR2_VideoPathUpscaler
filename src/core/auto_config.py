"""Hardware-aware SeedVR2 configuration recommendations."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


MODEL_MEMORY_GB = {
    "seedvr2_ema_7b_fp16.safetensors": 17.0,
    "seedvr2_7b_int8_convrot.safetensors": 9.0,
    "seedvr2_7b_nvfp4.safetensors": 5.5,
    "seedvr2_3b_int8_convrot.safetensors": 4.0,
    "seedvr2_3b_nvfp4.safetensors": 2.5,
    "seedvr2_ema_3b-Q4_K_M.gguf": 2.2,
}


def _floor_4n_plus_1(value: int) -> int:
    """Largest positive 4n+1 value no greater than ``value``."""
    if value < 5:
        return 1
    return 1 + 4 * ((value - 1) // 4)


def _ceil_4n_plus_1(value: int) -> int:
    """Smallest positive 4n+1 value no smaller than ``value``."""
    if value <= 1:
        return 1
    return 1 + 4 * math.ceil((value - 1) / 4)


def _select_model(profile: str, budget_gb: float, blackwell: bool) -> Tuple[str, str]:
    """Select a model and a short rationale from the usable VRAM budget."""
    if profile == "maximum_quality":
        if budget_gb >= 24:
            return "seedvr2_ema_7b_fp16.safetensors", "7B FP16 for maximum fidelity"
        if budget_gb >= 13:
            return "seedvr2_7b_int8_convrot.safetensors", "7B INT8 ConvRot to retain 7B quality"
        return "seedvr2_3b_int8_convrot.safetensors", "3B INT8 ConvRot fits the available VRAM"

    if profile == "maximum_throughput":
        if blackwell and budget_gb >= 8:
            return "seedvr2_3b_nvfp4.safetensors", "3B NVFP4 for Blackwell throughput"
        if budget_gb >= 8:
            return "seedvr2_3b_int8_convrot.safetensors", "3B INT8 ConvRot for throughput"
        return "seedvr2_ema_3b-Q4_K_M.gguf", "3B Q4 GGUF for the constrained VRAM budget"

    # Balanced favors 7B, but leaves more activation headroom than maximum quality.
    if budget_gb >= 38:
        return "seedvr2_ema_7b_fp16.safetensors", "7B FP16 with ample activation headroom"
    if budget_gb >= 18:
        return "seedvr2_7b_int8_convrot.safetensors", "7B INT8 ConvRot balances quality and VRAM"
    if budget_gb >= 10:
        return "seedvr2_3b_int8_convrot.safetensors", "3B INT8 ConvRot balances quality and speed"
    return "seedvr2_ema_3b-Q4_K_M.gguf", "3B Q4 GGUF fits the available VRAM"


def recommend_seedvr2_config(
    *,
    total_vram_gb: float,
    free_vram_gb: float,
    compute_capability: Tuple[int, int] = (0, 0),
    target_resolution: int = 1080,
    profile: str = "balanced",
    reserve_vram_gb: float = 4.0,
    frame_count: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    kitchen_attention_available: bool = False,
    enable_torch_compile: bool = False,
) -> Dict[str, Any]:
    """Build model and runtime recommendations from hardware and workload metadata."""
    if profile not in {"balanced", "maximum_quality", "maximum_throughput"}:
        raise ValueError(f"Unknown auto-configuration profile: {profile}")
    if total_vram_gb <= 0:
        raise ValueError("Automatic configuration requires a GPU with measurable memory")

    reserve_vram_gb = max(0.0, reserve_vram_gb)
    total_budget = max(1.0, total_vram_gb - reserve_vram_gb)
    free_budget = max(1.0, free_vram_gb - reserve_vram_gb)
    budget_gb = min(total_budget, free_budget)
    blackwell = compute_capability[0] >= 12
    model, model_reason = _select_model(profile, budget_gb, blackwell)
    model_memory = MODEL_MEMORY_GB[model]

    resolution_scale = max(0.25, (target_resolution / 1080.0) ** 2)
    is_7b = "7b" in model
    # Calibrated against the optimized native pipeline: a 7B FP16 1080p run
    # fits 165 frames at roughly 74% of a 95 GB RTX PRO 6000. Keep a sizeable
    # safety margin over that observed per-frame cost while avoiding the old,
    # overly conservative 33-frame ceiling.
    activation_per_frame = (0.42 if is_7b else 0.28) * resolution_scale
    activation_budget = max(1.0, budget_gb - model_memory - 3.0)
    estimated_frames = max(1, int(activation_budget / activation_per_frame))
    batch_cap = 257 if profile != "maximum_throughput" else 385
    batch_size = _floor_4n_plus_1(min(estimated_frames, batch_cap))

    if batch_size_override is not None:
        if batch_size_override < 1 or (batch_size_override - 1) % 4 != 0:
            raise ValueError("batch_size_override must follow 4n+1: 1, 5, 9, 13, ...")
        batch_size = batch_size_override
    elif frame_count and frame_count > 0 and frame_count < batch_size:
        batch_size = min(batch_size, _ceil_4n_plus_1(frame_count))

    use_tiling = (
        budget_gb < 18
        or (target_resolution > 1440 and budget_gb < 48)
        or (target_resolution > 2160 and budget_gb < 80)
    )
    tile_size = 1536 if budget_gb >= 24 else 1024
    tile_overlap = 128

    # Keeping both models cached on-device eliminates chunk-to-chunk PCIe transfers.
    keep_models_on_gpu = budget_gb >= 24 and not use_tiling
    model_offload_device = "none" if keep_models_on_gpu else "cpu"
    tensor_offload_device = "none" if budget_gb >= 20 else "cpu"

    chunk_multiplier = 6 if profile == "maximum_throughput" else 4
    chunk_size = min(4096, max(batch_size, batch_size * chunk_multiplier))
    if frame_count and 0 < frame_count <= chunk_size:
        chunk_size = 0

    compile_args = None
    if enable_torch_compile:
        compile_args = {
            "backend": "inductor",
            "mode": "reduce-overhead" if profile == "maximum_throughput" else "default",
            "fullgraph": False,
            "dynamic": True,
            "dynamo_cache_size_limit": 64,
            "dynamo_recompile_limit": 128,
        }

    attention_mode = "comfy_kitchen_int8" if kitchen_attention_available else "sdpa"
    blocks_to_swap = 0
    swap_io_components = False
    if budget_gb < 8:
        blocks_to_swap = 16
        swap_io_components = True
        model_offload_device = "cpu"
        keep_models_on_gpu = False

    dit = {
        "model": model,
        "offload_device": model_offload_device,
        "cache_model": True,
        "blocks_to_swap": blocks_to_swap,
        "swap_io_components": swap_io_components,
        "attention_mode": attention_mode,
        "torch_compile_args": compile_args,
    }
    vae = {
        "model": "ema_vae_fp16.safetensors",
        "offload_device": model_offload_device,
        "cache_model": True,
        "encode_tiled": use_tiling,
        "encode_tile_size": tile_size,
        "encode_tile_overlap": tile_overlap,
        "decode_tiled": use_tiling,
        "decode_tile_size": tile_size,
        "decode_tile_overlap": tile_overlap,
        "tile_debug": "false",
        "torch_compile_args": compile_args,
    }
    runtime = {
        "resolution": target_resolution,
        "max_resolution": 0,
        "batch_size": batch_size,
        "uniform_batch_size": True,
        "temporal_overlap": 2 if batch_size >= 5 else 0,
        "prepend_frames": 2 if batch_size >= 5 else 0,
        "color_correction": "lab",
        "input_noise_scale": 0.0,
        "latent_noise_scale": 0.0,
        "offload_device": tensor_offload_device,
        "chunk_size": chunk_size,
        "chunk_overlap": 2 if chunk_size != 0 else 0,
        "temporary_video_crf": 16 if profile == "maximum_quality" else 18,
    }

    return {
        "dit": dit,
        "vae": vae,
        "runtime": runtime,
        "hardware": {
            "total_vram_gb": total_vram_gb,
            "free_vram_gb": free_vram_gb,
            "budget_gb": budget_gb,
            "compute_capability": compute_capability,
            "blackwell": blackwell,
            "keep_models_on_gpu": keep_models_on_gpu,
        },
        "batch_size_overridden": batch_size_override is not None,
        "model_reason": model_reason,
    }


__all__ = ["recommend_seedvr2_config"]
