"""
Hardware abstraction layer for cross-platform model training and inference.

This module encapsulates ALL device-specific logic so that the rest of the
codebase can be device-agnostic. The training loop, model, and inference
code never need to check what hardware they're running on — they just call
these functions.

SUPPORTED DEVICES:
  1. CUDA (NVIDIA GPUs): Fastest option. Supports all features.
     - A100/H100 (Ampere/Hopper): bfloat16, FlashAttention, torch.compile
     - T4/V100 (Turing/Volta): float16 with GradScaler, SDPA attention

  2. MPS (Apple Silicon): Good performance on Mac.
     - M1/M2/M3/M4: float32 params, float16 autocast for compute
     - torch.compile NOT supported as of PyTorch 2.x
     - Some operations may silently fall back to CPU

  3. CPU: Fallback. Slowest but always available.
     - float32 only (no mixed precision)
     - Useful for debugging and small experiments

WHY THIS ABSTRACTION IS NEEDED:
  Different devices have different capabilities:
  - Supported dtypes: CUDA supports bf16/fp16/fp32; MPS has quirks with fp16
  - Autocast API: Different device_type strings ("cuda", "mps", "cpu")
  - GradScaler: Needed for fp16 but not bf16 or fp32
  - torch.compile: Only works on CUDA
  - Memory reporting: Different APIs per device

  Without this module, every piece of code would need device-specific branches.
"""

import torch
import torch.amp
from contextlib import nullcontext
from typing import Optional


def get_device() -> torch.device:
    """
    Auto-detect the best available compute device.

    Priority order: CUDA → MPS → CPU

    CUDA is preferred because:
      - Highest throughput for matrix operations
      - Best mixed-precision support
      - FlashAttention and torch.compile available

    MPS (Metal Performance Shaders) is second because:
      - Apple's GPU compute framework for M-series chips
      - Good performance for our model size (~15M params)
      - Available on all modern Macs without any setup

    CPU is the fallback:
      - Always available
      - Sufficient for debugging and tiny experiments
      - Our 15M model trains in ~hours even on CPU

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_dtype(requested: str, device: torch.device) -> torch.dtype:
    """
    Resolve a dtype string to the optimal torch.dtype for the given device.

    The "auto" option selects the best dtype based on hardware capabilities:

    BFLOAT16 (Brain Floating Point):
      - 1 sign bit, 8 exponent bits, 7 mantissa bits
      - SAME dynamic range as float32 (8 exponent bits)
      - LESS precision than float16 (7 vs 10 mantissa bits)
      - Advantage: No GradScaler needed! Values don't underflow/overflow
        because the exponent range matches fp32.
      - Available on: CUDA Ampere+ (A100, RTX 3090+), recent Intel CPUs
      - NOT available on: T4, V100, most MPS devices

    FLOAT16 (Half Precision):
      - 1 sign bit, 5 exponent bits, 10 mantissa bits
      - SMALLER dynamic range than float32 (5 vs 8 exponent bits)
      - Requires GradScaler to prevent gradient underflow
      - Available everywhere CUDA is available

    FLOAT32 (Single Precision):
      - The "safe" default. No precision issues.
      - 2× memory and ~2× slower than fp16/bf16
      - Used on CPU and as fallback

    Args:
        requested: One of "auto", "float16", "bfloat16", "float32".
        device: The target device (affects what dtypes are available).

    Returns:
        torch.dtype: The resolved dtype.
    """
    if requested == "auto":
        if device.type == "cuda":
            # Check if the GPU supports bfloat16 natively.
            # torch.cuda.is_bf16_supported() checks for Ampere+ architecture.
            # Ampere (compute capability 8.0+) includes: A100, A10, RTX 3090, etc.
            # Turing (7.5) = T4, RTX 2080 → does NOT support bf16 natively.
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16
        elif device.type == "mps":
            # MPS (Apple Silicon) mixed precision:
            # Parameters are stored in float32, but autocast can use float16
            # for compute. We return float32 here as the "storage" dtype.
            # The autocast context manager handles the actual precision switching.
            # NOTE: bfloat16 support on MPS is experimental/limited as of 2024.
            return torch.float32
        else:
            # CPU: float32 is the only practical option.
            # While PyTorch supports bf16 on some CPUs (Intel AMX), the
            # performance benefit is minimal compared to GPU mixed precision.
            return torch.float32
    else:
        # User explicitly requested a dtype — respect their choice.
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if requested not in dtype_map:
            raise ValueError(
                f"Unknown dtype '{requested}'. "
                f"Choose from: {list(dtype_map.keys())} or 'auto'"
            )
        return dtype_map[requested]


def get_autocast_context(
    device: torch.device, dtype: torch.dtype
):
    """
    Return the appropriate automatic mixed precision (AMP) context manager.

    WHAT IS AUTOCAST?
      torch.amp.autocast automatically selects the precision for each
      operation. Operations that benefit from lower precision (like matmul)
      run in fp16/bf16, while operations that need higher precision (like
      softmax, layer norm) stay in fp32.

      This gives you most of the speed/memory benefit of lower precision
      without manually casting every tensor.

    WHEN TO USE:
      - CUDA + bf16/fp16: Use autocast for ~2× speedup and ~50% memory savings
      - MPS + fp32 storage: Use autocast with fp16 for compute speedup
      - CPU + fp32: No autocast needed (returns nullcontext)

    Args:
        device: The compute device.
        dtype: The target dtype (from get_dtype).

    Returns:
        A context manager. Usage: `with get_autocast_context(device, dtype):`
    """
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        # CUDA autocast: handles matmul, conv, linear in lower precision
        # while keeping reductions (softmax, layernorm) in fp32.
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    elif device.type == "mps":
        # MPS autocast: relatively new, uses float16 for eligible operations.
        # Even though params are fp32, autocast casts inputs on-the-fly.
        # If this causes numerical issues, disable by returning nullcontext().
        return torch.amp.autocast(device_type="mps", dtype=torch.float16)
    else:
        # CPU or fp32: No autocast needed. Return a no-op context manager.
        return nullcontext()


def get_grad_scaler(
    device: torch.device, dtype: torch.dtype
) -> Optional[torch.amp.GradScaler]:
    """
    Create a GradScaler if needed for the device/dtype combination.

    WHAT IS GRADSCALER?
      When training in float16, gradient values can be very small (near the
      minimum representable fp16 value ≈ 6e-8). These small gradients
      "underflow" to zero, causing the model to stop learning.

      GradScaler prevents this by:
        1. Scaling the loss UP by a large factor before .backward()
        2. This makes all gradients proportionally larger (above underflow range)
        3. Before optimizer.step(), it scales gradients back DOWN
        4. If gradients overflow (become inf), it skips the step and reduces
           the scale factor for next time.

    WHEN NEEDED:
      - float16 on CUDA: YES (fp16 has limited dynamic range)
      - bfloat16 on CUDA: NO (bf16 has same exponent range as fp32)
      - float32 anywhere: NO (full precision, no underflow risk)
      - MPS with autocast: YES if using fp16 autocast

    Args:
        device: The compute device.
        dtype: The training dtype.

    Returns:
        GradScaler if needed, None otherwise.
    """
    if dtype == torch.float16:
        if device.type == "cuda":
            return torch.amp.GradScaler(device="cuda")
        elif device.type == "mps":
            # MPS GradScaler support was added in recent PyTorch versions.
            # If it's not available, return None (training will work but
            # might have occasional precision issues).
            try:
                return torch.amp.GradScaler(device="mps")
            except Exception:
                print(
                    "WARNING: GradScaler not supported on MPS. "
                    "Training without loss scaling."
                )
                return None
    # bfloat16 and float32 don't need scaling
    return None


def device_info(device: torch.device) -> str:
    """
    Pretty-print device capabilities for logging at training start.

    This is printed once at the beginning of training so the user knows
    exactly what hardware configuration is being used.

    Returns:
        A human-readable string describing the device.
    """
    lines = [f"Device: {device}"]

    if device.type == "cuda":
        # CUDA provides rich device information
        props = torch.cuda.get_device_properties(device)
        lines.append(f"  GPU: {props.name}")
        lines.append(f"  VRAM: {props.total_mem / 1024**3:.1f} GB")
        lines.append(f"  Compute Capability: {props.major}.{props.minor}")
        lines.append(f"  BF16 Support: {torch.cuda.is_bf16_supported()}")
        lines.append(f"  CUDA Version: {torch.version.cuda}")
    elif device.type == "mps":
        lines.append("  Backend: Metal Performance Shaders (Apple Silicon)")
        # MPS memory info (available in recent PyTorch)
        try:
            allocated = torch.mps.driver_allocated_memory() / 1024**3
            lines.append(f"  GPU Memory Allocated: {allocated:.2f} GB")
        except AttributeError:
            lines.append("  GPU Memory: (info not available)")
    else:
        lines.append("  Backend: CPU (no GPU acceleration)")

    lines.append(f"  PyTorch Version: {torch.__version__}")

    return "\n".join(lines)


def get_memory_usage(device: torch.device) -> dict:
    """
    Get current memory usage for logging during training.

    Returns a dict with 'allocated' and 'reserved' in MB.
    'allocated': Memory actively used by tensors.
    'reserved': Memory held by the caching allocator (includes allocated + free cache).

    The difference (reserved - allocated) is memory that PyTorch has claimed
    from the OS but isn't currently using. This is normal — the caching
    allocator keeps freed memory to avoid expensive OS allocation calls.
    """
    if device.type == "cuda":
        return {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        }
    elif device.type == "mps":
        try:
            return {
                "allocated_mb": torch.mps.driver_allocated_memory() / 1024**2,
                "reserved_mb": torch.mps.driver_allocated_memory() / 1024**2,
            }
        except AttributeError:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    else:
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
