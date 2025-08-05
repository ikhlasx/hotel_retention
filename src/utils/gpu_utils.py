"""Utilities for detecting GPU capabilities."""

from typing import List, Tuple


def detect_gpu_capability() -> Tuple[bool, List[str]]:
    """Detect CUDA-capable GPUs and return their names.

    The function checks for GPU support using commonly available libraries.
    It first attempts to query ``torch.cuda`` to enumerate available devices
    and obtain their names. If PyTorch is not available or does not detect
    any GPUs, it falls back to checking ``onnxruntime`` providers. Any import
    errors or runtime issues are swallowed so that the function safely
    reports no GPU support when capability cannot be determined.

    Returns:
        Tuple[bool, List[str]]: ``(available, names)`` where ``available`` is
        ``True`` if a CUDA execution provider or CUDA-enabled Torch runtime is
        available, and ``names`` is a list of detected GPU names. The list may
        be empty if device names cannot be determined.
    """

    available = False
    device_names: List[str] = []

    # Prefer PyTorch for detailed device information
    try:
        import torch

        if torch.cuda.is_available():
            available = True
            device_names = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            return available, device_names
    except Exception:
        # torch is not installed or CUDA query failed
        pass

    # Fallback to ONNX Runtime provider check
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            available = True
    except Exception:
        # onnxruntime is not installed or failed to query providers
        pass

    return available, device_names
