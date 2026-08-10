"""Explicit Qwen3-VL registration for LaMP."""

from .QWen3 import QWEN3_VL_REVISION, Qwen3VLInterface


def get_vlm_model(config):
    return Qwen3VLInterface(config)


__all__ = ["QWEN3_VL_REVISION", "Qwen3VLInterface", "get_vlm_model"]
