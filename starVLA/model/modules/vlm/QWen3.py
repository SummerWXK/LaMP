"""Qwen3-VL adapter used by LaMP."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from starVLA.constants import QWEN3_VL_REVISION

LOGGER = logging.getLogger(__name__)


class Qwen3VLInterface(nn.Module):
    """Keep Qwen preprocessing and model invocation behind one raw-input interface."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        qwen_config = config.framework.qwenvl
        model_id = qwen_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        revision = qwen_config.get("revision", QWEN3_VL_REVISION)
        requested_attention = qwen_config.get("attn_implementation", "flash_attention_2")
        dtype = _torch_dtype(qwen_config.get("dtype", "bfloat16"))

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                attn_implementation=requested_attention,
                torch_dtype=dtype,
            )
            self.attn_implementation = requested_attention
        except (ImportError, RuntimeError, ValueError) as error:
            if requested_attention != "flash_attention_2":
                raise
            LOGGER.warning("FlashAttention 2 is unavailable; falling back to SDPA: %s", error)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                attn_implementation="sdpa",
                torch_dtype=dtype,
            )
            self.attn_implementation = "sdpa"

        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.processor.tokenizer.padding_side = "left"
        self.config = config
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def forward(self, **kwargs: Any) -> Any:
        return self.model(**kwargs)

    def build_qwenvl_inputs(self, images: list[list[Any]], instructions: list[str]) -> Any:
        """Convert raw PIL image lists and instructions to Qwen tensors."""
        if len(images) != len(instructions):
            raise ValueError("images and instructions must have identical batch sizes")

        prompt_template = self.config.datasets.vla_data.get("CoT_prompt")
        messages = []
        for sample_images, instruction in zip(images, instructions, strict=True):
            if not isinstance(instruction, str):
                raise TypeError("each language instruction must be a string")
            content = [{"type": "image", "image": image} for image in sample_images]
            prompt = prompt_template.replace("{instruction}", instruction) if prompt_template else instruction
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        return batch.to(self.model.device)


def _torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    try:
        return getattr(torch, value)
    except AttributeError as error:
        raise ValueError(f"unsupported torch dtype: {value}") from error


_QWen3_VL_Interface = Qwen3VLInterface
