---
library_name: lamp-vla
license: other
pipeline_tag: robotics
tags:
  - vision-language-action
  - libero
  - scene-flow
---

# LaMP-LIBERO

Official LaMP policy checkpoint for LIBERO-Spatial, LIBERO-Object, LIBERO-Goal,
and LIBERO-10. It contains Qwen3-VL, Motion Expert, Motion Guidance, and Action
Expert weights.

## Usage

```python
from starVLA import load_policy

policy = load_policy(
    "summerwang11/LaMP-LIBERO",
    device="cuda",
    dtype="bfloat16",
)
```

See the [LaMP repository](https://github.com/SummerWXK/LaMP) for installation,
inference, training, and evaluation instructions.

## Results

| Suite | Successes / episodes |
| --- | ---: |
| LIBERO-Spatial | 497 / 500 |
| LIBERO-Object | 499 / 500 |
| LIBERO-Goal | 487 / 500 |
| LIBERO-10 | 483 / 500 |
| Average | 98.3% |
