"""Public LaMP interface."""

from typing import Any

__version__ = "0.1.0"


def load_policy(*args: Any, **kwargs: Any):
    """Lazily load a LaMP policy without importing GPU dependencies at package import."""
    from starVLA.checkpoint import load_policy as _load_policy

    return _load_policy(*args, **kwargs)


__all__ = ["__version__", "load_policy"]
