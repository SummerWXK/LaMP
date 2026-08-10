"""Explicit framework registration; LaMP is the only v0.1.0 model."""

from .LaMP import LaMP

FRAMEWORK_REGISTRY = {"LaMP": LaMP}


def build_framework(config):
    try:
        model_class = FRAMEWORK_REGISTRY[config.framework.name]
    except KeyError as error:
        raise NotImplementedError("LaMP v0.1.0 only provides framework.name=LaMP") from error
    return model_class(config)


__all__ = ["FRAMEWORK_REGISTRY", "LaMP", "build_framework"]
