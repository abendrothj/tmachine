"""
tests/conftest.py

Shared pytest fixtures and markers.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require a CUDA-capable GPU",
    )
