"""Data loaders for hardware and model presets."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .models import HardwareSpec, ModelPreset


def _get_data_dir() -> Path:
    """Get the data directory path."""
    # Try package data first
    package_data = Path(__file__).parent.parent.parent / "data"
    if package_data.exists():
        return package_data

    # Fall back to current working directory
    cwd_data = Path.cwd() / "data"
    if cwd_data.exists():
        return cwd_data

    raise FileNotFoundError(
        "Data directory not found. Expected at package/data or ./data"
    )


@lru_cache(maxsize=1)
def load_hardware() -> list[HardwareSpec]:
    """Load all hardware specifications from hardware.json."""
    data_dir = _get_data_dir()
    hardware_file = data_dir / "hardware.json"

    with open(hardware_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [HardwareSpec(**hw) for hw in data["hardware"]]


@lru_cache(maxsize=1)
def load_presets() -> list[ModelPreset]:
    """Load all model presets from presets.json."""
    data_dir = _get_data_dir()
    presets_file = data_dir / "presets.json"

    with open(presets_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [ModelPreset(**preset) for preset in data["presets"]]


def get_hardware(name: str) -> Optional[HardwareSpec]:
    """Get hardware specification by name (case-insensitive)."""
    name_lower = name.lower()
    for hw in load_hardware():
        if hw.name.lower() == name_lower:
            return hw
    return None


def get_preset(name: str) -> Optional[ModelPreset]:
    """Get model preset by name (case-insensitive)."""
    name_lower = name.lower()
    for preset in load_presets():
        if preset.name.lower() == name_lower:
            return preset
    return None


def list_hardware_names() -> list[str]:
    """List all available hardware names."""
    return [hw.name for hw in load_hardware()]


def list_preset_names() -> list[str]:
    """List all available model preset names."""
    return [preset.name for preset in load_presets()]
