"""Shared pytest fixtures."""

import sys
from pathlib import Path

# Make src/ importable without installation during development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
