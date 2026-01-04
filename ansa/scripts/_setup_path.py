"""Add src/ to Python path for all scripts."""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))