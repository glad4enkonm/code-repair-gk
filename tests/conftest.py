import sys
from pathlib import Path

# Make src/ importable for running tests without pip install -e .
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
