"""
Project path setup - ensures project root is importable when running scripts directly.

Usage in run.py:
    import sys
    from pathlib import Path
    exec(open(Path(__file__).resolve().parent.parent / 'core' / 'path_setup.py').read())
"""
import sys
from pathlib import Path

# Project root is two levels up from src/core/path_setup.py -> project root
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))