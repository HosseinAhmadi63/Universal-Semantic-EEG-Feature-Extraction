import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from useeg.cli import main

raise SystemExit(main(["figures", *sys.argv[1:]]))
