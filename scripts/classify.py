import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from useeg.cli import main

raise SystemExit(main(["classify", *sys.argv[1:]]))
