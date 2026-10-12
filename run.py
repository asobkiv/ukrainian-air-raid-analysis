"""Run the project directly from its checkout (installation is optional)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from airraid.cli import main

if __name__ == "__main__":
    main()
