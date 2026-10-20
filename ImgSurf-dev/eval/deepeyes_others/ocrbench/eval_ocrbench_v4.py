"""ocrbench v4 evaluation entry point."""
from pathlib import Path
import sys

try:
    from eval.deepeyes_others.common_eval import main_eval
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deepeyes_others.common_eval import main_eval

if __name__ == "__main__":
    main_eval("ocrbench", "v4")

