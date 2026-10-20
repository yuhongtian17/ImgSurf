"""Judge predictions for pope."""
from pathlib import Path
import sys

try:
    from eval.deepeyes_others.common_eval import main_judge
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deepeyes_others.common_eval import main_judge

if __name__ == "__main__":
    main_judge("pope")

