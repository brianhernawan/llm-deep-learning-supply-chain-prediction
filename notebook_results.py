"""Print the result lines (F1, accuracy, mAP, precision, recall, loss) saved inside your notebooks.
Usage:  python notebook_results.py  path/to/folder_with_ipynb   (searches sub-folders too)
"""
import json
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"(f1|accuracy|precision|recall|map50|mAP|overall|eval_loss|seqeval)", re.IGNORECASE)


def output_text(cell):
    """All text a code cell printed (images are skipped)."""
    parts = []
    for out in cell.get("outputs", []):
        text = out.get("text") or out.get("data", {}).get("text/plain", "")
        parts.append("".join(text) if isinstance(text, list) else text)
    return "\n".join(parts)


def show(path, last=15):
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    hits = [line.strip() for cell in cells if cell["cell_type"] == "code"
            for line in output_text(cell).splitlines() if PATTERN.search(line)]
    print(f"\n=== {path.name}  ({len(hits)} result lines, last {last} shown)")
    for line in hits[-last:]:
        print("  ", line[:160])


folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
for nb in sorted(folder.rglob("*.ipynb")):
    show(nb)
