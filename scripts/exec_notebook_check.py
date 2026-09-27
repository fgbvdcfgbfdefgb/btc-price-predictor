"""Execute every notebook cell in a __main__ namespace, exactly like Jupyter.

The 5-hour training driver is shrunk to a single short generation so the
whole notebook can be verified in a couple of minutes.
"""
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless

NB = Path("notebooks/btc_price.ipynb")
cells = json.load(NB.open())["cells"]

# resume point, so the shortened run does exactly one more generation
state = Path("checkpoints/state.json")
gen_now = json.loads(state.read_text())["generation"] if state.exists() else 0

SHRINK = {
    "generations=1000": f"generations={gen_now + 1}",
    "inner_steps=600": "inner_steps=40",
    "max_hours=5.0": "max_hours=0.08",
}

g = globals()
ok = fail = 0
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    tag = ""
    if "run(args)" in src:
        for a, b in SHRINK.items():
            src = src.replace(a, b)
        tag = f"  [shrunk -> 1 generation, 40 steps]"
    if "upload_to_github" in src:
        print(f"cell {i:>2}: SKIPPED (github upload)")
        continue
    head = next((l for l in src.splitlines() if l.strip()
                 and not l.strip().startswith("#")), "")[:58]
    try:
        exec(compile(src, f"<cell {i}>", "exec"), g)
        ok += 1
        print(f"cell {i:>2}: OK    {head}{tag}")
    except Exception:
        fail += 1
        print(f"cell {i:>2}: FAIL  {head}")
        traceback.print_exc(limit=6)
        break

print(f"\n=== executed {ok} cells, {fail} failure(s) ===")
sys.exit(1 if fail else 0)
