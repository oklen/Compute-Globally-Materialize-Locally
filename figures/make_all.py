"""Redraw every README figure in both GitHub themes: assets/<name>.png and assets/<name>-dark.png.

    python figures/make_all.py

The paper's Figs. 1, 4 and 5 (concept, envelope, active) are drawn by their own scripts with the
paper palette (_paper_style.py); robustness and consequences use the shared style (_style.py)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = ["fig1_concept.py", "fig4_access.py", "fig5_active.py"]
NEW = ["robustness.py", "consequences.py"]

for script in PAPER:
    for theme in ("light", "dark"):
        subprocess.run([sys.executable, script], cwd=HERE, check=True,
                       env={**os.environ, "FIG_THEME": theme})
for script in NEW:
    subprocess.run([sys.executable, script], cwd=HERE, check=True)
