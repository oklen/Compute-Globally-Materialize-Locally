"""Redraw every README figure: assets/<name>.png and assets/<name>-dark.png, plus assets/kv-demo.gif, and
their Chinese copies for the Chinese README: assets/<name>-zh.png, assets/<name>-zh-dark.png and
assets/kv-demo-zh.gif.

    python figures/make_all.py [--lang en|zh]     # both languages by default

The paper's Figs. 1, 3 and 4 (concept, envelope, active) are drawn by their own scripts with the
paper palette (_paper_style.py), once per theme (FIG_THEME) and language (FIG_LANG); robustness and
consequences use the shared style (_style.py) and draw both themes and languages themselves;
make_kv_demo.py builds the animated sketch from kv_demo.json, in both languages."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = ["fig1_concept.py", "fig4_access.py", "fig5_active.py"]
NEW = ["robustness.py", "consequences.py", "make_kv_demo.py"]
LANGS = [sys.argv[sys.argv.index("--lang") + 1]] if "--lang" in sys.argv else ["en", "zh"]

for script in PAPER:
    for lang in LANGS:
        for theme in ("light", "dark"):
            subprocess.run([sys.executable, script], cwd=HERE, check=True,
                           env={**os.environ, "FIG_THEME": theme, "FIG_LANG": lang})
for script in NEW:
    subprocess.run([sys.executable, script] + (["--lang", LANGS[0]] if len(LANGS) == 1 else []), cwd=HERE,
                   check=True)
