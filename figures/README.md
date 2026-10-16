# Figures on the README

`python figures/make_all.py` (from the repository root) redraws every figure on the README, twice:
`assets/<name>.png` for GitHub's light theme and `assets/<name>-dark.png` for its dark theme. The
README switches between them with `<picture>`. Requirements: matplotlib and numpy.

| README asset | Script | Paper element | Numbers from |
|---|---|---|---|
| `concept.png` | `fig1_concept.py` | Fig. 1 | schematic, no data |
| `envelope.png` | `fig4_access.py` | Fig. 3 (§6) | `data/fig_data.json` (`fig4`, `fig4C`) |
| `active.png` | `fig5_active.py` | Fig. 4 (§8) | `data/fig_data.json` (`fig5A`), `data/x9_rates.json` |
| `robustness.png` | `robustness.py` | Table 1 (§3), Fig. 2A (§5), appendix tables | inline in the script, each row with its line in the paper source |
| `consequences.png` | `consequences.py` | §9 and its appendix | inline in the script |

Script names follow the figure package the paper was drawn with; the paper numbers its figures
1-4 in the main text as listed above.

- `fig1_concept.py`, `fig4_access.py` and `fig5_active.py` draw the paper's own figures with the paper's
  palette (`_paper_style.py`). `FIG_THEME=dark` changes only the background, the text and the axes.
  Words on the light event boxes stay dark in both themes.
- `robustness.py` and `consequences.py` use `_style.py`, the style shared by the README figures of the
  three agent-memory repositories (VeraKV, TracePack and this one).
- The README copies differ from the paper's figures in layout only. Fig. 1's bottom labels no longer
  touch the frame, Fig. 3's panel titles wrap onto two lines, and Fig. 4's value labels sit above
  their error bars. Every plotted value is unchanged.
- The Gemma-4 mention-absent count in `robustness.png` (17:0) appears in the paper only in Fig. 2A,
  not in the text.
