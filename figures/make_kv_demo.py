"""Animated README figure: what a KV cache keeps after the source event is evicted.

    python figures/make_kv_demo.py        ->  assets/kv-demo.gif

Five scenes, following the paper's Discovery section and Fig. 1:
  1 compute globally: one prefill; the root event attends back to the source, so its cache rows take in S = ON
  2 evict: the source's text and rows are dropped; only the root is served
  3 serve and ask: "What is M?"; the reader answers ONLINE, though no served text says ONLINE
  4 the donor pair: same served text and positions, the source said OFF instead; different rows, different answer
  5 the measured result, with the paper's line references in figures/kv_demo.json

The cell colours are schematic: they show what the measurements imply, not real activations. The numbers
in scene 5 are measured. Texts, colours, timings and numbers live in figures/kv_demo.json.
Needs Pillow and matplotlib (only for its bundled DejaVu fonts)."""
import json
import os

import matplotlib
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "kv_demo.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "..", "assets", "kv-demo.gif")
FONTDIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
W, H, FPS = CFG["canvas"]["width"], CFG["canvas"]["height"], CFG["canvas"]["fps"]
STEP = 1000 // FPS                       # ms per transition frame
NUM = CFG["numbers"]


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


C = {k: rgb(v) for k, v in CFG["colors"].items()}


def mix(a, b, t):
    """Colour a moved toward b by t (0 = a, 1 = b)."""
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def fade(c, alpha):
    """Colour c at opacity alpha over the background."""
    return mix(C["bg"], c, max(0.0, min(1.0, alpha)))


_FONTS = {}


def font(size, bold=False, mono=False):
    name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    if (name, size) not in _FONTS:
        _FONTS[(name, size)] = ImageFont.truetype(os.path.join(FONTDIR, name), size)
    return _FONTS[(name, size)]


class Canvas:
    def __init__(self):
        self.im = Image.new("RGB", (W, H), C["bg"])
        self.d = ImageDraw.Draw(self.im)

    def text(self, xy, s, size=21, color=None, bold=False, mono=False, anchor="la", alpha=1.0):
        self.d.text(xy, s, font=font(size, bold, mono), fill=fade(color or C["fg"], alpha), anchor=anchor)

    def box(self, x0, y0, x1, y1, edge, fill=None, width=2, radius=10, alpha=1.0, dashed=False):
        fill = fade(fill, alpha) if fill is not None else None
        if not dashed:
            self.d.rounded_rectangle((x0, y0, x1, y1), radius=radius, outline=fade(edge, alpha),
                                     fill=fill, width=width)
            return
        if fill is not None:
            self.d.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=fill)
        for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) // 12)
            for i in range(0, n, 2):
                p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                q = (a[0] + (b[0] - a[0]) * (i + 1) / n, a[1] + (b[1] - a[1]) * (i + 1) / n)
                self.d.line((p, q), fill=fade(edge, alpha), width=width)

    def curve(self, p0, p1, lift, color, width=3, alpha=1.0, upto=1.0):
        """Quadratic curve from p0 to p1 bulging up by `lift`; drawn up to fraction `upto`."""
        cx, cy = (p0[0] + p1[0]) / 2, min(p0[1], p1[1]) - lift
        pts = []
        for i in range(41):
            t = upto * i / 40
            pts.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * cx + t * t * p1[0],
                        (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * cy + t * t * p1[1]))
        self.d.line(pts, fill=fade(color, alpha), width=width, joint="curve")
        if upto >= 1.0:   # arrowhead at p1
            (ax, ay), (bx, by) = pts[-4], pts[-1]
            dx, dy = bx - ax, by - ay
            n = max(1e-6, (dx * dx + dy * dy) ** 0.5)
            ux, uy = dx / n, dy / n
            self.d.polygon([(bx, by), (bx - 13 * ux + 6 * uy, by - 13 * uy - 6 * ux),
                            (bx - 13 * ux - 6 * uy, by - 13 * uy + 6 * ux)], fill=fade(color, alpha))

    def arrow(self, p0, p1, color, width=3, alpha=1.0):
        self.d.line((p0, p1), fill=fade(color, alpha), width=width)
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        n = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / n, dy / n
        bx, by = p1
        self.d.polygon([(bx, by), (bx - 14 * ux + 7 * uy, by - 14 * uy - 7 * ux),
                        (bx - 14 * ux - 7 * uy, by - 14 * uy + 7 * ux)], fill=fade(color, alpha))


# ---------------- layout of the event strip ----------------
EVENTS = CFG["events"]
NL = CFG["layers"]
CW, CH, GAP = 30, 18, 5                 # cache cell width, height, gap
PITCH = CW + GAP
BOX_Y0, BOX_Y1 = 226, 280               # event boxes
GRID_Y = 300                            # first cache row
MONO = 21


def grid_width(ev):
    return len(ev["tokens"]) * PITCH - GAP


def slot_width(ev):
    return int(max(font(MONO, mono=True).getlength(ev["text_off"] if "text_off" in ev else ev["text"]) + 36,
                   grid_width(ev) + 24))


SLOT_GAP = 28
_widths = [slot_width(e) for e in EVENTS]
_x = (W - sum(_widths) - SLOT_GAP * (len(EVENTS) - 1)) / 2
SLOT = {}
for e, w in zip(EVENTS, _widths):
    SLOT[e["key"]] = (_x, _x + w)
    _x += w + SLOT_GAP
EV = {e["key"]: e for e in EVENTS}


def grid_x0(key):
    x0, x1 = SLOT[key]
    return (x0 + x1) / 2 - grid_width(EV[key]) / 2


def draw_grid(cv, x0, y0, ncols, fill_cols, color, alpha=1.0, tint=None, tint_t=0.0):
    """Cache rows: one column per token, one row per layer. fill_cols columns are filled;
    a tint (colour, 0..1) moves filled cells toward that colour."""
    for c in range(ncols):
        for r in range(NL):
            x, y = x0 + c * PITCH, y0 + r * (CH + GAP)
            if c < fill_cols:
                col = color if tint is None else mix(color, tint, tint_t)
                cv.d.rectangle((x, y, x + CW, y + CH), fill=fade(col, alpha))
            else:
                cv.d.rectangle((x, y, x + CW, y + CH), fill=fade(C["cell_empty"], alpha),
                               outline=fade(C["edge"], alpha * 0.6))


def draw_event(cv, key, alpha=1.0, fill_cols=None, cell=None, tint=None, tint_t=0.0, text=None,
               dashed=False, label_alpha=None):
    e = EV[key]
    x0, x1 = SLOT[key]
    edge = {"src": C["source"], "root": C["root"]}.get(key, C["edge"])
    cv.text(((x0 + x1) / 2, BOX_Y0 - 12), e["label"], size=18, color=C["muted"], anchor="md",
            alpha=alpha if label_alpha is None else label_alpha)
    cv.box(x0, BOX_Y0, x1, BOX_Y1, edge, fill=C["box"], alpha=alpha, dashed=dashed)
    cv.text(((x0 + x1) / 2, (BOX_Y0 + BOX_Y1) / 2), text or e["text"], size=MONO, mono=True,
            color=C["fg"], anchor="mm", alpha=alpha)
    if fill_cols is not None:
        draw_grid(cv, grid_x0(key), GRID_Y, len(e["tokens"]), fill_cols, cell or C["cell"],
                  alpha=alpha, tint=tint, tint_t=tint_t)


def header(cv, num, title, lines, alpha=1.0):
    cv.text((64, 40), f"{num}  {title}", size=30, bold=True, alpha=alpha)
    for i, s in enumerate(lines):
        cv.text((64, 88 + 30 * i), s, size=21, color=C["muted"], alpha=alpha)


def schematic_note(cv, alpha=1.0, key=True):
    cv.text((W - 64, H - 34), "cache cells: schematic colours", size=18, color=C["faint"], anchor="rd",
            alpha=alpha)
    if key:
        colour_key(cv, alpha)


def grid_caption(cv, alpha=1.0):
    cv.text((W / 2, GRID_Y + NL * (CH + GAP) + 16), "cache rows under each event: one column per token, "
            "one row per layer", size=18, color=C["faint"], anchor="ma", alpha=alpha)


def colour_key(cv, alpha=1.0):
    """What the teal cells stand for (schematic)."""
    x, y = 64, H - 52
    cv.d.rectangle((x, y, x + CW, y + CH), fill=fade(C["on"], alpha))
    cv.text((x + CW + 12, y + CH / 2), "rows that carry S = ON", size=18, color=C["muted"], anchor="lm",
            alpha=alpha)


# ---------------- scenes ----------------
FRAMES = []          # (image, ms)


def emit(cv, ms):
    FRAMES.append((cv.im, int(ms)))


def ramp(n):
    return [(i + 1) / n for i in range(n)]


def scene_intro():
    cv = Canvas()
    cv.text((W / 2, 300), "Evicting an event need not evict its value", size=40, bold=True, anchor="mm")
    cv.text((W / 2, 360), "Compute Globally, Materialize Locally · what a KV cache keeps",
            size=24, color=C["muted"], anchor="mm")
    cv.text((W / 2, 420), "cache colours are schematic; the numbers at the end are measured",
            size=19, color=C["faint"], anchor="mm")
    emit(cv, CFG["seconds"]["intro"] * 1000)


ORDER = [e["key"] for e in EVENTS]
SRC_COLS = len(EV["src"]["tokens"])
ROOT_COLS = len(EV["root"]["tokens"])
CAP1 = ["The agent's history is prefilled once. Every event leaves cache rows behind."]
CAP1B = ["While the root is prefilled it attends back to the source.",
         "Its text never names the value; its cache rows take it in."]


def strip_state(cv, filled, root_tint, arcs=0.0, alpha=None):
    """Draw the five events with `filled` columns per event and the root's tint progress."""
    alpha = alpha or {}
    for k in ORDER:
        a = alpha.get(k, 1.0)
        if k == "src":
            draw_event(cv, k, alpha=a, fill_cols=filled.get(k, 0), cell=C["on"])
        elif k == "root":
            draw_event(cv, k, alpha=a, fill_cols=filled.get(k, 0), tint=C["on"], tint_t=root_tint)
        else:
            draw_event(cv, k, alpha=a, fill_cols=filled.get(k, 0))
    if arcs > 0:
        rx0, rx1 = SLOT["root"]
        sx0, sx1 = SLOT["src"]
        for i in range(3):
            p0 = (rx0 + (rx1 - rx0) * (0.3 + 0.2 * i), BOX_Y0 - 30)
            p1 = (sx0 + (sx1 - sx0) * (0.45 + 0.17 * i), BOX_Y0 - 30)
            cv.curve(p0, p1, 46 + 10 * i, C["attn"], width=3, upto=min(1.0, arcs))


def scene_compute():
    total = CFG["seconds"]["compute"] * 1000
    used = 0
    filled = {}
    cv = Canvas(); header(cv, 1, "Compute globally", CAP1); strip_state(cv, filled, 0); grid_caption(cv)
    schematic_note(cv); emit(cv, 1500); used += 1500
    for k in ["e1", "src", "e3"]:                       # prefill sweep, one token column per frame
        for c in range(len(EV[k]["tokens"])):
            filled[k] = c + 1
            cv = Canvas(); header(cv, 1, "Compute globally", CAP1); strip_state(cv, filled, 0)
            grid_caption(cv); schematic_note(cv); emit(cv, STEP); used += STEP
    for t in ramp(6):                                   # the root attends back to the source
        cv = Canvas(); header(cv, 1, "Compute globally", CAP1B); strip_state(cv, filled, 0, arcs=t)
        grid_caption(cv); schematic_note(cv); emit(cv, STEP); used += STEP
    filled["root"] = ROOT_COLS
    for t in ramp(8):                                   # its rows take in S = ON
        cv = Canvas(); header(cv, 1, "Compute globally", CAP1B); strip_state(cv, filled, t, arcs=1)
        grid_caption(cv); schematic_note(cv); emit(cv, STEP); used += STEP
    for c in range(len(EV["e5"]["tokens"])):
        filled["e5"] = c + 1
        cv = Canvas(); header(cv, 1, "Compute globally", CAP1B); strip_state(cv, filled, 1, arcs=1)
        grid_caption(cv); schematic_note(cv); emit(cv, STEP); used += STEP
    cv = Canvas(); header(cv, 1, "Compute globally", CAP1B); strip_state(cv, filled, 1, arcs=1)
    grid_caption(cv); schematic_note(cv)
    rx = (SLOT["root"][0] + SLOT["root"][1]) / 2
    cv.text((rx, GRID_Y + NL * (CH + GAP) + 56), "S = ON, absorbed", size=19, color=C["on"], anchor="ma")
    emit(cv, total - used)
    return filled


CAP2 = ["Sparse serving keeps a few rows and drops the rest,", "including the source that wrote the value."]
CAP2B = ["Kept: the root's text and its cache rows.", "No kept text says ONLINE."]


def evicted_label(cv, key, alpha):
    x0, x1 = SLOT[key]
    cv.text(((x0 + x1) / 2, GRID_Y + 40), "evicted", size=20, color=C["muted"], anchor="mm", alpha=alpha)


def scene_evict(filled):
    total = CFG["seconds"]["evict"] * 1000
    used = 0
    rx = (SLOT["root"][0] + SLOT["root"][1]) / 2
    for t in ramp(5):                                   # mark the source
        cv = Canvas(); header(cv, 2, "Evict", CAP2)
        strip_state(cv, filled, 1, arcs=1 - t)
        x0, x1 = SLOT["src"]
        cv.d.line(((x0 + 8, BOX_Y1 - 6), (x1 - 8, BOX_Y0 + 6)), fill=fade(C["off"], t), width=4)
        schematic_note(cv); emit(cv, STEP); used += STEP
    for t in ramp(8):                                   # drop the source and the other events
        a = {k: 1 - t for k in ORDER if k != "root"}
        cv = Canvas(); header(cv, 2, "Evict", CAP2)
        strip_state(cv, filled, 1, alpha=a)
        x0, x1 = SLOT["src"]
        cv.d.line(((x0 + 8, BOX_Y1 - 6), (x1 - 8, BOX_Y0 + 6)), fill=fade(C["off"], 1 - t), width=4)
        schematic_note(cv); emit(cv, STEP); used += STEP
    cv = Canvas(); header(cv, 2, "Evict", CAP2B)
    draw_event(cv, "src", alpha=0.35, dashed=True, label_alpha=0.6)
    evicted_label(cv, "src", 0.9)
    draw_event(cv, "root", fill_cols=ROOT_COLS, tint=C["on"], tint_t=1)
    cv.text((rx, GRID_Y + NL * (CH + GAP) + 56), "still S = ON", size=19, color=C["on"], anchor="ma")
    schematic_note(cv); emit(cv, total - used)


CAP3 = ["The question is appended after the kept rows. The reader sees", "only the root and the question."]


def ask_base(cv, show_q=1.0, show_reader=0.0, show_ans=0.0, show_check=0.0, caption=CAP3):
    header(cv, 3, "Serve and ask", caption)
    draw_event(cv, "src", alpha=0.35, dashed=True, label_alpha=0.6)
    evicted_label(cv, "src", 0.9)
    draw_event(cv, "root", fill_cols=ROOT_COLS, tint=C["on"], tint_t=1)
    # the question takes e5's slot: it comes after the root
    qx0, qx1 = SLOT["e5"]
    qx0, qx1 = qx0 - 20, qx1 + 20
    if show_q > 0:
        cv.text(((qx0 + qx1) / 2, BOX_Y0 - 12), "question", size=18, color=C["muted"], anchor="md", alpha=show_q)
        cv.box(qx0, BOX_Y0, qx1, BOX_Y1, C["fg"], fill=C["box"], alpha=show_q)
        cv.text(((qx0 + qx1) / 2, (BOX_Y0 + BOX_Y1) / 2), CFG["question"], size=MONO, mono=True,
                anchor="mm", alpha=show_q)
    # reader and answer, below
    rx0, rx1 = SLOT["root"][0], qx1
    mid = (rx0 + rx1) / 2
    ry = 470
    if show_reader > 0:
        cv.arrow((mid, GRID_Y + NL * (CH + GAP) + 8), (mid, ry - 8), C["muted"], alpha=show_reader)
        cv.box(mid - 150, ry, mid + 150, ry + 56, C["edge"], fill=C["box"], alpha=show_reader)
        cv.text((mid, ry + 28), "the model (Qwen3-8B)", size=21, anchor="mm", alpha=show_reader)
    if show_ans > 0:
        cv.arrow((mid, ry + 60), (mid, ry + 104), C["muted"], alpha=show_ans)
        cv.box(mid - 110, ry + 110, mid + 110, ry + 172, C["on"], fill=C["box"], width=3, alpha=show_ans)
        cv.text((mid, ry + 141), CFG["answers"]["on"], size=30, bold=True, color=C["on"], anchor="mm",
                alpha=show_ans)
    if show_check > 0:
        cv.text((64, 520), "Served text:", size=20, color=C["muted"], alpha=show_check)
        cv.text((64, 552), "“M mirrors S”", size=21, mono=True, alpha=show_check)
        cv.text((64, 584), "“What is M?”", size=21, mono=True, alpha=show_check)
        cv.text((64, 628), "The word ONLINE", size=21, color=C["fg"], alpha=show_check)
        cv.text((64, 658), "appears nowhere in it.", size=21, color=C["fg"], alpha=show_check)


def scene_ask():
    total = CFG["seconds"]["ask"] * 1000
    used = 0
    for t in ramp(4):
        cv = Canvas(); ask_base(cv, show_q=t); schematic_note(cv, key=False); emit(cv, STEP); used += STEP
    emit(cv, 600); used += 600
    for t in ramp(4):
        cv = Canvas(); ask_base(cv, show_reader=t); schematic_note(cv, key=False); emit(cv, STEP); used += STEP
    for t in ramp(4):
        cv = Canvas(); ask_base(cv, show_reader=1, show_ans=t); schematic_note(cv, key=False)
        emit(cv, STEP); used += STEP
    emit(cv, 900); used += 900
    for t in ramp(4):
        cv = Canvas(); ask_base(cv, show_reader=1, show_ans=1, show_check=t); schematic_note(cv, key=False)
        emit(cv, STEP); used += STEP
    emit(cv, total - used)


# ---------------- scene 4: the donor pair ----------------
CAP4 = ["Two histories, identical in every served token and position.",
        "Only the evicted source differs: in world B it said OFF."]
ROW_Y = {"A": 200, "B": 430}            # top of each world's row
DX = {"src": 150, "root": 520, "q": 760, "ans": 1010}


def world_row(cv, w, alpha=1.0, fill_t=1.0, ans_t=1.0):
    on = w == "A"
    val = C["on"] if on else C["off"]
    y = ROW_Y[w]
    cv.text((64, y + 28), f"world {w}", size=21, bold=True, color=C["muted"], anchor="lm", alpha=alpha)
    # the evicted source, ghosted
    s_text = EV["src"]["text"] if on else EV["src"]["text_off"]
    sw = font(MONO, mono=True).getlength(EV["src"]["text_off"]) + 36
    sx0 = DX["src"] + 30
    cv.box(sx0, y, sx0 + sw, y + 56, C["source"], fill=C["box"], alpha=0.4 * alpha, dashed=True)
    cv.text((sx0 + sw / 2, y + 28), s_text, size=MONO, mono=True, anchor="mm", alpha=0.55 * alpha)
    cv.text((sx0 + sw / 2, y + 78), "evicted", size=19, color=C["muted"], anchor="mm", alpha=0.9 * alpha)
    # the root: same text, its cache rows carry the value
    rw = font(MONO, mono=True).getlength(EV["root"]["text"]) + 36
    rx0 = DX["root"]
    cv.box(rx0, y, rx0 + rw, y + 56, C["root"], fill=C["box"], alpha=alpha)
    cv.text((rx0 + rw / 2, y + 28), EV["root"]["text"], size=MONO, mono=True, anchor="mm", alpha=alpha)
    gx = rx0 + rw / 2 - (ROOT_COLS * PITCH - GAP) / 2
    for c in range(ROOT_COLS):
        for r in range(NL):
            x, yy = gx + c * PITCH, y + 72 + r * (CH + GAP)
            col = mix(C["cell"], val, fill_t)
            cv.d.rectangle((x, yy, x + CW, yy + CH), fill=fade(col, alpha))
    # the question and the answer
    qw = font(MONO, mono=True).getlength(CFG["question"]) + 36
    qx0 = DX["q"]
    cv.box(qx0, y, qx0 + qw, y + 56, C["fg"], fill=C["box"], alpha=alpha)
    cv.text((qx0 + qw / 2, y + 28), CFG["question"], size=MONO, mono=True, anchor="mm", alpha=alpha)
    if ans_t > 0:
        cv.arrow((qx0 + qw + 12, y + 28), (DX["ans"] - 14, y + 28), C["muted"], alpha=ans_t * alpha)
        ans = CFG["answers"]["on" if on else "off"]
        cv.box(DX["ans"], y, DX["ans"] + 200, y + 56, val, fill=C["box"], width=3, alpha=ans_t * alpha)
        cv.text((DX["ans"] + 100, y + 28), ans, size=26, bold=True, color=val, anchor="mm", alpha=ans_t * alpha)


def donor_notes(cv, alpha):
    rw = font(MONO, mono=True).getlength(EV["root"]["text"]) + 36
    rx1 = DX["root"] + rw
    # brace between the two roots: same text, same positions
    x = rx1 + 22
    ya, yb = ROW_Y["A"] + 28, ROW_Y["B"] + 28
    cv.d.line(((x, ya), (x + 10, ya), (x + 10, yb), (x, yb)), fill=fade(C["muted"], alpha), width=2)
    cv.text((x + 18, (ya + yb) / 2 - 14), "same text,", size=19, color=C["fg"], alpha=alpha)
    cv.text((x + 18, (ya + yb) / 2 + 12), "same positions", size=19, color=C["fg"], alpha=alpha)
    cv.text((DX["root"] + rw / 2, ROW_Y["A"] + 72 + NL * (CH + GAP) + 6), "different cache rows",
            size=19, color=C["fg"], anchor="ma", alpha=alpha)


def scene_donor():
    total = CFG["seconds"]["donor"] * 1000
    used = 0
    for t in ramp(5):
        cv = Canvas(); header(cv, 4, "The donor pair", CAP4, alpha=1); world_row(cv, "A", alpha=t)
        schematic_note(cv, key=False); emit(cv, STEP); used += STEP
    emit(cv, 700); used += 700
    for t in ramp(4):
        cv = Canvas(); header(cv, 4, "The donor pair", CAP4); world_row(cv, "A")
        world_row(cv, "B", alpha=t, fill_t=0, ans_t=0); schematic_note(cv, key=False); emit(cv, STEP); used += STEP
    for t in ramp(6):
        cv = Canvas(); header(cv, 4, "The donor pair", CAP4); world_row(cv, "A")
        world_row(cv, "B", fill_t=t, ans_t=0); schematic_note(cv, key=False); emit(cv, STEP); used += STEP
    for t in ramp(4):
        cv = Canvas(); header(cv, 4, "The donor pair", CAP4); world_row(cv, "A")
        world_row(cv, "B", ans_t=t); schematic_note(cv, key=False); emit(cv, STEP); used += STEP
    emit(cv, 700); used += 700
    for t in ramp(4):
        cv = Canvas(); header(cv, 4, "The donor pair", CAP4); world_row(cv, "A"); world_row(cv, "B")
        donor_notes(cv, t)
        cv.text((W / 2, 652), "Same text. Different KV. Different answer.", size=32, bold=True, anchor="mm",
                alpha=t)
        schematic_note(cv, key=False); emit(cv, STEP); used += STEP
    emit(cv, total - used)


# ---------------- scene 5: the measured result ----------------
def result(cv, a1=1.0, a2=1.0, a3=1.0, a4=1.0):
    cv.text((64, 40), "5  What the paper measures", size=30, bold=True)
    cv.text((64, 88), "Same setup as scenes 2–4: the source evicted, donor pairs compared.", size=21,
            color=C["muted"])
    cv.text((64, 150), NUM["follow_anti"]["text"], size=96, bold=True, color=C["on"], alpha=a1)
    x = 64 + font(96, bold=True).getlength(NUM["follow_anti"]["text"]) + 40
    cv.text((x, 168), "donor-sensitive pairs on Qwen3-8B whose answer", size=22, alpha=a1)
    cv.text((x, 198), "follows the evicted value, against its opposite", size=22, alpha=a1)
    cv.text((x, 236), NUM["p"]["text"] + "  ·  " + NUM["replication"]["text"].split(" (")[0],
            size=20, color=C["muted"], alpha=a1)
    # inset: measured in the cache itself
    y0 = 320
    cv.box(64, y0, W - 64, y0 + 190, C["edge"], fill=C["box"], alpha=a2)
    cv.text((92, y0 + 22), "Measured in the cache itself (paper, App. J)", size=22, bold=True, alpha=a2)
    cv.text((92, y0 + 58), "A linear probe reads the source's state from the root's cached rows", size=20,
            color=C["muted"], alpha=a2)
    cv.text((92, y0 + 86), "under every construction tested, on four models.", size=20, color=C["muted"],
            alpha=a2)
    bx0, bx1, by = 92, 92 + 560, y0 + 140
    cv.d.rectangle((bx0, by - 14, bx1, by + 14), fill=fade(C["cell_empty"], a2))
    cv.d.rectangle((bx0, by - 14, bx0 + (bx1 - bx0) * 0.995, by + 14), fill=fade(C["on"], a2))
    cv.d.line(((bx0 + (bx1 - bx0) * 0.5, by - 24), (bx0 + (bx1 - bx0) * 0.5, by + 24)),
              fill=fade(C["fg"], a2), width=2)
    cv.text((bx0 + (bx1 - bx0) * 0.5, by + 30), "chance", size=18, color=C["muted"], anchor="ma", alpha=a2)
    cv.text((bx0, by + 30), "0", size=18, color=C["muted"], anchor="la", alpha=a2)
    cv.text((bx1, by + 30), "1", size=18, color=C["muted"], anchor="ra", alpha=a2)
    cv.text((bx1 + 24, by), "probe accuracy " + NUM["probe"]["text"], size=24, bold=True, color=C["on"],
            anchor="lm", alpha=a2)
    # the takeaway
    cv.text((W / 2, 590), "Dropping an event and seeing no accuracy loss", size=28, bold=True, anchor="mm",
            alpha=a3)
    cv.text((W / 2, 628), "does not show the event was unnecessary.", size=28, bold=True, anchor="mm",
            alpha=a3)
    cv.text((W / 2, 700), "Cache colours in this animation are schematic; the numbers on this screen are "
            "measured (arXiv 2607.23693, §3 and App. J).", size=18, color=C["faint"], anchor="mm", alpha=a4)


def scene_result():
    total = CFG["seconds"]["result"] * 1000
    used = 0
    for t in ramp(4):
        cv = Canvas(); result(cv, a1=t, a2=0, a3=0, a4=0); emit(cv, STEP); used += STEP
    emit(cv, 1200); used += 1200
    for t in ramp(4):
        cv = Canvas(); result(cv, a2=t, a3=0, a4=0); emit(cv, STEP); used += STEP
    emit(cv, 1500); used += 1500
    for t in ramp(4):
        cv = Canvas(); result(cv, a3=t, a4=t); emit(cv, STEP); used += STEP
    emit(cv, total - used)


def build():
    scene_intro()
    filled = scene_compute()
    scene_evict(filled)
    scene_ask()
    scene_donor()
    scene_result()
    # merge identical consecutive frames, then one shared palette for every frame
    merged = []
    for im, ms in FRAMES:
        if merged and merged[-1][0].tobytes() == im.tobytes():
            merged[-1] = (merged[-1][0], merged[-1][1] + ms)
        else:
            merged.append((im, ms))
    keys = [merged[i][0] for i in range(0, len(merged), max(1, len(merged) // 12))]
    sheet = Image.new("RGB", (W, H * len(keys)))
    for i, im in enumerate(keys):
        sheet.paste(im, (0, H * i))
    pal = sheet.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im, _ in merged]
    durs = [ms for _, ms in merged]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
    print("wrote", os.path.relpath(OUT, os.path.join(HERE, "..")), len(frames), "frames,",
          round(sum(durs) / 1000, 1), "s,", round(os.path.getsize(OUT) / 1024), "KB")


if __name__ == "__main__":
    build()
