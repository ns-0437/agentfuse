"""Render a short animated GIF of AgentFuse breaking a live agent loop.

Produces ``assets/demo.gif`` — a stylized terminal recording that reveals the
loop-trap run line by line: the agent repeats a doomed tool call, the circuit
breaker trips, the deterministic escalation ladder injects a steering path, and
the agent self-heals. Pure Pillow; no screen recorder needed.

Run:  python assets/make_demo_gif.py
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "demo.gif"

W, H = 960, 720
BG = (7, 10, 16)
CARD = (13, 20, 32)
LINE = (36, 48, 68)
TEXT = (230, 237, 246)
DIM = (138, 153, 173)
DIM2 = (95, 111, 133)
TRIP = (245, 197, 66)
HEAL = (56, 189, 248)
OK = (52, 211, 153)
TOOL = (167, 139, 250)
GREEN = (52, 211, 153)

FONTS = "C:/Windows/Fonts/"
def _font(name, size):
    try:
        return ImageFont.truetype(FONTS + name, size)
    except Exception:
        return ImageFont.load_default()

F = _font("consola.ttf", 18)
FB = _font("consolab.ttf", 18)
FT = _font("consolab.ttf", 15)
FS = _font("consola.ttf", 14)

MARGIN = 18
PAD_X = 34
TOP = 92
LH = 28

# Each block: (kind, [(text, color, font), ...])
# kind: "line" | "panel-trip" | "panel-heal"
BLOCKS = [
    ("line", [("$ python examples/demo_loop_trap.py", GREEN, F)]),
    ("gap", []),
    ("line", [("OBJECTIVE  Rotate the production database credential.", DIM, F)]),
    ("gap", []),
    ("line", [("step 1   tool_call    ", DIM2, F), ("search_files(./config, *.conn)", TOOL, F)]),
    ("line", [("step 1   tool_result  0 files matched", DIM2, F)]),
    ("line", [("step 2   tool_call    ", DIM2, F), ("search_files(./config, *.conn)", TOOL, F)]),
    ("line", [("step 2   tool_result  0 files matched", DIM2, F)]),
    ("line", [("step 3   tool_call    ", DIM2, F), ("search_files(./config, *.conn)", TOOL, F)]),
    ("panel-trip", [
        ("CIRCUIT BREAKER TRIPPED  -  LOOP", TRIP, FT),
        ("search_files called 3x, identical args, no state progress.", TEXT, FS),
    ]),
    ("panel-heal", [
        ("STEERING RECOVERY   ·   deterministic ladder", HEAL, FT),
        ("Stop repeating search_files - it isn't advancing the task.", TEXT, FS),
        ("Re-read the objective and take a different next action.", TEXT, FS),
    ]),
    ("line", [("resume    steering injected - agent resuming", HEAL, F)]),
    ("line", [("step 4   tool_call    ", DIM2, F), ("secret_manager.get(prod/db/primary)", TOOL, F)]),
    ("line", [("COMPLETE  credential rotated after self-healing", OK, FB)]),
    ("gap", []),
    ("line", [("trips 1    recoveries 1    steps 4    status complete", DIM, F)]),
]

# Duration (ms) to hold once each block appears.
DUR = {
    "default": 460,
    9: 1300,   # trip panel
    10: 1650,  # recovery panel
    13: 900,   # complete
    15: 2800,  # final hold
}


def block_height(kind, lines):
    if kind == "gap":
        return LH // 2
    if kind.startswith("panel"):
        return 18 + len(lines) * 24 + 16
    return LH


def draw_frame(n_blocks: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # terminal card
    d.rounded_rectangle([MARGIN, MARGIN, W - MARGIN, H - MARGIN], radius=16,
                        fill=CARD, outline=LINE, width=1)
    # title bar
    for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([PAD_X + i * 22, MARGIN + 20, PAD_X + i * 22 + 12, MARGIN + 32], fill=col)
    d.text((W // 2, MARGIN + 26), "agentfuse  ·  logical circuit breaker",
           font=FT, fill=DIM, anchor="mm")
    d.line([MARGIN + 1, MARGIN + 52, W - MARGIN - 1, MARGIN + 52], fill=LINE, width=1)

    y = TOP
    for bi in range(min(n_blocks, len(BLOCKS))):
        kind, lines = BLOCKS[bi]
        h = block_height(kind, lines)
        if kind == "gap":
            y += h
            continue
        if kind.startswith("panel"):
            accent = TRIP if kind == "panel-trip" else HEAL
            fill = (26, 24, 12) if kind == "panel-trip" else (10, 26, 34)
            d.rounded_rectangle([PAD_X, y, W - PAD_X - 6, y + h - 8], radius=10,
                                fill=fill, outline=accent, width=1)
            d.rectangle([PAD_X, y + 6, PAD_X + 4, y + h - 14], fill=accent)
            ty = y + 10
            for (text, color, font) in lines:
                d.text((PAD_X + 18, ty), text, font=font, fill=color)
                ty += 24
            y += h
            continue
        # normal line: may have multiple colored spans
        x = PAD_X + 6
        for (text, color, font) in lines:
            d.text((x, y), text, font=font, fill=color)
            x += d.textlength(text, font=font)
        y += h

    # blinking cursor at the end of revealed content
    if n_blocks < len(BLOCKS):
        d.rectangle([PAD_X + 6, y + 4, PAD_X + 16, y + 20], fill=DIM)
    return img


def main() -> None:
    frames, durations = [], []
    for k in range(1, len(BLOCKS) + 1):
        # skip standalone gap-only reveals to avoid dead frames
        kind = BLOCKS[k - 1][0]
        frames.append(draw_frame(k))
        durations.append(DUR.get(k - 1, DUR["default"]) if kind != "gap" else 120)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=True)
    kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT}  ({len(frames)} frames, {kb:.0f} KB, {W}x{H})")


if __name__ == "__main__":
    main()
