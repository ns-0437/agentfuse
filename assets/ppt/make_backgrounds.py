"""Generate premium dark background images for the AgentFuse pitch deck.

Deep navy base with soft teal/cyan radial glows, a faint dot grid, and subtle
concentric rings — an OpenAI-inspired, control-room aesthetic that matches the
AgentFuse dashboard palette. Pure Pillow.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent
W, H = 2560, 1440
BASE = (11, 15, 23)          # #0B0F17
TEAL = (16, 163, 127)        # OpenAI green #10A37F
CYAN = (56, 189, 248)        # #38BDF8
AMBER = (245, 197, 66)


def glow(size, cx, cy, r, color, alpha):
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (alpha,))
    return layer.filter(ImageFilter.GaussianBlur(r // 2))


def dot_grid(size, step=46, color=(255, 255, 255), alpha=10):
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for x in range(0, size[0], step):
        for y in range(0, size[1], step):
            d.ellipse([x, y, x + 2, y + 2], fill=color + (alpha,))
    return layer


def rings(size, cx, cy, radii, color, alpha, width=2):
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for r in radii:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color + (alpha,), width=width)
    return layer.filter(ImageFilter.GaussianBlur(1))


def compose(layers) -> Image.Image:
    img = Image.new("RGBA", (W, H), BASE + (255,))
    for l in layers:
        img = Image.alpha_composite(img, l)
    return img.convert("RGB")


def title_bg():
    layers = [
        dot_grid((W, H), alpha=8),
        rings((W, H), 2150, 780, [260, 420, 600, 820], TEAL, 22, 2),
        glow((W, H), 1950, 560, 620, TEAL, 90),
        glow((W, H), 520, 1150, 560, CYAN, 55),
        glow((W, H), 2300, 1250, 320, AMBER, 26),
    ]
    compose(layers).save(OUT / "bg_title.png")


def content_bg():
    layers = [
        dot_grid((W, H), alpha=6),
        glow((W, H), 2350, 180, 560, TEAL, 42),
        glow((W, H), 120, 1350, 480, CYAN, 26),
    ]
    compose(layers).save(OUT / "bg_content.png")


def thanks_bg():
    layers = [
        dot_grid((W, H), alpha=8),
        rings((W, H), 1280, 720, [300, 480, 680, 900], TEAL, 20, 2),
        glow((W, H), 1280, 720, 680, TEAL, 80),
        glow((W, H), 1280, 1300, 520, CYAN, 40),
    ]
    compose(layers).save(OUT / "bg_thanks.png")


if __name__ == "__main__":
    title_bg(); content_bg(); thanks_bg()
    print("wrote bg_title.png, bg_content.png, bg_thanks.png")
