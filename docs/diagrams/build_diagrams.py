"""Generate the project diagrams (diagram-design skill, default skin) as self-contained HTML.

Writes timing.html, scheme.html, scheme-zh.html and their -dark variants next to this file.
Geometry is computed here so the light and dark variants share it exactly.
Usage: python docs/diagrams/build_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent

SANS = "'Geist', system-ui, sans-serif"
MONO = "'Geist Mono', ui-monospace, monospace"
SERIF = "'Instrument Serif', 'Noto Serif', serif"

SKINS = {
    "light": dict(paper="#f5f5f5", ink="#2d3142", muted="#4f5d75", soft="#5a6580", accent="#bf4520",
                  accent_rgb="191,69,32", link="#2e5aa8", ink_rgb="45,49,66", backend="#ffffff",
                  input_fill="rgba(79,93,117,0.10)", input_stroke="#5a6580"),
    "dark": dict(paper="#2d3142", ink="#f5f5f5", muted="#bfc0c0", soft="#949eb2", accent="#f08a59",
                 accent_rgb="240,138,89", link="#739fdf", ink_rgb="245,245,245", backend="#393e53",
                 input_fill="rgba(191,192,192,0.10)", input_stroke="#949eb2"),
}


def ink(s, a):
    return f"rgba({s['ink_rgb']},{a})"


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def r4(v: float) -> int:
    return int(round(v / 4.0) * 4)


def mono_w(text: str, size: float, tracking: float = 0.0) -> float:
    return len(text) * size * (0.62 + tracking)


# ------------------------------------------------------------------ primitives

def defs(s, slug) -> str:
    return (f'<defs>'
            f'<marker id="{slug}-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
            f'<polygon points="0 0, 8 3, 0 6" fill="{s["muted"]}"/></marker>'
            f'<marker id="{slug}-arrow-soft" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
            f'<polygon points="0 0, 8 3, 0 6" fill="{s["soft"]}"/></marker>'
            f'<marker id="{slug}-arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
            f'<polygon points="0 0, 8 3, 0 6" fill="{s["accent"]}"/></marker>'
            f'</defs>')


NODE_STYLE = {
    "focal": lambda s: (f"rgba({s['accent_rgb']},0.08)", s["accent"], ""),
    "backend": lambda s: (s["backend"], s["ink"], ""),
    "store": lambda s: (ink(s, 0.05), s["muted"], ""),
    "input": lambda s: (s["input_fill"], s["input_stroke"], ""),
    "optional": lambda s: (ink(s, 0.02), ink(s, 0.20), ' stroke-dasharray="4,3"'),
}


def node(s, x, y, w, h, kind, tag, name, sub) -> str:
    fill, stroke, dash = NODE_STYLE[kind](s)
    tw = r4(mono_w(tag, 7, 0.08) + 10)
    cx = x + w / 2
    tag_stroke = f"rgba({s['accent_rgb']},0.50)" if kind == "focal" else ink(s, 0.40)
    tag_fill = s["accent"] if kind == "focal" else s["muted"]
    return (f'<g data-node="{esc(name)}">'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{s["paper"]}"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>'
            f'<rect x="{x + 8}" y="{y + 6}" width="{tw}" height="12" rx="2" fill="transparent" stroke="{tag_stroke}" stroke-width="0.8"/>'
            f'<text x="{x + 8 + tw / 2}" y="{y + 15}" fill="{tag_fill}" font-size="7" font-family="{MONO}" '
            f'text-anchor="middle" letter-spacing="0.08em">{esc(tag)}</text>'
            f'<text x="{cx}" y="{y + h / 2 + 6}" fill="{s["ink"]}" font-size="12" font-weight="600" '
            f'font-family="{SANS}" text-anchor="middle">{esc(name)}</text>'
            f'<text x="{cx}" y="{y + h / 2 + 22}" fill="{s["muted"]}" font-size="9" font-family="{MONO}" '
            f'text-anchor="middle">{esc(sub)}</text></g>')


def path(s, slug, d, dashed=False, soft=False) -> str:
    color = s["soft"] if soft else s["muted"]
    marker = f"{slug}-arrow-soft" if soft else f"{slug}-arrow"
    extra = ' stroke-dasharray="5,4"' if dashed else ""
    width = "1" if dashed else "1.2"
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{extra} marker-end="url(#{marker})"/>'


def label(s, cx, cy, text) -> str:
    """Arrow label centred at (cx, cy) on an opaque paper mask (caller keeps a 6-10px gap)."""
    w = r4(mono_w(text, 8, 0.06) + 8)
    return (f'<rect x="{cx - w / 2}" y="{cy - 6}" width="{w}" height="12" rx="2" fill="{s["paper"]}"/>'
            f'<text x="{cx}" y="{cy + 3}" fill="{s["soft"]}" font-size="8" font-family="{MONO}" '
            f'text-anchor="middle" letter-spacing="0.06em">{esc(text)}</text>')


def legend(s, y, width, items) -> str:
    """items: list of (swatch_svg_fn(x, y), text). Horizontal strip at the bottom."""
    out = [f'<line x1="40" y1="{y - 8}" x2="{width - 40}" y2="{y - 8}" stroke="{ink(s, 0.10)}" stroke-width="0.8"/>',
           f'<text x="40" y="{y + 8}" fill="{s["muted"]}" font-size="8" font-family="{MONO}" letter-spacing="0.18em">LEGEND</text>']
    x = 40
    for sw, text in items:
        out.append(sw(x, y + 20))
        out.append(f'<text x="{x + 22}" y="{y + 28}" fill="{s["muted"]}" font-size="8.5" font-family="{SANS}">{esc(text)}</text>')
        x += 22 + len(text) * 8.5 * 0.6 + 28
    return "".join(out)


def box_swatch(s, kind):
    fill, stroke, dash = NODE_STYLE[kind](s)
    return lambda x, y: f'<rect x="{x}" y="{y}" width="14" height="10" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>'


def line_swatch(s, dashed=False, color=None):
    c = color or s["muted"]
    d = ' stroke-dasharray="5,4"' if dashed else ""
    return lambda x, y: f'<line x1="{x}" y1="{y + 5}" x2="{x + 16}" y2="{y + 5}" stroke="{c}" stroke-width="1.2"{d}/>'


def page(slug, variant, eyebrow, title, desc, vb_w, vb_h, body, s) -> str:
    sid = f"{slug}-dark" if variant == "dark" else slug
    return f"""<!DOCTYPE html>
<html lang="{'zh-CN' if slug.endswith('-zh') else 'en'}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{esc(title)}</title>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&family=Noto+Serif:ital@0;1&family=Noto+Sans+SC:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{ --color-paper: {s['paper']}; --color-ink: {s['ink']}; --color-muted: {s['muted']}; }}
    body {{ font-family: {SANS}; background: var(--color-paper); color: var(--color-ink); min-height: 100vh;
           display: flex; align-items: center; justify-content: center; padding: 3rem 2rem; }}
    .frame {{ max-width: {max(1200, vb_w)}px; width: 100%; }}
    .diagram-container {{ width: 100%; overflow-x: auto; }}
    .eyebrow {{ font-family: {MONO}; font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em;
               text-transform: uppercase; color: var(--color-muted); margin-bottom: 0.5rem; }}
    h1 {{ font-family: 'Instrument Serif', 'Noto Serif', 'Noto Serif SC', 'Songti SC', serif; font-size: clamp(1.5rem, 2.4vw + 0.75rem, 2rem); font-weight: 400;
         letter-spacing: -0.02em; line-height: 1.15; color: var(--color-ink); margin-bottom: 1.5rem; }}
    svg {{ width: 100%; min-width: {vb_w}px; display: block; }}
    @media print {{ .diagram-container {{ overflow-x: visible; }} svg {{ min-width: 0; }} }}
  </style>
</head>
<body>
  <div class="frame">
    <p class="eyebrow">{esc(eyebrow)}</p>
    <h1>{esc(title)}</h1>
    <div class="diagram-container">
      <svg viewBox="0 0 {vb_w} {vb_h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{sid}-title {sid}-desc">
        <title id="{sid}-title">{esc(title)}</title>
        <desc id="{sid}-desc">{esc(desc)}</desc>
        {defs(s, sid)}
        <rect width="100%" height="100%" fill="{s['paper']}"/>
        {body}
      </svg>
    </div>
  </div>
</body>
</html>
"""


# ------------------------------------------------------------------ 1. timing (Timeline)

def timing(s, sid) -> tuple[str, int, int]:
    """Timeline with event cards in the scheme's role colours; horizontal spacing proportional to time."""
    k = s["mod"]
    c = k["c"]
    W, H, Y = 1200, 540, 300
    x0, x1, h0, h1 = 48, 1152, 8, 76                 # hours after D-2 00:00 on the axis

    def X(hours):
        return round(x0 + (hours - h0) * (x1 - x0) / (h1 - h0), 1)

    o = ["<defs>" + "".join(
        f'<marker id="{sid}-arrow-{v.lstrip("#")}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
        f'<polygon points="0 0, 8 3, 0 6" fill="{v}"/></marker>' for v in sorted(set(c.values()))) + "</defs>"]
    # day bands on the axis
    BH = 32
    for d0, d1, txt, col in ((h0, 24, "D−2", None), (24, 48, "D−1", None), (48, 72, "D  (delivery day)", c["ext"]),
                             (72, h1, "D+1", None)):
        fill = hexa(col, k["tint_focal"]) if col else k["zone"]
        stroke = col if col else k["zone_stroke"]
        o.append(f'<rect x="{X(d0)}" y="{Y - BH / 2}" width="{X(d1) - X(d0)}" height="{BH}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="1.4"/>')
        o.append(f'<text x="{X(d0) + (40 if d0 == h0 else 12)}" y="{Y + 5}" fill="{col or k["sub"]}" font-size="14" font-weight="600" '
                 f'font-family="{CJK_SANS}">{txt}</text>')
    # information set: everything published by gate closure
    gx = X(36)
    by = Y - BH / 2 - 44
    o.append(f'<path d="M{x0},{by + 10} V{by} H{gx - 12} V{by + 10}" fill="none" stroke="{c["pub"]}" stroke-width="2"/>')
    o.append(f'<rect x="{x0 + 8}" y="{by - 26}" width="{gx - x0 - 160}" height="20" fill="{s["paper"]}"/>')
    o.append(f'<text x="{x0 + 12}" y="{by - 11}" fill="{c["pub"]}" font-size="14" font-weight="600" font-family="{CJK_SANS}">'
             f'What the planner may use for D: prices of days ≤ D−1</text>')

    def card(xe, top, col, when, name, sub, focal=False):
        w, h = 240, 84
        x = min(max(xe - w / 2, 16), W - 16 - w)
        fill = hexa(col, k["tint_focal"] if focal else k["tint"])
        o.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="6" fill="{k["node"]}"/>')
        o.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{col}" '
                 f'stroke-width="{2.4 if focal else 1.6}"/>')
        cx = x + w / 2
        o.append(f'<text x="{cx}" y="{top + 24}" fill="{col}" font-size="13" font-weight="600" font-family="{MONO}" '
                 f'text-anchor="middle" letter-spacing="0.06em">{esc(when)}</text>')
        o.append(f'<text x="{cx}" y="{top + 48}" fill="{col}" font-size="17" font-weight="600" font-family="{CJK_SANS}" '
                 f'text-anchor="middle">{esc(name)}</text>')
        o.append(f'<text x="{cx}" y="{top + 69}" fill="{k["sub"]}" font-size="13.5" font-family="{CJK_SANS}" '
                 f'text-anchor="middle">{esc(sub)}</text>')
        assert tw(name, 17, 0.56) <= w - 16 and tw(sub, 13.5, 0.53) <= w - 16, (name, sub)
        return x, w, h

    def stem(xe, ya, yb, col):
        o.append(f'<line x1="{xe}" y1="{ya}" x2="{xe}" y2="{yb}" stroke="{col}" stroke-width="2"/>')

    def dot(xe, col, r=7):
        o.append(f'<circle cx="{xe}" cy="{Y}" r="{r + 2.5}" fill="{s["paper"]}"/><circle cx="{xe}" cy="{Y}" r="{r}" fill="{col}"/>')

    TOP, BOT = 64, Y + BH / 2 + 56
    # stems first, then cards and dots
    stem(gx, TOP + 84, Y, c["milp"])
    stem(X(13), Y, BOT, c["pub"])
    stem(X(37), Y, BOT, c["real"])
    dx = X(60)
    stem(dx, TOP + 84, Y - BH / 2, c["ext"])
    # D's prices -> settlement of D (dashed)
    xs = X(37) + 120
    ys = BOT + 42
    o.append(f'<path d="M{xs},{ys} H{dx - 8} Q{dx},{ys} {dx},{ys - 8} V{Y + BH / 2}" fill="none" stroke="{c["real"]}" '
             f'stroke-width="2" stroke-dasharray="7,5" marker-end="url(#{sid}-arrow-{c["real"].lstrip("#")})"/>')
    lab_txt = "USED ONLY TO SETTLE D"
    lw = r4(len(lab_txt) * 11.5 * 0.68 + 14)
    o.append(f'<rect x="{(xs + dx) / 2 - lw / 2}" y="{ys - 26}" width="{lw}" height="16" rx="2" fill="{s["paper"]}"/>'
             f'<text x="{(xs + dx) / 2}" y="{ys - 14}" fill="{c["real"]}" font-size="11.5" font-weight="500" font-family="{MONO}" '
             f'text-anchor="middle" letter-spacing="0.06em">{lab_txt}</text>')
    card(gx, TOP, c["milp"], "D−1 12:00", "Gate closure for D", "forecast, then lock the day", focal=True)
    card(dx, TOP, c["ext"], "D 00:00 – 24:00", "Delivery of D", "23, 24 or 25 hours (DST)")
    card(X(13), BOT, c["pub"], "D−2 13:00", "Prices of D−1 published", "latest day the planner sees")
    card(X(37), BOT, c["real"], "D−1 13:00", "Prices of D published", "after the plan is locked")
    for xe, col in ((X(13), c["pub"]), (gx, c["milp"]), (X(37), c["real"])):
        dot(xe, col)
    o.append(f'<text x="{W - 48}" y="{H - 20}" fill="{k["sub"]}" font-size="13" font-style="italic" font-family="{CJK_SANS}" '
             f'text-anchor="end">Horizontal spacing is proportional to elapsed time (Europe/Berlin).</text>')
    return "".join(o), W, H


# ------------------------------------------------------------------ 2. scheme (Architecture, en / zh)

CJK_SANS = "'Geist', 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif"


def is_wide(ch: str) -> bool:
    import unicodedata
    return unicodedata.east_asian_width(ch) in ("W", "F")


def text_w(text: str, size: float, latin: float = 0.60) -> float:
    return sum(size if is_wide(c) else size * latin for c in text)


def has_cjk(text: str) -> bool:
    return any(is_wide(c) for c in text)


def node_i18n(s, x, y, w, h, kind, tag, name, sub) -> str:
    """Node with Han-aware type: names in sans with the SC stack; prose sublabels in 10px sans."""
    fill, stroke, dash = NODE_STYLE[kind](s)
    tw = r4(mono_w(tag, 7, 0.08) + 10)
    cx = x + w / 2
    tag_stroke = f"rgba({s['accent_rgb']},0.50)" if kind == "focal" else ink(s, 0.40)
    tag_fill = s["accent"] if kind == "focal" else s["muted"]
    sub_font = (f'font-size="10.5" font-family="{CJK_SANS}"' if has_cjk(sub)
                else f'font-size="9" font-family="{MONO}"')
    return (f'<g data-node="{esc(name)}">'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{s["paper"]}"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>'
            f'<rect x="{x + 8}" y="{y + 6}" width="{tw}" height="12" rx="2" fill="transparent" stroke="{tag_stroke}" stroke-width="0.8"/>'
            f'<text x="{x + 8 + tw / 2}" y="{y + 15}" fill="{tag_fill}" font-size="7" font-family="{MONO}" '
            f'text-anchor="middle" letter-spacing="0.08em">{esc(tag)}</text>'
            f'<text x="{cx}" y="{y + h / 2 + 6}" fill="{s["ink"]}" font-size="12.5" font-weight="600" '
            f'font-family="{CJK_SANS}" text-anchor="middle">{esc(name)}</text>'
            f'<text x="{cx}" y="{y + h / 2 + 22}" fill="{s["muted"]}" {sub_font} text-anchor="middle">{esc(sub)}</text></g>')


def label_i18n(s, cx, cy, text) -> str:
    """Arrow label; Han labels switch register to 12px sans 500 on a 16px mask (style guide)."""
    if has_cjk(text):
        w = r4(text_w(text, 12) + 12)
        return (f'<rect x="{cx - w / 2}" y="{cy - 8}" width="{w}" height="16" rx="2" fill="{s["paper"]}"/>'
                f'<text x="{cx}" y="{cy + 4}" fill="{s["soft"]}" font-size="12" font-weight="500" '
                f'font-family="{CJK_SANS}" text-anchor="middle">{esc(text)}</text>')
    return label(s, cx, cy, text)


def zone_i18n(s, x, y, w, h, text, lx=None) -> str:
    """Zone rect with its label; `lx` = left edge of the label (default 12px inside the zone)."""
    lx = x + 12 if lx is None else lx
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{ink(s, 0.02)}" stroke="{ink(s, 0.10)}" stroke-width="0.8"/>']
    if has_cjk(text):
        lw = r4(text_w(text, 12) + 12)
        out.append(f'<rect x="{lx}" y="{y - 8}" width="{lw}" height="16" rx="2" fill="{s["paper"]}"/>')
        out.append(f'<text x="{lx + lw / 2}" y="{y + 4}" fill="{s["soft"]}" font-size="12" font-weight="500" '
                   f'font-family="{CJK_SANS}" text-anchor="middle">{esc(text)}</text>')
    else:
        lw = r4(mono_w(text, 7, 0.14) + 12)
        out.append(f'<rect x="{lx}" y="{y + 4}" width="{lw}" height="12" rx="2" fill="{s["paper"]}"/>')
        out.append(f'<text x="{lx + lw / 2}" y="{y + 13}" fill="{s["soft"]}" font-size="7" font-family="{MONO}" '
                   f'text-anchor="middle" letter-spacing="0.14em">{text}</text>')
    return "".join(out)


SCHEME_TEXT = {
    "en": dict(
        zones=[("1  Input", "SMARD DE-LU prices"), ("2  Forecasting", "before gate closure"),
               ("3  Reinforcement learning", "offline: fit 2021–24"),
               ("4  MILP scheduling", "each delivery day D"), ("5  Output", "test 2026, retrospective")],
        real=(["Realised prices"], ["of D: reward and", "settlement only"]),
        pub=(["Published prices"], ["delivery days ≤ D−1"]),
        xgb=(["XGBoost forecast"], ["prices of D and D+1", "forward-fold training"]),
        cache=(["Reward cache"], ["MILPs for every start /", "end SoC of every day"]),
        fqi=(["Fitted Q iteration"], ["ExtraTrees, γ = 0.9", "offline batch RL"]),
        val=(["Terminal SoC value"], ["W(φ_D, e): worth of", "ending day D at level e"]),
        milp=(["24 h MILP"], ["forecast revenue", "+ terminal value"]),
        bt=(["Chained backtest"], ["net revenue, share", "of the ISEA index"]),
        isea=(["ISEA index"], ["RWTH Aachen, daily", "day-ahead revenue"]),
        l=dict(reward="REWARD", fc="FORECASTS + φ", samples="SAMPLES", q="Q FUNCTION", phi="φ_D", value="VALUE",
               plan="DAY PLAN", fcd="FORECAST OF D", settle="SETTLEMENT", ref="REFERENCE"),
        leg=["Published prices", "Realised prices", "Forecast", "Reinforcement learning", "MILP", "Evaluation", "External reference"]),
    "zh": dict(
        zones=[("1  输入", "SMARD 德卢日前价格"), ("2  价格预测", "关闸 D−1 12:00 前"),
               ("3  强化学习", "离线：2021–24 年拟合"),
               ("4  MILP 调度", "每个交付日 D"), ("5  输出", "2026 年回溯测试")],
        real=(["D 日实际价格"], ["只用于奖励和结算"]),
        pub=(["已公布价格"], ["交付日 ≤ D−1"]),
        xgb=(["XGBoost 价格预测"], ["D 和 D+1 两天", "前向折训练"]),
        cache=(["奖励缓存"], ["每天所有起止电量", "组合的 MILP 结果"]),
        fqi=(["拟合 Q 迭代"], ["ExtraTrees，γ = 0.9", "离线批量强化学习"]),
        val=(["日末电量价值"], ["W(φ_D, e)：D 日结束", "时停在 e 档值多少"]),
        milp=(["24 小时 MILP"], ["预测收益 + 日末价值", "输出 15 分钟计划"]),
        bt=(["链式回测"], ["净收益，占 ISEA", "指数的比例"]),
        isea=(["ISEA 收益指数"], ["亚琛工大，逐日", "日前收益"]),
        l=dict(reward="奖励", fc="预测与 φ", samples="训练样本", q="Q 函数", phi="φ_D", value="价值",
               plan="当天计划", fcd="D 日预测", settle="只用于结算", ref="外部参照"),
        leg=["已公布价格", "实际价格（虚线）", "价格预测", "强化学习", "MILP 调度", "评估", "外部参照"]),
}

NAME_PX, SUB_PX, LAB_PX, LAB_CJK_PX = 17, 13.5, 11.5, 13


def sub_d(t: str) -> str:
    """Escape, then typeset phi_D with a real subscript."""
    return esc(t).replace("φ_D", 'φ<tspan baseline-shift="sub" font-size="75%">D</tspan>')


def tw(text: str, size: float, latin: float) -> float:
    return sum(size if is_wide(c) else size * latin for c in text)


MOD = {  # colour per role (reference: tinted box, coloured border and name, arrows in the source colour)
    "light": dict(c=dict(pub="#4f46e5", real="#7c3aed", fc="#0891b2", rl="#ea580c", milp="#2563eb", out="#334155",
                         ext="#059669"),
                  ink="#0f172a", sub="#475569", line="#475569", node="#ffffff", paper="#ffffff",
                  zone="#f8fafc", zone_stroke="#e2e8f0", tint=0.07, tint_focal=0.14),
    "dark": dict(c=dict(pub="#818cf8", real="#c4b5fd", fc="#22d3ee", rl="#fb923c", milp="#60a5fa", out="#cbd5e1",
                        ext="#34d399"),
                 ink="#f1f5f9", sub="#cbd5e1", line="#94a3b8", node="#161b22", paper="#0d1117",
                 zone="#131820", zone_stroke="#2a313c", tint=0.12, tint_focal=0.22),
}


def hexa(c: str, a: float) -> str:
    c = c.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{a})"


def node_big(s, x, y, w, h, color, focal, names, subs) -> str:
    """Node: white (or tinted, if focal) box with a module-coloured border; 17 px name, 13.5 px sublabel."""
    k = s["mod"]
    for t in names:
        assert tw(t, NAME_PX, 0.56) <= w - 16, (t, tw(t, NAME_PX, 0.56), w)
    for t in subs:
        assert tw(t, SUB_PX, 0.53) <= w - 16, (t, tw(t, SUB_PX, 0.53), w)
    block = 22 * len(names) + 6 + 18 * len(subs)
    yy = y + (h - block) / 2
    cx = x + w / 2
    fill = hexa(color, k["tint_focal"] if focal else k["tint"])
    sw = "2.4" if focal else "1.6"
    out = [f'<g data-node="{esc(names[0])}">',
           f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{k["node"]}"/>',
           f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{color}" stroke-width="{sw}"/>']
    for i, t in enumerate(names):
        out.append(f'<text x="{cx}" y="{yy + 17 + 22 * i:.1f}" fill="{color}" font-size="{NAME_PX}" font-weight="600" '
                   f'font-family="{CJK_SANS}" text-anchor="middle">{sub_d(t)}</text>')
    y0 = yy + 22 * len(names) + 6
    for i, t in enumerate(subs):
        out.append(f'<text x="{cx}" y="{y0 + 14 + 18 * i:.1f}" fill="{k["sub"]}" font-size="{SUB_PX}" '
                   f'font-family="{CJK_SANS}" text-anchor="middle">{sub_d(t)}</text>')
    out.append("</g>")
    return "".join(out)


def lab_w(text: str) -> int:
    return r4((tw(text, LAB_CJK_PX, 0.6) if has_cjk(text) else len(text) * LAB_PX * 0.68) + 14)


def lab(s, cx, cy, text, color=None) -> str:
    w = lab_w(text)
    if has_cjk(text):
        font = f'font-size="{LAB_CJK_PX}" font-weight="600" font-family="{CJK_SANS}"'
    else:
        font = f'font-size="{LAB_PX}" font-weight="500" font-family="{MONO}" letter-spacing="0.06em"'
    return (f'<rect x="{cx - w / 2}" y="{cy - 8}" width="{w}" height="16" rx="2" fill="{s["paper"]}"/>'
            f'<text x="{cx}" y="{cy + 4}" fill="{color or s["mod"]["line"]}" {font} text-anchor="middle">{sub_d(text)}</text>')


def arrow(s, slug, d, color, dashed=False) -> str:
    name = color.lstrip("#")
    extra = ' stroke-dasharray="7,5"' if dashed else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"{extra} '
            f'marker-end="url(#{slug}-arrow-{name})"/>')


def scheme(s, sid, lang="en") -> tuple[str, int, int]:
    """Architecture, left to right: input | forecasting | reinforcement learning | MILP | output.
    Rows A-B carry the offline learning, row C the daily decision path."""
    T, L = SCHEME_TEXT[lang], SCHEME_TEXT[lang]["l"]
    ZW = [208, 208, 256, 208, 208]
    GAP, M = 32, 32
    ZX = [M]
    for w in ZW[:-1]:
        ZX.append(ZX[-1] + w + GAP)
    W = ZX[-1] + ZW[-1] + M
    NX = [x + 16 for x in ZX]
    NWS = [w - 32 for w in ZW]
    CX = [x + w / 2 for x, w in zip(ZX, ZW)]
    NH = 96
    A, B, C = 160, 296, 432
    ZY0, ZY1 = 64, 600
    H = 692
    o = []
    k = s["mod"]
    c = k["c"]
    o.append("<defs>" + "".join(
        f'<marker id="{sid}-arrow-{v.lstrip("#")}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
        f'<polygon points="0 0, 8 3, 0 6" fill="{v}"/></marker>' for v in sorted(set(c.values()))) + "</defs>")
    zone_col = [c["pub"], c["fc"], c["rl"], c["milp"], c["out"]]
    # module zones (columns): quiet neutral panels, a 4 px top bar and the title in the module colour
    for (title, cap), x, w, zc in zip(T["zones"], ZX, ZW, zone_col):
        o.append(f'<rect x="{x}" y="{ZY0}" width="{w}" height="{ZY1 - ZY0}" rx="8" fill="{k["zone"]}" '
                 f'stroke="{k["zone_stroke"]}" stroke-width="1.2"/>')
        o.append(f'<path d="M{x},{ZY0 + 8} Q{x},{ZY0} {x + 8},{ZY0} H{x + w - 8} Q{x + w},{ZY0} {x + w},{ZY0 + 8} V{ZY0 + 4} '
                 f'H{x} Z" fill="{zc}"/>')
        o.append(f'<text x="{x + 16}" y="{ZY0 + 30}" fill="{zc}" font-size="16" font-weight="700" '
                 f'font-family="{CJK_SANS}">{esc(title)}</text>')
        o.append(f'<text x="{x + 16}" y="{ZY0 + 50}" fill="{k["sub"]}" font-size="12.5" '
                 f'font-family="{CJK_SANS}">{esc(cap)}</text>')
        assert tw(cap, 12.5, 0.55) <= w - 24, cap
    P = lambda d, col, **kw: o.append(arrow(s, sid, d, col, **kw))
    mid = lambda r: r + NH / 2
    r8 = 8
    # daily path (row C)
    P(f"M{NX[0] + NWS[0]},{mid(C)} H{NX[1]}", c["pub"])                       # published -> XGB
    P(f"M{NX[1] + NWS[1]},{mid(C)} H{NX[2]}", c["fc"])                        # XGB -> value (phi_D)
    P(f"M{NX[2] + NWS[2]},{mid(C)} H{NX[3]}", c["rl"])                        # value -> MILP
    gx = ZX[3] + ZW[3] + GAP / 2                                               # MILP -> backtest (row B)
    P(f"M{NX[3] + NWS[3]},{mid(C)} H{gx - r8} Q{gx},{mid(C)} {gx},{mid(C) - r8} V{mid(B) + r8} "
      f"Q{gx},{mid(B)} {gx + r8},{mid(B)} H{NX[4]}", c["milp"])
    yc = C + NH + 32                                                           # forecast of D -> MILP, below row C
    P(f"M{CX[1]},{C + NH} V{yc - r8} Q{CX[1]},{yc} {CX[1] + r8},{yc} H{CX[3] - r8} Q{CX[3]},{yc} {CX[3]},{yc - r8} V{C + NH}",
      c["fc"])
    # offline learning (rows A-B)
    ya1, ya2 = A + 34, A + 62
    P(f"M{NX[0] + NWS[0]},{ya1} H{NX[2]}", c["real"], dashed=True)            # realised -> cache (reward)
    P(f"M{CX[1]},{C} V{ya2 + r8} Q{CX[1]},{ya2} {CX[1] + r8},{ya2} H{NX[2]}", c["fc"])   # XGB -> cache
    P(f"M{CX[2]},{A + NH} V{B}", c["rl"])                                     # cache -> FQI
    P(f"M{CX[2]},{B + NH} V{C}", c["rl"])                                     # FQI -> value
    yt = A - 24                                                                # realised -> settlement
    P(f"M{CX[0]},{A} V{yt + r8} Q{CX[0]},{yt} {CX[0] + r8},{yt} H{CX[4] - r8} Q{CX[4]},{yt} {CX[4]},{yt + r8} V{B}",
      c["real"], dashed=True)
    P(f"M{CX[4]},{C} V{B + NH}", c["ext"])                                    # ISEA -> backtest
    # labels in the colour of their arrow (opaque masks, 8 px from the stroke)
    def LB(cx, cy, key, col):
        o.append(lab(dict(s, paper=k["zone"]), cx, cy, L[key], col))
    LB(CX[1], ya1 - 16, "reward", c["real"])
    LB(CX[1] + 8 + lab_w(L["fc"]) / 2, (ya2 + C) / 2, "fc", c["fc"])
    LB(CX[2] + 8 + lab_w(L["samples"]) / 2, (A + NH + B) / 2, "samples", c["rl"])
    LB(CX[2] + 8 + lab_w(L["q"]) / 2, (B + NH + C) / 2, "q", c["rl"])
    o.append(lab(s, (NX[1] + NWS[1] + NX[2]) / 2, mid(C) - 18, L["phi"], c["fc"]))
    o.append(lab(s, (NX[2] + NWS[2] + NX[3]) / 2, mid(C) - 18, L["value"], c["rl"]))
    LB(gx - 8 - lab_w(L["plan"]) / 2, (mid(B) + C) / 2 + 20, "plan", c["milp"])
    LB(CX[2], yc + 18, "fcd", c["fc"])
    LB(CX[3], yt + 18, "settle", c["real"])
    LB(CX[4] + 8 + lab_w(L["ref"]) / 2, (B + NH + C) / 2, "ref", c["ext"])
    for w0, gap_txt in ((NX[2] - NX[1] - NWS[1], L["phi"]), (NX[3] - NX[2] - NWS[2], L["value"])):
        assert lab_w(gap_txt) <= w0 - 8, gap_txt
    # nodes (after arrows and labels)
    N = lambda i, row, key, col, focal=False: o.append(node_big(s, NX[i], row, NWS[i], NH, col, focal, *T[key]))
    N(0, A, "real", c["real"])
    N(0, C, "pub", c["pub"])
    N(1, C, "xgb", c["fc"])
    N(2, A, "cache", c["rl"])
    N(2, B, "fqi", c["rl"], True)
    N(2, C, "val", c["rl"])
    N(3, C, "milp", c["milp"], True)
    N(4, B, "bt", c["out"])
    N(4, C, "isea", c["ext"])
    # legend strip: one swatch per role
    lg = T["leg"]
    items = [(c["pub"], lg[0], False), (c["real"], lg[1], True), (c["fc"], lg[2], False), (c["rl"], lg[3], False),
             (c["milp"], lg[4], False), (c["out"], lg[5], False), (c["ext"], lg[6], False)]
    yl = ZY1 + 40
    o.append(f'<line x1="{M}" y1="{yl - 16}" x2="{W - M}" y2="{yl - 16}" stroke="{k["zone_stroke"]}" stroke-width="1"/>')
    x = M
    for col, text, dashed in items:
        if dashed:
            o.append(f'<line x1="{x}" y1="{yl + 7.5}" x2="{x + 18}" y2="{yl + 7.5}" stroke="{col}" stroke-width="2" stroke-dasharray="5,3"/>')
        else:
            o.append(f'<rect x="{x}" y="{yl + 1}" width="16" height="13" rx="3" fill="{hexa(col, k["tint_focal"])}" stroke="{col}" stroke-width="1.6"/>')
        o.append(f'<text x="{x + 26}" y="{yl + 12}" fill="{k["sub"]}" font-size="13.5" font-family="{CJK_SANS}">{esc(text)}</text>')
        x += 26 + tw(text, 13.5, 0.55) + 32
    assert x <= W, x
    return "".join(o), W, H


DIAGRAMS = {
    "timing": ("Timeline", "When the battery decides for delivery day D",
               "Timeline of the day-ahead information rule: prices of D−1 are known from D−2 13:00, the whole day D is "
               "planned and locked at gate closure D−1 12:00, and the prices of D published at D−1 13:00 are used only to settle.",
               timing),
    "scheme": ("Architecture", "MILP with a terminal SoC value learned by fitted Q iteration",
               "Five modules from input to output: published SMARD prices feed an XGBoost forecaster; an offline reward "
               "cache trains fitted Q iteration, whose terminal SoC value enters the 24 h MILP; the day plan is settled at "
               "realised prices in a chained backtest and compared with the ISEA index.",
               lambda s, sid: scheme(s, sid, "en")),
    "scheme-zh": ("架构", "MILP + 拟合 Q 迭代学到的日末电量价值",
                  "从输入到输出的五个模块：已公布的 SMARD 价格进入 XGBoost 预测；离线奖励缓存训练拟合 Q 迭代，"
                  "学到的日末电量价值写进 24 小时 MILP；当天计划按实际价格链式回测结算，并与 ISEA 指数比较。",
                  lambda s, sid: scheme(s, sid, "zh")),
}


def main() -> None:  # noqa: D103
    for slug, (eyebrow, title, desc, fn) in DIAGRAMS.items():
        for variant, s in SKINS.items():
            sid = f"{slug}-dark" if variant == "dark" else slug
            if slug.startswith(("scheme", "timing")):
                s = dict(s, mod=MOD[variant], variant=variant, paper=MOD[variant]["paper"])
            body, w, h = fn(s, sid)
            (OUT / f"{sid}.html").write_text(page(slug, variant, eyebrow, title, desc, w, h, body, s), encoding="utf-8")
            print("wrote", OUT / f"{sid}.html")


if __name__ == "__main__":
    main()
