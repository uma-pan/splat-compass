"""1試合の詳しい振り返り（アプリの「直前の試合」タブ）。

試合が終わったら、その試合だけを調べて出す。全試合を集計するまとめのページより軽い。
- 結果と、自分のキル・デス・スペシャルを普段と比べた数字
- 勝率の推移（自分が倒した・倒された・スペシャルを使った時刻の印つき）とカウントの推移
- 振り返りの指摘
- 倒された1回ごとの状況（録画があればその場面を再生）
- 勝率が大きく動いた場面
- 8人のブキと、それぞれが倒された回数
- マップ画面と塗りの割合
"""
import html
import json
import os

from . import coach, feedback, learning, weapons, winprob
from .hud import SLOT_NAMES, STAGES

ESC = html.escape


def _mmss(sec):
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def recent(store, n=20):
    ms = sorted([m for m in store.matches() if m.get("video_end") and not m.get("interrupted")],
                key=lambda m: m.get("played_at", ""))
    return ms, ms[-n:][::-1]


def _chart(m, pr, me_events, kills, W=720, H=200):
    """勝率の線と、自分の出来事の印。カウントの進み（100 − 残り）も薄く描く。"""
    dur = max(m["video_end"] - m["video_start"], 1)
    x = lambda rel: 40 + (W - 50) * min(max(rel / dur, 0), 1)
    y = lambda p: 10 + (H - 40) * (1 - p)
    parts = [f'<line x1="40" y1="{y(0.5):.1f}" x2="{W - 10}" y2="{y(0.5):.1f}" stroke="#ccc" stroke-dasharray="4 4"/>',
             f'<text x="34" y="{y(1) + 4:.1f}" text-anchor="end" font-size="11" fill="#888">100%</text>',
             f'<text x="34" y="{y(0.5) + 4:.1f}" text-anchor="end" font-size="11" fill="#888">50%</text>',
             f'<text x="34" y="{y(0) + 4:.1f}" text-anchor="end" font-size="11" fill="#888">0%</text>']
    for mm in range(0, int(dur // 60) + 1):
        parts.append(f'<text x="{x(mm * 60):.1f}" y="{H - 8}" text-anchor="middle" font-size="11" fill="#888">{mm}:00</text>')
    tl = m.get("timeline") or []
    if tl:  # カウントの進み（残りが減るほど上に行く。味方は青、敵は橙）
        for col, color in ((1, "#9ec5fe"), (2, "#ffd0a8")):
            pts = " ".join(f"{x(r[0] - m['video_start']):.1f},{y(1 - r[col] / 100):.1f}" for r in tl if r[col] is not None)
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    if pr is not None:
        tt, pp = pr
        pts = " ".join(f"{x(t - m['video_start']):.1f},{y(p):.1f}" for t, p in zip(tt, pp))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#1f6feb" stroke-width="2"/>')
    for rel, kind in me_events:
        color = {"倒された": "#c0392b", "スペシャル": "#d4a000"}[kind]
        parts.append(f'<circle cx="{x(rel):.1f}" cy="{H - 26}" r="4" fill="{color}"><title>{_mmss(rel)} {kind}</title></circle>')
    for rel in kills:
        parts.append(f'<circle cx="{x(rel):.1f}" cy="{H - 34}" r="4" fill="#1a7f37"><title>{_mmss(rel)} 倒した</title></circle>')
    legend = ('<div class="muted" style="font-size:12px">青い線: 勝率　薄い線: カウントの進み（上ほど勝ちに近い。青が味方・橙が敵）　'
              '<span style="color:#1a7f37">●</span>倒した　<span style="color:#c0392b">●</span>倒された　'
              '<span style="color:#d4a000">●</span>スペシャル</div>')
    return f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;background:#fff;border-radius:8px">{"".join(parts)}</svg>{legend}'


def html_for(store, link, video_dir=None, file=None):
    ms, rec = recent(store)
    if not ms:
        return '<p class="muted">保存された試合がまだありません</p>'
    m = next((x for x in ms if x["_file"] == file), None) if file else None
    m = m or ms[-1]
    i = ms.index(m)
    base = feedback.baseline(ms[:i])
    me = m.get("me_slot")
    dur = m["video_end"] - m["video_start"]
    res = m.get("result") or "勝敗不明"
    model = winprob.load_model(store) if m.get("rule") == "ガチエリア" else None
    pr = learning.predict_match(store, m, model)
    wp = coach.wp_func(store, m)

    # 試合の選び直し
    opts = "".join(f'<option value="{ESC(x["_file"])}"{" selected" if x is m else ""}>'
                   f'{ESC(x.get("played_at", "")[5:16].replace("T", " "))} {ESC(x.get("rule") or "ルール不明")} {ESC(x.get("result") or "")}</option>'
                   for x in rec)
    out = [f'<div class="lm-head"><select onchange="loadLast(this.value)">{opts}</select></div>']
    names = weapons.weapons_of(store, m) if m.get("weapon_icons") else None
    my_weapon = names[SLOT_NAMES.index(me)] if names and me and names[SLOT_NAMES.index(me)] else None
    stage = m.get("stage")
    out.append(f'<h2 class="lm-title">{ESC(m.get("played_at", "")[:16].replace("T", " "))}　{ESC(m.get("rule") or "ルール不明")}　'
               f'{ESC(stage or "")}　<span class="res-{ESC(res)}">{ESC(res)}</span>　<span class="muted">{_mmss(dur)}'
               f'{"　" + ESC(my_weapon) if my_weapon else ""}</span></h2>')
    stem = os.path.splitext(m["_file"])[0]
    crop = os.path.join(store.root, "stages", stem + ".png")
    if os.path.exists(crop):  # ステージの名前を付ける・直す
        dl = "".join(f'<option value="{ESC(n)}">' for n in STAGES)
        msg = "ステージが分かりませんでした。名前を入れると、次から見本として使います" if not stage else "違っていたら直してください"
        out.append(f'<div class="lm-stage"><img src="{link(crop)}" height="30"> '
                   f'<input list="stagenames" value="{ESC(stage or "")}" placeholder="ステージ名" '
                   f'onchange="setStage({ESC(json.dumps(m["_file"]))}, this.value)"><datalist id="stagenames">{dl}</datalist> '
                   f'<span class="muted">{msg}</span></div>')

    # 数字
    n = feedback._numbers(m)
    if n is not None:
        mins = n["mins"]

        def cell(label, got, key, more_is_good):
            exp = base.get(key) * mins if base and base.get(key) is not None else None
            mark = ""
            if exp is not None and exp >= 0.5:
                if got >= exp * 1.4 + 0.5:
                    mark = "good" if more_is_good else "bad"
                elif got <= exp * 0.6 - 0.5:
                    mark = "bad" if more_is_good else "good"
            usual = f'<div class="muted">普段 {exp:.1f}</div>' if exp is not None else ""
            return f'<div class="lm-num {mark}"><div class="muted">{label}</div><div class="v">{got}</div>{usual}</div>'
        cells = []
        if n["kills"] is not None:
            cells.append(cell("倒した", len(n["kills"]), "kills", True))
        cells += [cell("倒された", len(n["deaths"]), "deaths", False),
                  cell("スペシャル", len(n["specials"]), "specials", True),
                  cell("溜めたまま倒された", len(n["held"]), "held", False)]
        out.append('<div class="lm-nums">' + "".join(cells) + "</div>")
    else:
        out.append('<p class="muted">この試合は自分の枠が分からなかったので、自分の数字は出せません</p>')

    # 勝率の推移
    me_events = []
    for t, rel, slot, what, *_ in m["events"]:
        if slot == me and "倒された" in what:
            me_events.append((rel, "倒された"))
        elif slot == me and what == "スペシャル発動":
            me_events.append((rel, "スペシャル"))
    kills = [k["rel"] for k in m.get("kills") or []]
    out.append('<div class="card wide"><h3>勝率の推移</h3>' + _chart(m, pr, me_events, kills) + "</div>")

    # 振り返り
    lis = "".join(f'<li class="k-{ESC(k)}"><b>{ESC(k)}</b> {ESC(t)}</li>' for k, t in feedback.match_feedback(m, base, store))
    out.append(f'<div class="card wide"><h3>振り返り</h3><ul class="fb">{lis}</ul></div>')

    # 録画があれば場面を再生できるようにする
    src = None
    if video_dir and m.get("source") and m["source"] != "camera" and os.path.exists(os.path.join(video_dir, m["source"])):
        src = link(os.path.join(video_dir, m["source"]))
        out.append('<div class="card wide"><video id="lv" controls preload="none" style="width:100%;max-width:720px"></video>'
                   '<div id="lvnow" class="muted">「再生」を押すと、その場面の少し前から流します</div></div>')

    def play_btn(t_video, label):
        if not src:
            return ""
        a, b = t_video - 8, t_video + 4
        return (f'<button class="btn" onclick="{ESC(f"lmPlay({json.dumps(src)}, {a:.1f}, {b:.1f}, {json.dumps(label, ensure_ascii=False)})")}">再生</button>')

    # 倒された場面
    ds = coach.deaths(m, wp) or []
    if ds:
        trs = "".join(f'<tr><td>{_mmss(d["rel"])}</td><td style="text-align:left">{ESC(coach.describe(d))}</td>'
                      f'<td>{play_btn(d["t"], _mmss(d["rel"]) + " 倒された")}</td></tr>' for d in ds)
        out.append(f'<div class="card wide"><h3>倒された場面 {len(ds)}回</h3>'
                   f'<table><tr><th>時刻</th><th style="text-align:left">そのときの状況</th><th></th></tr>{trs}</table></div>')

    # 勝率が大きく動いた場面
    if pr is not None:
        tt, pp = pr
        moves = []
        for k in range(len(tt)):
            j = k
            while j + 1 < len(tt) and tt[j + 1] - tt[k] <= 5:
                j += 1
            if j > k:
                moves.append((pp[j] - pp[k], tt[k], tt[j]))
        moves.sort(key=lambda v: -abs(v[0]))
        picked = []
        for d, a, b in moves:
            if abs(d) < 0.08 or len(picked) >= 6:
                break
            if all(abs(a - p[1]) > 10 for p in picked):
                picked.append((d, a, b))
        if picked:
            rows = []
            for d, a, b in sorted(picked, key=lambda v: v[1]):
                what = [f"{e[2]} {e[3]}" for e in m["events"] if a - 1 <= e[0] <= b and ("倒された" in e[3] or e[3] == "スペシャル発動")]
                what += ["自分が倒した" for k in m.get("kills") or [] if a - 1 <= k["t"] <= b]
                rel = a - m["video_start"]
                rows.append(f'<tr><td>{_mmss(rel)}</td><td class="{"good" if d > 0 else "bad"}">{d * 100:+.0f}%</td>'
                            f'<td style="text-align:left">{ESC("、".join(what[:6]) or "（出来事なし。カウントや確保の動き）")}</td>'
                            f'<td>{play_btn(a + 2, _mmss(rel) + " 勝率 " + format(d * 100, "+.0f") + "%")}</td></tr>')
            out.append('<div class="card wide"><h3>勝率が大きく動いた場面（5秒のあいだに8%以上）</h3>'
                       '<table><tr><th>時刻</th><th>勝率</th><th style="text-align:left">そのとき起きたこと</th><th></th></tr>'
                       + "".join(rows) + "</table></div>")

    # 8人
    if names:
        deaths_by = {s: 0 for s in SLOT_NAMES}
        sp_by = {s: 0 for s in SLOT_NAMES}
        for e in m["events"]:
            if "倒された" in e[3]:
                deaths_by[e[2]] += 1
            elif e[3] == "スペシャル発動":
                sp_by[e[2]] += 1
        rows = []
        for k, s in enumerate(SLOT_NAMES):
            side = "味方" if k < 4 else "相手"
            tag = "（自分）" if s == me else ""
            rows.append(f'<tr><td>{side}</td><td style="text-align:left">{ESC(names[k] or "分からない")}{tag}</td>'
                        f'<td>{deaths_by[s]}</td><td>{sp_by[s]}</td></tr>')
        out.append('<div class="card wide"><h3>8人</h3><table><tr><th></th><th style="text-align:left">ブキ</th>'
                   '<th>倒された</th><th>スペシャル</th></tr>' + "".join(rows) + "</table>"
                   '<p class="muted">倒した数は自分の分しか読めないので、倒された回数とスペシャルの回数を出しています。</p></div>')

    # マップ
    shots = m.get("map_shots") or []
    if shots:
        inks = {round(x[1]): x[2] for x in m.get("inks") or []}
        imgs = []
        for x in shots:
            ink = inks.get(round(x["rel"]))
            cap = f'{_mmss(x["rel"])}' + (f' 塗り {ink * 100:.0f}%' if ink is not None else "")
            u = link(os.path.join(store.root, "maps", x["image"]))
            imgs.append(f'<figure style="display:inline-block;margin:4px"><a href="{u}" target="_blank">'
                        f'<img loading="lazy" src="{u}" height="110"></a><figcaption class="muted">{ESC(cap)}</figcaption></figure>')
        out.append('<div class="card wide"><h3>マップ画面</h3>' + "".join(imgs) + "</div>")
    return "".join(out)
