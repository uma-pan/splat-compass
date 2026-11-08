"""貯まった記録をまとめて、ブラウザで見る1ページ（report.html）を作る。

  python -m splat_compass.report "D:\\Spla Capture"   … 録画のフォルダを渡すと「見返す場面」も入る（次からは省略可）
"""
import argparse
import datetime as dt
import html
import json
import os
import pathlib
import webbrowser
from collections import defaultdict

import numpy as np

from . import analyze, area, coach, feedback, killcam, objective, scenes, weapons, winprob
from .storage import Store

ESC = html.escape
RULES = ("ガチエリア", "ガチヤグラ", "ガチホコ", "ガチアサリ")


# ---------- 小さな部品 ----------
def pct(x):
    return "—" if x is None or x != x else f"{x * 100:.0f}%"


def num(x, d=1):
    return "—" if x is None or x != x else f"{x:.{d}f}"


def avg(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None


def card(title, body, wide=False):
    return f'<div class="card{" wide" if wide else ""}"><h3>{ESC(title)}</h3>{body}</div>'


def stat(label, value, sub=""):
    return (f'<div class="stat"><div class="v">{value}</div><div class="l">{ESC(label)}</div>'
            + (f'<div class="s">{ESC(sub)}</div>' if sub else "") + "</div>")


def table(head, rows, align=None):
    th = "".join(f"<th>{ESC(h)}</th>" for h in head)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


def bars(items, fmt=lambda v: str(v), color="#3d7bd9", maxv=None):
    """横棒グラフ。items: [(ラベル, 値)]"""
    if not items:
        return '<p class="muted">まだ記録がありません</p>'
    maxv = maxv or max(v for _, v in items) or 1
    rows = "".join(
        f'<div class="bar"><span class="bl">{ESC(k)}</span><span class="bt">'
        f'<span class="bf" style="width:{max(v, 0) / maxv * 100:.1f}%;background:{color}"></span></span>'
        f'<span class="bv">{ESC(fmt(v))}</span></div>' for k, v in items)
    return f'<div class="bars">{rows}</div>'


def line_chart(series, ymin=None, ymax=None, yfmt=lambda v: f"{v:.2f}", w=640, h=200, xlabel=""):
    """折れ線グラフ（SVG）。series: [(名前, 色, [(x, y)])]"""
    pts = [p for _, _, s in series for p in s]
    if len(pts) < 2:
        return '<p class="muted">まだ記録が足りません</p>'
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    y0 = min(ys) if ymin is None else ymin
    y1 = max(ys) if ymax is None else ymax
    if y1 <= y0:
        y1 = y0 + 1
    L, R, T, B = 44, 10, 10, 26
    X = lambda x: L + (x - x0) / (x1 - x0) * (w - L - R)
    Y = lambda y: T + (1 - (y - y0) / (y1 - y0)) * (h - T - B)
    grid = ""
    for i in range(5):
        yv = y0 + (y1 - y0) * i / 4
        grid += (f'<line x1="{L}" x2="{w - R}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" class="grid"/>'
                 f'<text x="{L - 6}" y="{Y(yv) + 4:.1f}" class="axis" text-anchor="end">{ESC(yfmt(yv))}</text>')
    lines, legend = "", ""
    for i, (name, color, s) in enumerate(series):
        if len(s) >= 2:
            d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in s)
            lines += f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"/>'
        legend += f'<span class="lg"><i style="background:{color}"></i>{ESC(name)}</span>'
    xl = f'<text x="{(w + L) / 2:.0f}" y="{h - 4}" class="axis" text-anchor="middle">{ESC(xlabel)}</text>'
    return (f'<div class="legend">{legend}</div><svg viewBox="0 0 {w} {h}" class="chart">{grid}{lines}{xl}</svg>')


# ---------- 集計 ----------
def rolling(matches, window=20):
    """新しい試合ほど右。window 試合ずつの勝率と1分あたりデス。"""
    wins, deaths = [], []
    for i in range(len(matches)):
        g = matches[max(0, i - window + 1):i + 1]
        known = [m for m in g if m.get("result") in ("勝ち", "負け")]
        if len(known) >= min(window, 10):
            wins.append((i + 1, sum(m["result"] == "勝ち" for m in known) / len(known)))
        withme = [m for m in g if m.get("me_slot")]
        if len(withme) >= min(window, 10):
            d = sum(sum(1 for e in m["events"] if e[2] == m["me_slot"] and "倒された" in e[3]) for m in withme)
            mins = sum((m["video_end"] - m["video_start"]) / 60 for m in withme)
            deaths.append((i + 1, d / mins))
    return wins, deaths


def me_stats(ms):
    withme = [m for m in ms if m.get("me_slot")]
    if not withme:
        return {}
    f = [analyze.features(m) for m in withme]
    out = {k: avg([x[k] for x in f]) for k in f[0]}
    known = [m for m in ms if m.get("result") in ("勝ち", "負け")]
    out["勝率"] = sum(m["result"] == "勝ち" for m in known) / len(known) if known else None
    out["試合数"] = len(ms)
    return out


def section_overview(ms):
    known = [m for m in ms if m.get("result") in ("勝ち", "負け")]
    wins = sum(m["result"] == "勝ち" for m in known)
    recent = ms[-20:]
    a, r = me_stats(ms), me_stats(recent)
    first = ms[0].get("played_at", "")[:10] if ms else ""
    last = ms[-1].get("played_at", "")[:10] if ms else ""
    top = (stat("試合", f"{len(ms)}", f"{first} 〜 {last}")
           + stat("勝敗", f"{wins}勝{len(known) - wins}敗", f"勝率 {pct(wins / len(known)) if known else '—'}")
           + stat("1分あたりのデス", num(a.get("自分のデス（1分あたり）"), 2), f"直近20試合 {num(r.get('自分のデス（1分あたり）'), 2)}")
           + stat("1分あたりのスペシャル", num(a.get("自分のスペシャル発動（1分あたり）"), 2),
                  f"直近20試合 {num(r.get('自分のスペシャル発動（1分あたり）'), 2)}"))
    km = [m for m in ms if "kills" in m and m.get("video_end")]
    if km:  # 倒した数は、読むようになってからの試合だけ
        mins = sum((m["video_end"] - m["video_start"]) / 60 for m in km)
        kr = sum(len(m["kills"]) for m in km) / mins
        top += stat("1分あたりの倒した数", num(kr, 2), f"{len(km)}試合分（1試合平均 {sum(len(m['kills']) for m in km) / len(km):.1f}人）")
    rows = []
    for label, key, f in (("勝率", "勝率", pct), ("自分のデス（1分あたり）", "自分のデス（1分あたり）", lambda v: num(v, 2)),
                          ("自分のスペシャル発動（1分あたり）", "自分のスペシャル発動（1分あたり）", lambda v: num(v, 2)),
                          ("スペシャルを持ったまま倒された（1分あたり）", "自分がスペシャルを持ったまま倒された（1分あたり）", lambda v: num(v, 2)),
                          ("人数不利の時間の割合", "人数不利の割合", pct), ("人数有利の時間の割合", "人数有利の割合", pct)):
        rows.append([ESC(label), f(a.get(key)), f(r.get(key))])
    w, d = rolling(ms)
    trend = line_chart([("勝率（20試合ごと）", "#3d7bd9", w)], 0, 1, pct, xlabel="試合（古い→新しい）")
    trend2 = line_chart([("1分あたりのデス（20試合ごと）", "#d9534f", d)], yfmt=lambda v: f"{v:.2f}", xlabel="試合（古い→新しい）")
    # 状況ごとのデスのしやすさ
    alive_t, deaths, sp_use, held, total = analyze.situations(ms)
    sit = []
    for key, label in [(("人数", d_), f"人数{d_}") for d_ in ("有利", "同数", "不利")]:
        tm = alive_t[key] / 60
        if tm > 0:
            sit.append((label, deaths[key] / tm))
    spr = [[f"人数{k}", str(sp_use[k][0]), num(sp_use[k][1] / sp_use[k][0], 2) if sp_use[k][0] else "—",
            pct(sp_use[k][2] / sp_use[k][0]) if sp_use[k][0] else "—"] for k in ("有利", "同数", "不利")]
    return (f'<div class="stats">{top}</div>'
            + '<div class="grid2">'
            + card("全体と直近20試合", table(["", "全体", "直近20試合"], rows))
            + card("状況ごとの倒されやすさ（生きていた1分あたりのデス）", bars(sit, lambda v: f"{v:.2f}回", "#d9534f"))
            + card("勝率の移り変わり", trend)
            + card("デスの移り変わり", trend2)
            + card("自分がスペシャルを使ったあと10秒", table(["そのときの人数", "回数", "倒した敵（平均）", "自分も倒された"], spr))
            + card("スペシャルを持ったまま倒された", f'<p class="big">{held}回</p><p class="muted">デス全体の {pct(held / total) if total else "—"}</p>')
            + "</div>")


def section_stages(ms):
    """ステージごとの勝率と、自分のキル・デスのペース、エリアの打開の成功率。"""
    st = sorted({m.get("stage") for m in ms if m.get("stage")})
    if not st:
        return '<p class="muted">ステージが分かっている試合がまだありません。試合前のルールの表示から読みます。</p>'
    rows = []
    for name in st:
        g = [m for m in ms if m.get("stage") == name and m.get("result") in ("勝ち", "負け")]
        if not g:
            continue
        w = sum(m["result"] == "勝ち" for m in g)
        by = []
        for rule in RULES:
            r = [m for m in g if m.get("rule") == rule]
            if r:
                rw = sum(m["result"] == "勝ち" for m in r)
                by.append(f"{rule.replace('ガチ', '')} {rw}-{len(r) - rw}")
        nums = [feedback._numbers(m) for m in g]
        nums = [n for n in nums if n]
        mins = sum(n["mins"] for n in nums)
        kn = [n for n in nums if n["kills"] is not None]
        kpm = sum(len(n["kills"]) for n in kn) / sum(n["mins"] for n in kn) if kn else None
        dpm = sum(len(n["deaths"]) for n in nums) / mins if mins else None
        eps = [e for m in g if m.get("rule") == "ガチエリア" and m.get("timeline") for e in area.episodes(m) if e["seconds"] >= 3]
        dk = pct(sum(e["success"] for e in eps) / len(eps)) if eps else "—"
        rows.append((len(g), [ESC(name), f"{w}勝{len(g) - w}敗", pct(w / len(g)), ESC("、".join(by)),
                              num(kpm) if kpm is not None else "—", num(dpm) if dpm is not None else "—", dk]))
    rows.sort(key=lambda r: -r[0])
    allm = [m for m in ms if m.get("result") in ("勝ち", "負け")]
    known = [m for m in allm if m.get("stage")]
    note = (f'<p class="muted">ステージが分かっている {len(known)}試合（全 {len(allm)}試合のうち）。'
            '「倒した・倒された」は1分あたりの自分の回数、「打開」はエリアで確保されてから取り返せた割合です。'
            '試合が少ないステージは、たまたまの差が大きいので参考程度に見てください。</p>')
    return card("ステージごと", table(["ステージ", "成績", "勝率", "ルール別", "倒した/分", "倒された/分", "打開"],
                                    [r for _, r in rows]) + note, wide=True)


def section_rules(ms, store=None):
    out = []
    rows = []
    for rule in RULES:
        g = [m for m in ms if m.get("rule") == rule and m.get("result") in ("勝ち", "負け")]
        if g:
            w = sum(m["result"] == "勝ち" for m in g)
            rows.append([ESC(rule), f"{w}勝{len(g) - w}敗", pct(w / len(g))])
    out.append(card("ルールごとの勝率", table(["ルール", "成績", "勝率"], rows)))
    # エリア
    am = [m for m in ms if m.get("rule") == "ガチエリア" and m.get("timeline")]
    if am:
        eps = [(m, e) for m in am for e in area.episodes(m) if e["seconds"] >= 3]

        def row(label, sel):
            g = [e for _, e in sel]
            ok = [e for e in g if e["success"]]
            return [ESC(label), str(len(g)), pct(len(ok) / len(g)) if g else "—",
                    num(avg([e["seconds"] for e in ok])) + "秒", num(avg([e["count_lost"] for e in g]))]
        base = [row("すべて", eps),
                row("確保された時点で味方4人", [x for x in eps if x[1]["ally_alive_at_start"] == 4]),
                row("確保された時点で味方3人以下", [x for x in eps if x[1]["ally_alive_at_start"] < 4]),
                row("最初に落ちたのが敵", [x for x in eps if x[1]["first_death"] == "敵"]),
                row("最初に落ちたのが味方", [x for x in eps if x[1]["first_death"] == "味方"])]
        long_ = [(m, e, area.early(e, m)) for m, e in eps if e["seconds"] > area.EARLY]

        def row2(label, sel):
            ok = [e for _, e, _ in sel if e["success"]]
            return [ESC(label), str(len(sel)), num(avg([e["seconds"] - area.EARLY for e in ok])) + "秒"]
        early = [row2("味方が1人も落ちなかった", [x for x in long_ if x[2]["ally_deaths"] == 0]),
                 row2("味方が2人以上落ちた", [x for x in long_ if x[2]["ally_deaths"] >= 2]),
                 row2("敵を1人以上倒した", [x for x in long_ if x[2]["enemy_deaths"] >= 1]),
                 row2("敵を1人も倒せなかった", [x for x in long_ if x[2]["enemy_deaths"] == 0]),
                 row2("スペシャルを使わなかった", [x for x in long_ if x[2]["n_specials"] == 0]),
                 row2("味方3人以上で最初のスペシャル", [x for x in long_ if (x[2]["alive_at_first_special"] or 0) >= 3]),
                 row2("味方2人以下で最初のスペシャル", [x for x in long_ if x[2]["alive_at_first_special"] is not None and x[2]["alive_at_first_special"] <= 2]),
                 row2("2人以上でスペシャルを合わせた", [x for x in long_ if x[2]["synced"]])]
        wl = []
        for res in ("勝ち", "負け"):
            g = [[e for mm, e in eps if mm is m] for m in am if m.get("result") == res]
            if g:
                wl.append([res, num(avg([len(x) for x in g])) + "回", num(avg([sum(e["count_lost"] for e in x) for x in g])),
                           num(avg([e["seconds"] for x in g for e in x if e["success"]])) + "秒"])
        out.append(card(f"ガチエリアの打開（{len(am)}試合・{len(eps)}回）",
                        table(["", "回数", "成功", "打開までの平均", "取られたカウント"], base), wide=True))
        out.append(card("確保されてから最初の10秒に何をしたか", '<p class="muted">10秒以内に終わらなかった打開だけで比べています</p>'
                        + table(["最初の10秒", "回数", "そこから打開まで"], early)))
        out.append(card("勝ち試合と負け試合（1試合の平均）", table(["", "確保された回数", "取られたカウント合計", "打開までの平均"], wl)))
        if store is not None:
            out.append(special_card(store, eps))
    for rule in ("ガチヤグラ", "ガチホコ"):
        g = [m for m in ms if m.get("rule") == rule and m.get("objective")]
        if g:
            eps = [e for m in g for e in objective.push_episodes(m) if e["seconds"] >= 3]
            ok = [e for e in eps if e["success"]]
            body = table(["", ""], [["押し込まれた回数", str(len(eps))], ["中央まで戻せた", pct(len(ok) / len(eps)) if eps else "—"],
                                    ["止めるまで", num(avg([e["to_stop"] for e in eps])) + "秒"],
                                    ["中央に戻すまで", num(avg([e["seconds"] for e in ok])) + "秒"]])
            out.append(card(f"{rule}の打開（{len(g)}試合）", body))
    g = [m for m in ms if m.get("rule") == "ガチアサリ" and m.get("objective")]
    if g:
        eps = [e for m in g for e in objective.asari_episodes(m)]
        out.append(card(f"ガチアサリの打開（{len(g)}試合）", table(["", ""], [
            ["ゴールを割られた回数", str(len(eps))], ["閉じるまで", num(avg([e["seconds"] for e in eps])) + "秒"],
            ["その間に取られたカウント", num(avg([e["count_lost"] for e in eps]))]])))
    return '<div class="grid2">' + "".join(out) + "</div>"


def special_card(store, eps):
    """エリアの打開で、味方のスペシャルの種類ごとに、早めに使ったときと使わなかったときの成功率を比べる。"""
    from .weapon_info import Info
    namer, info = weapons.shared_namer(store), Info(store)

    def specials_of(m):
        names = weapons.weapons_of(store, m, namer)
        return [info.get(n)["スペシャル"] if n else None for n in names] if names else None
    d = area.by_special(eps, specials_of)
    rows = []
    for sp, x in sorted(d.items(), key=lambda kv: -kv[1]["n"]):
        if x["n"] < 5:
            continue
        nu, nn = x["used"], x["n"] - x["used"]
        rows.append([ESC(sp), str(x["n"]), pct(nu / x["n"]),
                     f'{pct(x["ok_used"] / nu)}（{nu}）' if nu else "—",
                     f'{pct(x["ok_not"] / nn)}（{nn}）' if nn else "—"])
    if not rows:
        return card("打開でのスペシャルの種類ごとの使い方", '<p class="muted">ブキの名前が分かっている打開がまだ足りません</p>')
    return card(f"打開でのスペシャルの種類ごとの使い方（最初の{int(area.SP_EARLY)}秒）",
                table(["味方のスペシャル", "味方にいた打開", "早めに使った割合", "使ったときの成功", "使わなかったときの成功"], rows)
                + f'<p class="muted">{int(area.SP_EARLY)}秒より早く終わった打開は除いています。かっこの中は打開の数です。'
                  '数が少ない行はたまたまの差が大きいので、10回を超えるまでは参考程度に見てください。'
                  '使えなかったのは、溜まっていなかったからのこともあります。</p>', wide=True)


def section_killers(store, ms):
    labels, protos = killcam.load_labels(store)
    count, total, unknown = defaultdict(int), 0, 0
    for m in ms:
        for kc in m.get("killed_by") or []:
            total += 1
            n = killcam.name_of(store, os.path.join(store.root, "killcam", kc["image"]), labels, protos)
            if n:
                count[n] += 1
            else:
                unknown += 1
    if not total:
        return '<p class="muted">倒されたときの画面の記録がまだありません</p>'
    by_kind = defaultdict(list)
    for n, c in count.items():
        by_kind[killcam.kind_of(n)].append((n, c))
    kinds = [(k, sum(c for _, c in by_kind[k])) for k in ("メイン", "サブ", "スペシャル") if by_kind[k]]
    colors = {"メイン": "#3d7bd9", "サブ": "#e0a020", "スペシャル": "#c0392b"}
    body = card("何で倒されたか", bars(kinds, lambda v: f"{v}回", "#666")
                + f'<p class="muted">倒された画面 {total}回のうち、名前が分かったもの {total - unknown}回</p>')
    for k in ("メイン", "スペシャル", "サブ"):
        items = sorted(by_kind.get(k, []), key=lambda x: -x[1])[:12]
        if items:
            body += card(f"{k}", bars(items, lambda v: f"{v}回", colors[k]))
    if unknown:
        body += card("名前の無いまとまり", f'<p>{unknown}回分の名前がまだ分かりません。'
                     '<code>python -m splat_compass.killcam</code> で一覧を作り、killcam_labels.json に名前を書いてください。</p>')
    return '<div class="grid2">' + body + "</div>"


def section_statink(store, ms, st):
    """stat.ink（世界のプレイヤーの試合）と比べる。"""
    from . import statink
    sm = statink.load_summary(store)
    if not sm:
        return card("世界のプレイヤーと比べる（stat.ink）",
                    '<p class="muted">stat.ink の公開データを取り込むと、ブキごとの世界での勝率や、編成だけで見た勝率と比べられます。'
                    '<code>statink.bat</code>（<code>python -m splat_compass.statink</code>）で取り込めます。</p>', wide=True)
    # 全ルールまとめたブキの勝率（上げた本人を除く）
    allwr = defaultdict(lambda: [0, 0])
    for md in sm["modes"].values():
        for n, (c, w) in md["wr"].items():
            allwr[n][0] += c
            allwr[n][1] += w
    rows = []
    for n, (c, w) in sorted(st["enemy"].items(), key=lambda x: -x[1][0]):
        if c < 3:
            continue
        g = allwr.get(norm_name(n))
        rows.append([ESC(n), f"{pct(1 - w / c)}（{c}試合）", pct(g[1] / g[0]) if g and g[0] >= 50 else "—"])
    out = card("相手にいたブキ: 自分が負けた割合と、世界でのそのブキの勝率",
               table(["ブキ", "相手にいた時に自分が負けた割合", "世界でのそのブキの勝率"], rows[:20])
               + '<p class="muted">世界の勝率が50%より高いのに、自分が負けた割合がもっと高いブキは、自分が特に苦手な相手かもしれません。</p>')
    # 編成だけで見た勝率が、自分の試合の勝ち負けを言い当てるか
    comp = []
    namer = weapons.shared_namer(store)
    for m in ms:
        if m.get("result") not in ("勝ち", "負け") or not m.get("me_slot") or not m.get("rule"):
            continue
        names = weapons.weapons_of(store, m, namer)
        if not names or any(x is None for x in names):
            continue
        me = analyze.SLOTS.index(m["me_slot"])
        ally = names[:4] if me < 4 else names[4:]
        enemy = names[4:] if me < 4 else names[:4]
        p = statink.comp_prior(sm, m["rule"], ally, enemy)
        if p is not None:
            comp.append((p, m["result"] == "勝ち"))
    if len(comp) >= 20:
        comp.sort()
        k = len(comp) // 3
        parts = [("編成が不利な試合", comp[:k]), ("中くらい", comp[k:2 * k]), ("編成が有利な試合", comp[2 * k:])]
        trs = [[lb, f"{np.mean([p for p, _ in g]) * 100:.0f}%", pct(np.mean([w for _, w in g])), str(len(g))] for lb, g in parts]
        out += card("編成だけで見た勝率（stat.ink のブキの強さから）と、実際の勝率",
                    table(["", "編成だけで見た勝率（平均）", "実際に勝った割合", "試合"], trs)
                    + '<p class="muted">上から下へ実際の勝率が上がっていれば、編成の有利さが自分の試合でも効いているということです。</p>')
    out += card("取り込んだ stat.ink のデータ",
                f'<p class="muted">{sm["battles"]:,}試合（{len(sm["files"])}日分、{ESC(sm.get("made_at", ""))} に集計）。'
                'Xマッチとバンカラマッチのガチルールだけを使い、ブキごとの数字からは試合を上げた本人を除いています。'
                '上げているのはやり込んでいる人が多いので、腕前は平均より高めです。データ: <a href="https://stat.ink/" target="_blank">stat.ink</a></p>', wide=True)
    return out


def norm_name(n):
    from .weapon_info import norm
    return norm(n)


def section_weapons(store, ms):
    st, rate = weapons.stats(store, ms)
    if rate is None:
        return ('<p class="muted">ブキのアイコンの記録がまだありません。前の録画からは '
                '<code>python -m splat_compass.weapons --extract</code> で切り出せます。</p>')
    known = [m for m in ms if m.get("result") in ("勝ち", "負け")]
    base = sum(m["result"] == "勝ち" for m in known) / len(known) if known else 0.5
    MIN = 3

    def rows(side, order):
        r = [(n, c, w / c) for n, (c, w) in st[side].items() if c >= MIN]
        r.sort(key={"low": lambda x: (x[2], -x[1]), "high": lambda x: (-x[2], -x[1]),
                    "many": lambda x: -x[1]}[order])
        return [(f"{n}（{c}試合）", v) for n, c, v in r]

    def chart(items, color):
        if not items:
            return f'<p class="muted">{MIN}試合以上出てきたブキがまだありません</p>'
        return bars(items[:15], pct, color, 1.0)

    body = card("名前の分かったアイコン", f'<p>{pct(rate)}（全体の勝率 {pct(base)}）。'
                '残りは <code>python -m splat_compass.weapons --label</code> で名前を付けられます。'
                '試合数が少ないブキの勝率は、たまたまの影響が大きいので参考程度に見てください。</p>', wide=True)
    body += card("自分が使ったブキ", chart(rows("me", "many"), "#3d7bd9"))
    body += card("相手にいると負けやすいブキ", chart(rows("enemy", "low"), "#c0392b"))
    body += card("味方にいると勝ちやすいブキ", chart(rows("ally", "high"), "#27ae60"))
    body += card("相手にいても勝てているブキ", chart(rows("enemy", "high"), "#888"))
    # 種類・射程・スペシャルでまとめた勝率（1つ1つのブキより試合数が多く、数字が安定する）
    try:
        gs = weapons.group_stats(store, ms)
    except Exception as e:
        gs = {}
        body += card("編成でまとめた勝率", f'<p class="muted">作れませんでした: {ESC(str(e))}</p>', wide=True)
    for title, rs in gs.items():
        if title.endswith("の数") or "の数（" in title:
            items = [(f"{k}（{c}試合）", w / c) for k, c, w in rs if c]
        else:
            items = [(f"{k}（{c}試合）", w / c) for k, c, w in sorted(rs, key=lambda x: x[2] / x[1]) if c >= 5]
        body += card(title, bars(items[:15], pct, "#7b61c9", 1.0) if items else '<p class="muted">まだ試合が足りません</p>')
    body += section_statink(store, ms, st)
    body += card("ブキの情報について", '<p class="muted">射程の長い・短いは、メインのブキごとのおおまかな目安です。'
                 'サブ・スペシャルは攻略サイトの一覧から入れたもので、新しい亜種などは空のことがあります。'
                 '「ブキの確認」タブの下の表で直せます。</p>', wide=True)
    return '<div class="grid2">' + body + "</div>"


def section_model(store, ms):
    hist_path = os.path.join(store.root, "model_history.json")
    try:
        with open(hist_path, encoding="utf-8") as f:
            hist = json.load(f)
    except (OSError, ValueError):
        hist = []
    am = [m for m in ms if m.get("rule") == "ガチエリア" and m.get("timeline") and m.get("result") in ("勝ち", "負け")]
    out = []
    if hist:
        pts_auc = [(i + 1, h["oof_auc"]) for i, h in enumerate(hist)]
        lo, hi = min(p[1] for p in pts_auc), max(p[1] for p in pts_auc)
        out.append(card("モデルの育ち具合（作り直すたびの精度）",
                        line_chart([("AUC（高いほど良い）", "#3d7bd9", pts_auc)], lo - 0.02, hi + 0.02,
                                   yfmt=lambda v: f"{v:.3f}", xlabel="作り直した回数（古い→新しい）")
                        + f'<p class="muted">最新: {hist[-1]["matches"]}試合 / AUC {hist[-1]["oof_auc"]:.3f} / 対数損失 {hist[-1]["oof_logloss"]:.3f}</p>'))
    if len(am) >= 20:
        X, y, g, ts, p, w, ya = winprob.oof_predict(am, aux=winprob.USE_AUX, spl=winprob.USE_SPLINE)
        cal = winprob.calibration(y, p, w)
        out.append(card("予測した勝率と実際の勝率", table(["予測した勝率", "予測の平均", "実際に勝った割合"],
                        [[f"{a * 100:.0f}〜{min(b, 1) * 100:.0f}%", pct(pm), pct(ym)] for a, b, pm, ym, n in cal])))
        vals = winprob.event_values(am, ts, g, p, after=15.0)
        items = []
        for key in [("自分", "倒した"), ("敵", "倒された"), ("自分", "スペシャル発動"), ("味方", "スペシャル発動"),
                    ("味方", "倒された"), ("自分", "倒された"), ("敵", "スペシャル発動")]:
            v = vals.get(key, [])
            if v:
                items.append((f"{key[0]}が{key[1]}", float(np.mean(v))))
        rows = "".join(
            f'<div class="bar"><span class="bl">{ESC(k)}</span><span class="bt center">'
            f'<span class="bf {"pos" if v >= 0 else "neg"}" style="width:{min(abs(v) / 0.08, 1) * 50:.1f}%"></span></span>'
            f'<span class="bv">{v * 100:+.1f}%</span></div>' for k, v in items)
        out.append(card("出来事の前後で勝率がどれだけ動いたか（15秒後まで）", f'<div class="bars">{rows}</div>'))
    else:
        out.append(card("勝率モデル", f'<p class="muted">ガチエリアの試合が20試合たまると作られます（いま {len(am)}試合）</p>'))
    return '<div class="grid2">' + "".join(out) + "</div>"


def section_feedback(store, link=None, app=False):
    blocks = []
    color = {"気になる": "#c0392b", "良い": "#1a7f37", "情報": "#6b7280"}
    for m, items in feedback.for_latest(store, 15):
        res = m.get("result") or ""
        lis = "".join(f'<li><span class="tag" style="background:{color[k]}">{ESC(k)}</span>{ESC(t)}</li>' for k, t in items)
        fix = ""
        if app:  # アプリの画面では、勝敗を読み違えていたら直せる
            rc = m.get("result_check") or {}
            img = ""
            if rc.get("image") and link:
                img = (f'<a href="{link(os.path.join(store.root, "results", rc["image"]))}" target="_blank">'
                       f'<img loading="lazy" src="{link(os.path.join(store.root, "results", rc["image"]))}" height="54" '
                       f'title="勝敗を読んだ画面（WIN! {rc.get("win")} / LOSE {rc.get("lose")}）"></a>')
            f = ESC(m["_file"])
            fix = (f'<div class="fix">{img}<span class="muted">勝敗が違っていたら:</span>'
                   f'<button onclick="fixResult(\'{f}\', \'勝ち\')">勝ちに直す</button>'
                   f'<button onclick="fixResult(\'{f}\', \'負け\')">負けに直す</button>'
                   + ('<span class="muted">（手で直した）</span>' if m.get("result_fixed") else "") + "</div>")
        ds = coach.deaths(m, coach.wp_func(store, m)) or []
        dl = ""
        if ds:
            trs = "".join(f'<tr><td>{coach._mmss(d["rel"])}</td><td style="text-align:left">{ESC(coach.describe(d))}</td></tr>' for d in ds)
            dl = (f'<details><summary class="muted">倒された場面 {len(ds)}回</summary>'
                  f'<table><tr><th>時刻</th><th style="text-align:left">そのときの状況</th></tr>{trs}</table></details>')
        blocks.append(f'<section><h2>{ESC(m.get("played_at", "")[:16].replace("T", " "))}　{ESC(m.get("rule") or "ルール不明")}　'
                      f'<span class="res-{ESC(res)}">{ESC(res)}</span></h2>{fix}<ul class="fb">{lis}</ul>{dl}</section>')
    return ('<p class="muted">新しい15試合を、それぞれの試合より前の自分（普段の自分）と比べて、目立ったところだけを挙げています。'
            '時刻は試合開始からの秒です。</p>' + trend_card(store) + "".join(blocks))


def trend_card(store):
    """最近の倒され方を、それより前の自分と比べる表。"""
    try:
        rows, nn, no, r, o = coach.trend(store)
    except Exception:
        return ""
    if not nn or not no:
        return ""
    trs = []
    for label, n, sh, prev in rows:
        diff = (sh - prev) * 100 if prev is not None else 0
        mark = "多くなった" if diff >= 8 else ("少なくなった" if diff <= -8 else "")
        trs.append([ESC(label), f"{n}回", pct(sh), pct(prev) if prev is not None else "—", mark])
    return card(f"倒され方の傾向（最近{r}試合と、それより前の{o}試合）",
                table(["倒されたときの状況", "最近の回数", "最近の割合", "前の割合", ""], trs)
                + '<p class="muted">1回の倒され方に当てはまるものは全部数えるので、割合を足すと100%を超えます。'
                  '「人数が不利」は倒される直前の上の表示の人数（自分を含む）で見ています。</p>', wide=True)


PAGE = """<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8"><title>splat-compass</title>
<style>
:root {{ --bg:#f3f4f6; --card:#fff; --ink:#1f2328; --muted:#6b7280; --line:#e5e7eb; --accent:#3d7bd9; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:"Meiryo","Hiragino Sans",sans-serif; background:var(--bg); color:var(--ink); }}
header {{ background:#1f2937; color:#fff; padding:14px 22px; display:flex; align-items:baseline; gap:16px; }}
header h1 {{ font-size:18px; margin:0; }} header span {{ color:#cbd5e1; font-size:13px; }}
nav {{ background:#fff; border-bottom:1px solid var(--line); padding:0 16px; display:flex; gap:4px; position:sticky; top:0; z-index:5; }}
nav button {{ border:0; background:none; padding:12px 14px; font-size:14px; cursor:pointer; color:var(--muted); border-bottom:3px solid transparent; font-family:inherit; }}
nav button.on {{ color:var(--ink); border-bottom-color:var(--accent); font-weight:bold; }}
main {{ max-width:1180px; margin:0 auto; padding:16px; }}
.tab {{ display:none; }} .tab.on {{ display:block; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:12px; }}
.stat {{ background:var(--card); border-radius:10px; padding:14px 16px; box-shadow:0 1px 2px rgba(0,0,0,.06); }}
.stat .v {{ font-size:26px; font-weight:bold; }} .stat .l {{ color:var(--muted); font-size:13px; }} .stat .s {{ color:var(--muted); font-size:12px; margin-top:4px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:12px; }}
.card {{ background:var(--card); border-radius:10px; padding:14px 16px; box-shadow:0 1px 2px rgba(0,0,0,.06); }}
.card.wide {{ grid-column:1/-1; }}
.card h3 {{ font-size:14px; margin:0 0 10px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ padding:6px 8px; border-bottom:1px solid var(--line); text-align:right; }}
th:first-child, td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-weight:normal; }}
.bars {{ display:flex; flex-direction:column; gap:6px; }}
.bar {{ display:grid; grid-template-columns:200px 1fr 70px; align-items:center; gap:8px; font-size:13px; }}
.bt {{ background:#eef0f3; height:14px; border-radius:7px; overflow:hidden; position:relative; }}
.bf {{ display:block; height:100%; border-radius:7px; }}
.bt.center .bf {{ position:absolute; top:0; }} .bt.center .bf.pos {{ left:50%; background:#2e9e5b; }} .bt.center .bf.neg {{ right:50%; background:#d9534f; }}
.bt.center::after {{ content:""; position:absolute; left:50%; top:0; bottom:0; width:1px; background:#9ca3af; }}
.bv {{ text-align:right; font-variant-numeric:tabular-nums; }}
.chart {{ width:100%; height:auto; }} .chart .grid {{ stroke:#e5e7eb; }} .chart .axis {{ font-size:11px; fill:#6b7280; }}
.legend {{ font-size:12px; color:var(--muted); margin-bottom:4px; }} .lg {{ margin-right:12px; }}
.lg i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:middle; }}
.muted {{ color:var(--muted); font-size:12px; }} .big {{ font-size:28px; font-weight:bold; margin:4px 0; }}
code {{ background:#eef0f3; padding:1px 5px; border-radius:4px; }}
#player {{ position:sticky; top:46px; background:#111; text-align:center; padding:8px; border-radius:10px; z-index:4; }}
#player video {{ max-width:100%; max-height:48vh; }} #now {{ color:#ddd; font-size:13px; margin-top:4px; }}
section {{ background:#fff; border-radius:10px; padding:10px 14px; margin:10px 0; }}
section h2 {{ font-size:14px; margin:0 0 6px; }}
.res-勝ち {{ color:#1a7f37; }} .res-負け {{ color:#c0392b; }}
.scene {{ display:flex; align-items:center; gap:10px; padding:4px 0; border-top:1px solid #eee; font-size:13px; }}
.scene button {{ padding:2px 10px; cursor:pointer; }}
.kind {{ font-size:12px; background:#eef0f3; border-radius:4px; padding:1px 6px; white-space:nowrap; }}
.time {{ font-variant-numeric:tabular-nums; color:#666; width:44px; }}
.filters {{ margin:10px 0; font-size:13px; }} .filters label {{ margin-right:12px; }}
.maps {{ font-size:12px; color:var(--muted); padding-top:4px; }}
ul.fb {{ list-style:none; padding:0; margin:0; font-size:13px; }} ul.fb li {{ padding:4px 0; border-top:1px solid #eee; }}
.tag {{ color:#fff; font-size:11px; border-radius:4px; padding:1px 6px; margin-right:8px; }}
.fix {{ display:flex; align-items:center; gap:8px; margin:2px 0 6px; font-size:12px; }} .fix img {{ border-radius:4px; }}
.fix button {{ font-size:12px; padding:2px 8px; cursor:pointer; }}
</style>{head}</head><body>
<header><h1>splat-compass</h1><span>{subtitle}</span></header>
<nav>{tabs}</nav>
<main>{bodies}</main>
<script>
document.querySelectorAll("nav button").forEach(b => b.addEventListener("click", () => {{
  document.querySelectorAll("nav button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("on", t.id === b.dataset.tab));
  try {{ localStorage.setItem("sc-tab", b.dataset.tab); }} catch (e) {{}}
}}));
try {{ const t = localStorage.getItem("sc-tab"); const b = document.querySelector('nav button[data-tab="' + t + '"]'); if (b) b.click(); }} catch (e) {{}}
const v = document.getElementById("v");
let stopAt = null;
function play(src, a, b, label) {{
  if (!v) return;
  if (v.getAttribute("src") !== src) v.setAttribute("src", src);
  const go = () => {{ v.currentTime = a; v.play(); }};
  if (v.readyState >= 1) go(); else v.addEventListener("loadedmetadata", go, {{ once: true }});
  stopAt = b; document.getElementById("now").textContent = label;
}}
if (v) v.addEventListener("timeupdate", () => {{ if (stopAt !== null && v.currentTime >= stopAt) {{ v.pause(); stopAt = null; }} }});
document.querySelectorAll(".filters input").forEach(cb => cb.addEventListener("change", () => {{
  document.querySelectorAll(".scene[data-kind='" + cb.value + "']").forEach(el => el.style.display = cb.checked ? "" : "none");
}}));
</script>{script}</body></html>"""


def render(store, video_dir=None, link=scenes.file_link, first_tabs=(), head="", script=""):
    """まとめのページの HTML。first_tabs: 先頭に足すタブ [(キー, 名前, HTML)]（アプリの「試合中」など）。"""
    ms = [m for m in store.matches() if m.get("video_end") and not m.get("interrupted")]
    ms.sort(key=lambda m: m.get("played_at", ""))
    tabs = [(k, t, (lambda b=b: b)) for k, t, b in first_tabs]
    tabs += [("overview", "概要", lambda: section_overview(ms)),
            ("feedback", "試合ごとの振り返り", lambda: section_feedback(store, link, bool(first_tabs))),
            ("rules", "ルール別・打開", lambda: section_rules(ms, store)),
            ("stages", "ステージ別", lambda: section_stages(ms)),
            ("killers", "倒されたブキ", lambda: section_killers(store, ms)),
            ("weapons", "ブキ別の勝率", lambda: section_weapons(store, ms)),
            ("model", "勝率モデル", lambda: section_model(store, ms))]
    if video_dir and os.path.isdir(video_dir):
        def sc():
            secs, kinds, total = scenes.scene_sections(store, video_dir, last=60, link=link)
            return ('<div id="player"><video id="v" controls preload="none"></video>'
                    '<div id="now">場面の「再生」を押すと、ここで再生します</div></div>'
                    f'<div class="filters">{scenes.filters_html(kinds)}</div>'
                    f'<p class="muted">新しい60試合から {total}場面</p>' + "".join(secs))
        tabs.append(("scenes", "見返す場面", sc))
    nav = "".join(f'<button data-tab="{k}" class="{"on" if i == 0 else ""}">{ESC(t)}</button>' for i, (k, t, _) in enumerate(tabs))
    bodies = ""
    for i, (k, t, fn) in enumerate(tabs):
        try:
            body = fn()
        except Exception as e:  # 1つの集計が失敗しても、ほかは見られるようにする
            body = f'<p class="muted">この集計を作れませんでした: {ESC(str(e))}</p>'
        bodies += f'<div class="tab{" on" if i == 0 else ""}" id="{k}">{body}</div>'
    sub = f"{len(ms)}試合 ・ {dt.datetime.now():%Y-%m-%d %H:%M} 時点"
    return PAGE.format(subtitle=ESC(sub), tabs=nav, bodies=bodies, head=head, script=script)


def build(store, video_dir=None):
    """まとめのページを report.html に書き出す（ブラウザで直接開く用）。"""
    path = os.path.join(store.root, "report.html")
    page = render(store, video_dir)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="?", help="録画のフォルダ（次からは省略できます）")
    ap.add_argument("--data", help="記録の保存先")
    ap.add_argument("--no-open", action="store_true", help="ブラウザで開かない")
    a = ap.parse_args()
    store = Store(a.data)
    cfg = store.load_config()
    if a.videos:
        cfg["video_dir"] = a.videos
        store.save_config(cfg)
    path = build(store, cfg.get("video_dir"))
    print(f"まとめのページを作りました: {path}")
    if not a.no_open:
        webbrowser.open(pathlib.Path(path).resolve().as_uri())


if __name__ == "__main__":
    main()
