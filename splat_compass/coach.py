"""試合後の振り返りを深くする。自分が倒された1回ごとに、そのときの状況を調べて分類する。

- 倒された1回ずつ: 人数差、スペシャルを溜めていたか、復活してから何秒か、カウントのリード、エリアの確保、
  打開中か、直前に倒していたか、勝率がどれだけ動いたか
- 何試合分かまとめた倒され方の傾向（最近の試合と、それより前の自分を比べる）

  python -m splat_compass.coach            … 一番新しい試合の倒された場面
  python -m splat_compass.coach --trend    … 最近20試合の倒され方の傾向
"""
import argparse
from collections import Counter

from . import area, clock
from .storage import Store

QUICK = 8.0       # 復活してこの秒数以内に倒されたら「復活してすぐ」
TRADE = 3.0       # 倒されるこの秒数前までに自分が倒していたら「相打ち」
LEAD = 20         # カウントでこれ以上リードしていたら「リード中」
BEFORE, AFTER = 1.0, 4.0  # 勝率の動きを見る、倒される前と後の秒数
BIG_DROP = 0.08   # 勝率がこれ以上下がったら「大きく下がった」

# 倒され方の分類。上ほど優先して、その1回の「主な理由」にする
TAGS = [
    ("held", "スペシャルを溜めたまま"),
    ("outnumbered", "人数が不利なところで"),
    ("last", "味方がいなくなって最後の1人で"),
    ("quick", f"復活して{int(QUICK)}秒以内に"),
    ("dakai_alone", "打開で味方がそろう前に"),
    ("lead", "リードしている間に"),
    ("trade", "倒してから（相打ち）"),
]
LABEL = dict(TAGS)
BAD = {"held", "outnumbered", "last", "quick", "dakai_alone", "lead"}  # 振り返りで「気になる」として扱うもの


def _mmss(sec):
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _row_at(tl, t):
    row = None
    for r in tl:
        if r[0] > t:
            break
        row = r
    return row


_CACHE = {}


def deaths(m, wp=None):
    """deaths の中身。勝率を使わないときは、同じ試合を何度も調べないよう覚えておく（振り返りで全試合を何度も見るため）。"""
    if wp is not None:
        return _deaths(m, wp)
    key = (m.get("_file"), len(m.get("events") or []), m.get("video_end"), m.get("me_slot"), m.get("rule"))
    if key[0] is None:
        return _deaths(m, None)
    if key not in _CACHE:
        _CACHE[key] = _deaths(m, None)
    return _CACHE[key]


def _deaths(m, wp=None):
    """自分が倒された1回ずつ: [{rel, tags, main, ally, enemy, drop, held, since_respawn}]。自分の枠が分からなければ None。
    wp: 時刻 → 勝率 の関数（無ければ勝率の動きは出さない）。"""
    me = m.get("me_slot")
    if not me:
        return None
    tl = [] if m.get("rule") == clock.TURF_RULE else m.get("timeline") or []
    kills = sorted(k["t"] for k in m.get("kills") or [])
    eps = area.episodes(m) if m.get("rule") == "ガチエリア" and tl else []
    out, last_respawn = [], None
    for t, rel, slot, what, *rest in sorted(m["events"], key=lambda e: e[0]):
        if slot != me:
            continue
        if what == "復活":
            last_respawn = t
            continue
        if "倒された" not in what:
            continue
        ally_after = rest[0] if len(rest) >= 2 else None
        enemy = rest[1] if len(rest) >= 2 else None
        ally = ally_after + 1 if ally_after is not None else None  # 倒される直前（自分を含む）
        tags = []
        if "持ったまま" in what:
            tags.append("held")
        if ally is not None and enemy is not None and ally < enemy:
            tags.append("outnumbered")
        if ally_after == 0:
            tags.append("last")
        since = t - last_respawn if last_respawn is not None else None
        if since is not None and since <= QUICK:
            tags.append("quick")
        ep = next((e for e in eps if e["start"] <= t <= e["end"]), None)
        if ep is not None and t - ep["start"] <= area.EARLY and ally is not None and ally <= 2:
            tags.append("dakai_alone")
        row = _row_at(tl, t)
        if row is not None and row[2] - row[1] >= LEAD:
            tags.append("lead")
        if any(t - TRADE <= k <= t for k in kills):
            tags.append("trade")
        drop = None
        if wp is not None:
            try:
                drop = float(wp(t + AFTER) - wp(t - BEFORE))
            except Exception:
                drop = None
        main = next((k for k, _ in TAGS if k in tags), None)
        out.append({"t": t, "rel": rel, "tags": tags, "main": main, "ally": ally, "enemy": enemy,
                    "drop": drop, "since_respawn": since})
    return out


_MODEL = {}


def _area_model(store):
    """勝率モデルを、ファイルが変わったときだけ読み直す。"""
    import os
    from . import winprob
    p = os.path.join(store.root, "winprob_area.json")
    mt = os.path.getmtime(p) if os.path.exists(p) else None
    if _MODEL.get("mt") != mt or "m" not in _MODEL:
        _MODEL["m"], _MODEL["mt"] = winprob.load_model(store), mt
    return _MODEL["m"]


def wp_func(store, m):
    """試合の毎秒の勝率から、時刻 → 勝率 の関数を作る。出せなければ None。"""
    try:
        import numpy as np
        from . import learning
        pr = learning.predict_match(store, m, _area_model(store) if m.get("rule") == "ガチエリア" else None)
    except Exception:
        return None
    if pr is None:
        return None
    tt, pp = pr

    def f(t):
        return pp[min(max(int(np.searchsorted(tt, t)), 0), len(pp) - 1)]
    return f


def describe(d):
    """1回分を1行の文にする。"""
    parts = [LABEL[k] for k, _ in TAGS if k in d["tags"]]
    s = "、".join(parts) if parts else "目立った状況なし"
    if d["ally"] is not None and d["enemy"] is not None:
        s += f"（{d['ally']}対{d['enemy']}）"
    if d["drop"] is not None and abs(d["drop"]) >= 0.02:
        s += f" 推定勝率 {d['drop'] * 100:+.0f}pt"
    return s


def shares(ds_list):
    """倒された回の並びから、分類ごとの割合と回数。{分類: (回数, 割合)} と全体の回数。"""
    n = sum(len(ds) for ds in ds_list)
    c = Counter(k for ds in ds_list for d in ds for k in d["tags"])
    return {k: (c[k], c[k] / n if n else 0.0) for k, _ in TAGS}, n


def match_lines(m, history_shares=None, wp=None):
    """1試合の振り返りに足す行。[(良い/気になる/情報, 文)]
    history_shares: それより前の自分の分類ごとの割合（shares の戻り値の1つ目）。"""
    ds = deaths(m, wp)
    if not ds:
        return []
    out = []
    n = len(ds)
    c = Counter(k for d in ds for k in d["tags"])
    for k, label in TAGS:
        if k not in BAD or k == "held" or c[k] < 2:  # 溜めたままは feedback の方で比べている
            continue
        share = c[k] / n
        usual = history_shares.get(k, (0, 0.0))[1] if history_shares else None
        # 普段より明らかに多いときだけ出す（普段の割合が分からなければ半分以上のとき）
        if (usual is not None and share >= usual * 1.6 + 0.1) or (usual is None and share >= 0.5):
            times = "、".join(clock.fmt(m, d["rel"]) for d in ds if k in d["tags"])
            tail = f"（普段は {usual * 100:.0f}%）" if usual is not None else ""
            out.append(("気になる", f"倒された{n}回のうち{c[k]}回が「{label}」{tail} {times}"))
    big = sorted([d for d in ds if d["drop"] is not None and d["drop"] <= -BIG_DROP], key=lambda d: d["drop"])
    if big:
        d = big[0]
        out.append(("気になる", f"一番響いたデス {clock.fmt(m, d['rel'])}: {describe(d)}"))
    if c["trade"] >= 2:
        out.append(("良い", f"倒してから倒された（相打ち）が{c['trade']}回。ただ倒されるよりは仕事ができています"))
    return out


def trend(store, recent=20):
    """最近 recent 試合の倒され方を、それより前の試合と比べる。[(分類名, 最近の回数, 最近の割合, 前の割合)]"""
    ms = sorted([m for m in store.matches() if m.get("video_end") and not m.get("interrupted") and m.get("me_slot")],
                key=lambda m: m.get("played_at", ""))
    if len(ms) < recent + 10:
        recent = max(len(ms) // 2, 1)
    new = [deaths(m) or [] for m in ms[-recent:]]
    old = [deaths(m) or [] for m in ms[:-recent]]
    sn, nn = shares(new)
    so, no = shares(old)
    rows = [(label, sn[k][0], sn[k][1], so[k][1] if no else None) for k, label in TAGS]
    return rows, nn, no, recent, len(ms) - recent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--trend", action="store_true")
    a = ap.parse_args()
    store = Store(a.data) if a.data else Store()
    if a.trend:
        rows, nn, no, r, o = trend(store)
        print(f"最近{r}試合（倒された{nn}回）と、それより前の{o}試合（{no}回）")
        for label, n, s, p in rows:
            print(f"  {label:20s} {n:3d}回 {s * 100:4.0f}%  前は {p * 100:4.0f}%" if p is not None else f"  {label} {n}回 {s * 100:.0f}%")
        return
    ms = sorted([m for m in store.matches() if m.get("video_end")], key=lambda m: m.get("played_at", ""))
    m = ms[-1]
    for d in deaths(m, wp_func(store, m)) or []:
        print(clock.fmt(m, d["rel"]), describe(d))


if __name__ == "__main__":
    main()
