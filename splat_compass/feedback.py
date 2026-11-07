"""試合が終わったあとに、その試合を普段の自分と比べて目立ったところを短く挙げる。

- 普段より明らかに多かった・少なかったものだけを出す（毎回同じ指摘が並ばないように）
- 打開のスペシャルの使い方のように、集計で裏付けが取れたものを使う
- 時刻（試合開始からの秒）を付けて、見返せるようにする
- 良かったところも出す

  python -m splat_compass.feedback        … 一番新しい試合
  python -m splat_compass.feedback --last 5
"""
import argparse
import os
from collections import Counter

from . import area, killcam
from .storage import Store

QUICK_DEATH = 8.0


def _mmss(sec):
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _numbers(m):
    """1試合の数字（自分の枠が分かる試合だけ）。"""
    me = m.get("me_slot")
    if not me:
        return None
    mins = max((m["video_end"] - m["video_start"]) / 60, 0.5)
    deaths, held, quick, specials = [], [], [], []
    last_respawn = None
    for t, rel, slot, what, *_ in m["events"]:
        if slot != me:
            continue
        if what == "復活":
            last_respawn = t
        elif what == "スペシャル発動":
            specials.append(rel)
        elif "倒された" in what:
            deaths.append(rel)
            if "持ったまま" in what:
                held.append(rel)
            elif last_respawn is not None and t - last_respawn <= QUICK_DEATH:
                quick.append(rel)
    kills = [k["rel"] for k in m["kills"]] if "kills" in m else None  # 倒した数を読む前の記録には無い
    n = {"mins": mins, "deaths": deaths, "held": held, "quick": quick, "specials": specials,
         "early_sp_few": [], "dakai": [], "kills": kills}
    if m.get("rule") == "ガチエリア" and m.get("timeline"):
        for e in area.episodes(m):
            if e["seconds"] < 3:
                continue
            n["dakai"].append(e["seconds"] if e["success"] else None)
            if e["seconds"] > area.EARLY:
                f = area.early(e, m)
                if f["alive_at_first_special"] is not None and f["alive_at_first_special"] <= 2:
                    n["early_sp_few"].append(e["start"] - m["video_start"] + f["first_special_at"])
    return n


def baseline(history):
    """普段の自分: 1分あたりの各回数と、エリアの打開にかかる平均秒数。"""
    ns = [x for x in (_numbers(m) for m in history) if x]
    if len(ns) < 10:
        return None
    mins = sum(x["mins"] for x in ns)
    rate = {k: sum(len(x[k]) for x in ns) / mins for k in ("deaths", "held", "quick", "specials", "early_sp_few")}
    dk = [s for x in ns for s in x["dakai"] if s is not None]
    rate["dakai_sec"] = sum(dk) / len(dk) if dk else None
    rate["matches"] = len(ns)
    kn = [x for x in ns if x["kills"] is not None]
    rate["kills"] = sum(len(x["kills"]) for x in kn) / sum(x["mins"] for x in kn) if len(kn) >= 5 else None
    return rate


def match_feedback(m, base, store=None):
    """[(良い/気になる, 文)]"""
    n = _numbers(m)
    if n is None:
        return [("情報", "この試合は自分の枠が分からなかったので、振り返りを出せません")]
    if base is None:
        return [("情報", "普段の自分と比べるには、自分の枠が分かった試合があと少し必要です（10試合から）")]
    out = []
    mins = n["mins"]
    times = lambda xs: "（" + "、".join(_mmss(x) for x in xs[:5]) + "）" if xs else ""

    def compare(key, label, unit="回", bad_if_more=True, min_count=2):
        got, exp = len(n[key]), base[key] * mins
        if bad_if_more and got >= min_count and got >= exp * 1.8 + 0.5:
            out.append(("気になる", f"{label} {got}{unit}（普段なら {exp:.1f}{unit}くらい）{times(n[key])}"))
        elif not bad_if_more and exp >= 1.5 and got <= exp * 0.5:
            out.append(("気になる", f"{label} {got}{unit}（普段なら {exp:.1f}{unit}くらい）"))

    compare("held", "スペシャルを持ったまま倒された")
    compare("quick", f"復活して{int(QUICK_DEATH)}秒以内に倒された")
    compare("early_sp_few", "打開で、味方2人以下のときに最初のスペシャルを吐いた", min_count=2)
    compare("specials", "スペシャルを使った回数が少なめ", bad_if_more=False)
    d_got, d_exp = len(n["deaths"]), base["deaths"] * mins
    if d_got >= d_exp * 1.4 + 1:
        out.append(("気になる", f"デスが多め {d_got}回（普段なら {d_exp:.1f}回くらい）"))
    elif d_exp >= 3 and d_got <= d_exp * 0.6:
        out.append(("良い", f"デスが少なかった {d_got}回（普段なら {d_exp:.1f}回くらい）"))
    s_got, s_exp = len(n["specials"]), base["specials"] * mins
    if s_exp >= 2 and s_got >= s_exp * 1.4 + 1:
        out.append(("良い", f"スペシャルをよく使えた {s_got}回（普段なら {s_exp:.1f}回くらい）"))
    if n["kills"] is not None and base.get("kills"):
        k_got, k_exp = len(n["kills"]), base["kills"] * mins
        if k_got >= k_exp * 1.4 + 1:
            out.append(("良い", f"よく倒せた {k_got}人（普段なら {k_exp:.1f}人くらい）"))
        elif k_exp >= 3 and k_got <= k_exp * 0.5:
            out.append(("気になる", f"倒した数が少なめ {k_got}人（普段なら {k_exp:.1f}人くらい）"))
    dk = [s for s in n["dakai"] if s is not None]
    if base.get("dakai_sec") and len(dk) >= 2:
        avg = sum(dk) / len(dk)
        if avg <= base["dakai_sec"] * 0.7:
            out.append(("良い", f"エリアの打開が速かった 平均{avg:.0f}秒（普段は {base['dakai_sec']:.0f}秒）"))
        elif avg >= base["dakai_sec"] * 1.4:
            out.append(("気になる", f"エリアの打開に時間がかかった 平均{avg:.0f}秒（普段は {base['dakai_sec']:.0f}秒）"))
    if store is not None and m.get("killed_by"):
        labels, protos = killcam.load_labels(store)
        names = [killcam.name_of(store, os.path.join(store.root, "killcam", k["image"]), labels, protos)
                 for k in m["killed_by"]]
        c = Counter(x for x in names if x)
        if c:
            name, k = c.most_common(1)[0]
            if k >= 3:
                out.append(("情報", f"{name}に{k}回倒された"))
    if not out:
        out.append(("情報", "普段と比べて大きく違うところはありませんでした"))
    if store is not None:
        try:
            from . import learning
            c = learning.contribution(store, m)
        except Exception:
            c = None
        if c and any(v[0] for v in c.values()):
            parts = [f"{k} {v[0]}回 {v[1] * 100:+.0f}%" for k, v in c.items() if v[0]]
            total = sum(v[1] for v in c.values())
            out.append(("情報", f"自分の行動の前後で動いた勝率（合計 {total * 100:+.0f}%）: " + "、".join(parts)))
    order = {"気になる": 0, "良い": 1, "情報": 2}
    return sorted(out, key=lambda x: order[x[0]])


def for_latest(store, last=1):
    """新しい試合から順に [(試合, 振り返り)]。"""
    ms = sorted([m for m in store.matches() if m.get("video_end") and not m.get("interrupted")],
                key=lambda m: m.get("played_at", ""))
    out = []
    for i in range(len(ms) - 1, max(len(ms) - 1 - last, -1), -1):
        base = baseline(ms[:i])
        out.append((ms[i], match_feedback(ms[i], base, store)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="記録の保存先")
    ap.add_argument("--last", type=int, default=1, help="新しい順に何試合分出すか")
    a = ap.parse_args()
    for m, items in for_latest(Store(a.data), a.last):
        print(f"■ {m.get('played_at', '')[:16].replace('T', ' ')} {m.get('rule') or ''} {m.get('result') or ''}")
        for kind, text in items:
            print(f"  [{kind}] {text}")


if __name__ == "__main__":
    main()
