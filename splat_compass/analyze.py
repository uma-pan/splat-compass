"""貯まった試合の記録から、自分（左1）について「どんな状況で何が起きやすいか」を集計する。

  python -m splat_compass.analyze
"""
import argparse
from collections import defaultdict

from .storage import Store

SLOTS = ["左1", "左2", "左3", "左4", "右1", "右2", "右3", "右4"]
AFTER = 10  # スペシャル発動後に見る秒数


def _replay(m):
    """出来事を順にたどり、(時刻, 出来事, その直前の状態) を返す。"""
    st = {s: 1 for s in SLOTS}  # 0=倒された 1=生きている 2=スペシャルあり
    for t, rel, slot, what, *_ in m["events"]:
        yield t, slot, what, dict(st)
        if "倒された" in what:
            st[slot] = 0
        elif what in ("復活", "スペシャル発動"):
            st[slot] = 1
        elif what == "スペシャル溜まった":
            st[slot] = 2


def _situation(st):
    ally = sum(st[s] > 0 for s in SLOTS[:4])
    enemy = sum(st[s] > 0 for s in SLOTS[4:])
    diff = "有利" if ally > enemy else ("不利" if ally < enemy else "同数")
    return st["左1"] > 0, diff, min(sum(st[s] == 2 for s in SLOTS[4:]), 2)


def situations(matches):
    alive_t, deaths = defaultdict(float), defaultdict(int)
    sp_use = defaultdict(lambda: [0, 0, 0])
    held = total = 0
    for m in matches:
        last = m["video_start"]
        evs = list(_replay(m))
        for i, (t, slot, what, st) in enumerate(evs):
            me_alive, diff, esp = _situation(st)
            if me_alive:
                alive_t[("人数", diff)] += t - last
                alive_t[("敵", esp)] += t - last
            last = t
            if slot == "左1" and "倒された" in what:
                total += 1
                held += "持ったまま" in what
                deaths[("人数", diff)] += 1
                deaths[("敵", esp)] += 1
            if slot == "左1" and what == "スペシャル発動":
                rec = sp_use[diff]
                rec[0] += 1
                for t2, s2, w2, _ in evs[i + 1:]:
                    if t2 - t > AFTER:
                        break
                    if s2.startswith("右") and "倒された" in w2:
                        rec[1] += 1
                    if s2 == "左1" and "倒された" in w2:
                        rec[2] += 1
                        break
    return alive_t, deaths, sp_use, held, total


def features(m):
    f = defaultdict(float)
    last = m["video_start"]
    for t, slot, what, st in _replay(m):
        ally = sum(st[s] > 0 for s in SLOTS[:4])
        enemy = sum(st[s] > 0 for s in SLOTS[4:])
        f["不利"] += (t - last) * (ally < enemy)
        f["有利"] += (t - last) * (ally > enemy)
        last = t
        if "倒された" in what:
            f["味方のデス" if slot[0] == "左" else "敵のデス"] += 1
            if slot == "左1":
                f["自分のデス"] += 1
                f["自分がスペシャルを持ったまま倒された"] += "持ったまま" in what
        elif what == "スペシャル発動":
            f["自分のスペシャル発動" if slot == "左1" else ("味方のスペシャル発動" if slot[0] == "左" else "敵のスペシャル発動")] += 1
    mins = (m["video_end"] - m["video_start"]) / 60
    out = {k + "（1分あたり）": f[k] / mins for k in
           ["自分のデス", "自分のスペシャル発動", "自分がスペシャルを持ったまま倒された",
            "味方のデス", "敵のデス", "味方のスペシャル発動", "敵のスペシャル発動"]}
    out["人数不利の割合"] = f["不利"] / (mins * 60)
    out["人数有利の割合"] = f["有利"] / (mins * 60)
    return out


def report(matches):
    mins = sum(m["video_end"] - m["video_start"] for m in matches) / 60
    alive_t, deaths, sp_use, held, total = situations(matches)
    print(f"{len(matches)}試合 / {mins:.0f}分 / 自分のデス {total}回（1分あたり {total / max(mins, 1e-9):.2f}回）")
    print("\n■ 状況ごとの自分のデスのしやすさ（生きていた1分あたり）")
    keys = [(("人数", d), f"人数{d}") for d in ("有利", "同数", "不利")] + \
           [(("敵", k), f"敵のスペシャル準備 {k}{'人以上' if k == 2 else '人'}") for k in (0, 1, 2)]
    for key, label in keys:
        tm = alive_t[key] / 60
        if tm > 0:
            print(f"  {label:<18} 生存{tm:6.1f}分  デス{deaths[key]:4d}回  → {deaths[key] / tm:.2f}回")
    print(f"\n■ スペシャルを溜めたまま倒された: {held}回（デスの {held / max(total, 1) * 100:.0f}%）")
    print(f"\n■ 自分がスペシャルを発動したあと{AFTER}秒以内")
    for d in ("有利", "同数", "不利"):
        n, k, died = sp_use[d]
        if n:
            print(f"  人数{d}で発動 {n:4d}回  敵が倒れた数 平均{k / n:.2f}人  自分も倒された {died / n * 100:.0f}%")
    known = [m for m in matches if m.get("result") in ("勝ち", "負け")]
    if not known:
        return
    wins = sum(m["result"] == "勝ち" for m in known)
    print(f"\n■ 勝敗が分かった試合: {len(known)}試合（{wins}勝{len(known) - wins}敗）")
    by_rule = defaultdict(lambda: [0, 0])
    for m in known:
        by_rule[m.get("rule") or "ルール不明"][m["result"] != "勝ち"] += 1
    for r, (w, l) in sorted(by_rule.items(), key=lambda x: -sum(x[1])):
        print(f"  {r:<6} {w}勝{l}敗  勝率{w / (w + l) * 100:.0f}%")
    print("\n■ 勝った試合と負けた試合の違い（平均）")
    agg = {"勝ち": defaultdict(list), "負け": defaultdict(list)}
    for m in known:
        for k, v in features(m).items():
            agg[m["result"]][k].append(v)
    if not agg["勝ち"] or not agg["負け"]:
        return
    print(f"  {'':<30}{'勝ち':>6}{'負け':>8}")
    for k in agg["勝ち"]:
        a = sum(agg["勝ち"][k]) / len(agg["勝ち"][k])
        b = sum(agg["負け"][k]) / len(agg["負け"][k])
        fa, fb = (f"{a * 100:.0f}%", f"{b * 100:.0f}%") if "割合" in k else (f"{a:.2f}", f"{b:.2f}")
        print(f"  {k:<30}{fa:>6}{fb:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="記録の保存先")
    ap.add_argument("--rule", help="このルールだけ集計（例: ガチエリア）")
    a = ap.parse_args()
    ms = [m for m in Store(a.data).matches() if m.get("video_end")]
    if a.rule:
        ms = [m for m in ms if m.get("rule") == a.rule]
    if not ms:
        print("まだ試合の記録がありません")
        return
    report(ms)


if __name__ == "__main__":
    main()
