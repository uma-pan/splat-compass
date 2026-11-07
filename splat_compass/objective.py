"""ガチヤグラ・ガチホコ・ガチアサリの打開の集計。

ヤグラ・ホコ（位置は -1＝味方のゴール 〜 +1＝敵のゴール）
  始まり: 相手に押されて、位置が中央を越えて味方側に入った
  止めた: 一番深く押された地点から、一定の距離（STOP_BACK）戻った
  完了:   位置が中央まで戻った
  ホコのリセット: 味方側の深いところから一気に中央付近へ戻ったものは印を付ける

アサリ
  始まり: 味方のゴールが割られた（ゴールが開き、その間に相手のカウントが減った）
  完了:   ゴールが閉じた

  python -m splat_compass.objective
"""
import argparse
from statistics import median

from .storage import Store

CENTER_MARGIN = 0.05  # 中央の「あそび」（棒の長さの片側に対する割合）
STOP_BACK = 0.10      # 一番深い地点からこれだけ戻ったら「止めた」とみなす
RESET_JUMP = 0.30     # 2秒以内にこれ以上中央へ跳んだら、ホコのリセットとみなす


def _smooth_positions(obj):
    """[(時刻, 位置)]。読めなかった秒は捨て、前後5秒の中央値でならす。"""
    pts = [(t, p) for t, p, _ in obj if p is not None]
    out = []
    for i, (t, _) in enumerate(pts):
        win = [p for _, p in pts[max(0, i - 2):i + 3]]
        out.append((t, median(win)))
    return out


def push_episodes(m):
    pts = _smooth_positions(m.get("objective") or [])
    eps, ep = [], None
    for i, (t, p) in enumerate(pts):
        if ep is None:
            if p < -CENTER_MARGIN:
                ep = {"start": t, "deepest": p, "stopped": None, "reset": False}
            continue
        if p < ep["deepest"]:
            ep["deepest"] = p
            ep["stopped"] = None
        elif ep["stopped"] is None and p - ep["deepest"] >= STOP_BACK:
            ep["stopped"] = t
        if p >= -CENTER_MARGIN:  # 中央（のあそびの中）まで戻った
            prev_t, prev_p = pts[i - 1]
            ep["reset"] = (t - prev_t) <= 2 and (p - prev_p) >= RESET_JUMP
            ep.update(end=t, success=True)
            eps.append(ep)
            ep = None
    if ep is not None and pts:
        ep.update(end=pts[-1][0], success=False)
        eps.append(ep)
    for e in eps:
        e["seconds"] = e["end"] - e["start"]
        e["to_stop"] = (e["stopped"] - e["start"]) if e["stopped"] else None
    return eps


def checkpoint_times(m):
    """各チェックポイントが最初に通られた時刻（試合開始からの秒）と、どちらが通ったか。
    中央より左（味方側）のチェックポイントは相手が、右（敵側）は味方が通る。中央に近い順に並べる。"""
    cps = m.get("checkpoints") or []
    pts = _smooth_positions(m.get("objective") or [])
    out = []
    for c in sorted(cps, key=abs):
        who = "味方" if c > 0 else "敵"
        hit = next((t for t, p in pts if (p >= c if c > 0 else p <= c)), None)
        out.append({"pos": c, "by": who, "t": None if hit is None else hit - m["video_start"],
                    "order": sum(1 for x in cps if (x > 0) == (c > 0) and abs(x) < abs(c)) + 1})
    return out


def asari_episodes(m):
    """ゴールが開いていた区間のうち、相手のカウントが減ったもの（味方のゴールが割られた）。"""
    obj = m.get("objective") or []
    tl = {round(t): (l, r) for t, l, r, *_ in (m.get("timeline") or [])}
    eps, start = [], None
    for t, _, open_ in obj + [[obj[-1][0] + 1, None, 0]] if obj else []:
        if open_ and start is None:
            start = t
        elif not open_ and start is not None:
            a = tl.get(round(start)) or tl.get(round(start) + 1)
            b = tl.get(round(t)) or tl.get(round(t) - 1)
            if a and b and t - start >= 3:
                lost_l, lost_r = a[0] - b[0], a[1] - b[1]
                if lost_r > lost_l:  # 相手のカウントが減った＝味方のゴールが割られていた
                    eps.append({"start": start, "end": t, "seconds": t - start, "count_lost": lost_r})
            start = None
    return eps


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _report_checkpoints(ms):
    """一番中央に近いチェックポイント（第1チェックポイント）を、どちらが先に、何秒で取ったか。"""
    rows = []
    for m in ms:
        first = [x for x in checkpoint_times(m) if x["order"] == 1 and x["t"] is not None]
        if not first:
            continue
        f = min(first, key=lambda x: x["t"])
        rows.append((m, f))
    if not rows:
        print("  チェックポイントの記録がまだありません")
        return
    for who in ("味方", "敵"):
        g = [(m, f) for m, f in rows if f["by"] == who]
        known = [m for m, _ in g if m.get("result") in ("勝ち", "負け")]
        wins = sum(m["result"] == "勝ち" for m in known)
        if g:
            print(f"  第1チェックポイントを先に取ったのが{who}: {len(g)}試合  平均{_avg([f['t'] for _, f in g]):.0f}秒で取得"
                  + (f"  勝率{wins / len(known) * 100:.0f}%" if known else ""))


def report(matches):
    for rule in ("ガチヤグラ", "ガチホコ"):
        ms = [m for m in matches if m.get("rule") == rule and m.get("objective")]
        if not ms:
            continue
        eps = [(m, e) for m in ms for e in push_episodes(m) if e["seconds"] >= 3]
        ok = [e for _, e in eps if e["success"]]
        print(f"■ {rule} {len(ms)}試合 / 押し込まれた回数 {len(eps)}回")
        print(f"  中央まで戻せた割合 {len(ok) / max(len(eps), 1) * 100:.0f}%")
        print(f"  止めるまで平均 {_avg([e['to_stop'] for _, e in eps]):.1f}秒 / 中央に戻すまで平均 {_avg([e['seconds'] for e in ok]):.1f}秒")
        print(f"  押し込まれた深さの平均 {_avg([-e['deepest'] for _, e in eps]) * 100:.0f}%（ゴールまでを100%として）")
        if rule == "ガチホコ":
            print(f"  そのうちリセットで中央に戻ったもの {sum(e['reset'] for e in ok)}回")
        _report_checkpoints(ms)
        for res in ("勝ち", "負け"):
            g = [e for m, e in eps if m.get("result") == res]
            n = sum(1 for m in ms if m.get("result") == res)
            if n:
                print(f"  {res}: 1試合あたり押し込まれた回数 {len(g) / n:.1f}回 / 中央に戻すまで平均 {_avg([e['seconds'] for e in g if e['success']]):.1f}秒")
        print()
    ms = [m for m in matches if m.get("rule") == "ガチアサリ" and m.get("objective")]
    if ms:
        eps = [(m, e) for m in ms for e in asari_episodes(m)]
        print(f"■ ガチアサリ {len(ms)}試合 / 味方のゴールを割られた回数 {len(eps)}回")
        print(f"  ゴールが閉じるまで平均 {_avg([e['seconds'] for _, e in eps]):.1f}秒 / その間に取られたカウント平均 {_avg([e['count_lost'] for _, e in eps]):.1f}")
        for res in ("勝ち", "負け"):
            g = [e for m, e in eps if m.get("result") == res]
            n = sum(1 for m in ms if m.get("result") == res)
            if n:
                print(f"  {res}: 1試合あたり割られた回数 {len(g) / n:.1f}回 / 取られたカウント合計 {sum(e['count_lost'] for e in g) / n:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="記録の保存先")
    a = ap.parse_args()
    ms = [m for m in Store(a.data).matches() if not m.get("interrupted")]
    if not any(m.get("objective") for m in ms):
        print("ヤグラ・ホコの位置やアサリのゴールの記録がまだありません（録画の読み直しが必要です）")
        return
    report(ms)


if __name__ == "__main__":
    main()
