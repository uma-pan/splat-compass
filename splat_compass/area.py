"""ガチエリアの打開の集計。

エリアを相手に確保されてから（敵の箱が塗られてから）、味方が確保し返すまでを「打開」の1回として数える。
相手が確保している間は相手のカウントが減るので、その減った分を「取られたカウント」とする。

  python -m splat_compass.area
"""
import argparse
from collections import defaultdict

from .storage import Store

SLOTS = ["左1", "左2", "左3", "左4", "右1", "右2", "右3", "右4"]
STABLE = 2         # 確保の状態が変わったとみなす連続秒数
SPECIAL_SYNC = 5.0  # この秒数以内に味方が2人以上スペシャルを使ったら「合わせた」とみなす


def _states(timeline):
    """[(時刻, 状態, 味方カウント, 敵カウント)]。状態は "味方" / "敵" / "中立"。数秒のちらつきはならす。"""
    out, cur, pend, n = [], None, None, 0
    for t, l, r, lf, rf in timeline:
        s = "味方" if lf and not rf else ("敵" if rf and not lf else "中立")
        if cur is None:
            cur = s
        elif s != cur:
            if s == pend:
                n += 1
            else:
                pend, n = s, 1
            if n >= STABLE:
                cur, pend, n = s, None, 0
        else:
            pend, n = None, 0
        out.append((t, cur, l, r))
    return out


def episodes(m):
    """1試合の中の打開を順に返す。"""
    st = _states(m.get("timeline") or [])
    eps, ep = [], None
    for i, (t, s, l, r) in enumerate(st):
        prev = st[i - 1][1] if i else None
        if s == "敵" and prev != "敵" and ep is None:
            ep = {"start": t, "enemy_count_start": r, "stopped": None}
        elif ep is not None:
            if s == "中立" and ep["stopped"] is None:
                ep["stopped"] = t  # 相手の確保を崩した（カウントが止まった）
            elif s == "敵":
                ep["stopped"] = None
            if s == "味方":
                ep.update(end=t, success=True, enemy_count_end=r)
                eps.append(ep)
                ep = None
    if ep is not None and st:
        t, s, l, r = st[-1]
        ep.update(end=t, success=False, enemy_count_end=r)
        eps.append(ep)
    for ep in eps:
        ep["seconds"] = ep["end"] - ep["start"]
        ep["count_lost"] = max(0, (ep["enemy_count_start"] or 0) - (ep["enemy_count_end"] or 0))
        _describe(ep, m)
    return eps


def _describe(ep, m):
    """打開の間に何が起きたかを、出来事の記録から数える。"""
    me = m.get("me_slot")
    st = {s: 1 for s in SLOTS}
    ally_sp, first_death, me_deaths, ally_deaths, enemy_deaths = [], None, 0, 0, 0
    min_ally = 4
    for t, rel, slot, what, *_ in m["events"]:
        inside = ep["start"] <= t <= ep["end"]
        if inside:
            if "倒された" in what:
                if first_death is None:
                    first_death = "味方" if slot[0] == "左" else "敵"
                if slot[0] == "左":
                    ally_deaths += 1
                else:
                    enemy_deaths += 1
                if slot == me:
                    me_deaths += 1
            if what == "スペシャル発動" and slot[0] == "左":
                ally_sp.append(t)
        if "倒された" in what:
            st[slot] = 0
        elif what in ("復活", "スペシャル発動"):
            st[slot] = 1
        elif what == "スペシャル溜まった":
            st[slot] = 2
        if t <= ep["start"]:
            ep["ally_alive_at_start"] = sum(st[s] > 0 for s in SLOTS[:4])
            ep["ally_specials_at_start"] = sum(st[s] == 2 for s in SLOTS[:4])
        if inside:
            min_ally = min(min_ally, sum(st[s] > 0 for s in SLOTS[:4]))
    ep["ally_specials_used"] = len(ally_sp)
    ep["specials_synced"] = any(b - a <= SPECIAL_SYNC for a, b in zip(ally_sp, ally_sp[1:]))
    ep["first_death"] = first_death
    ep["me_deaths"] = me_deaths
    ep["ally_deaths"] = ally_deaths
    ep["enemy_deaths"] = enemy_deaths
    ep["wiped"] = min_ally == 0
    ep.setdefault("ally_alive_at_start", 4)
    ep.setdefault("ally_specials_at_start", 0)


EARLY = 10.0  # 確保されてから最初の何秒を見るか


def early(ep, m, w=EARLY):
    """確保されてから最初の w 秒に起きたことを数える。
    打開が長引くほど出来事が増えて見える偏りを避けるため、w 秒より長く続いた打開だけで使う。"""
    me = m.get("me_slot")
    st = {s: 1 for s in SLOTS}
    f = {"ally_deaths": 0, "enemy_deaths": 0, "me_died": False, "first_special_at": None,
         "alive_at_first_special": None, "specials": []}
    for t, rel, slot, what, *_ in m["events"]:
        inside = ep["start"] <= t <= ep["start"] + w
        if inside and "倒された" in what:
            f["ally_deaths" if slot[0] == "左" else "enemy_deaths"] += 1
            f["me_died"] |= slot == me
        if inside and what == "スペシャル発動" and slot[0] == "左":
            if f["first_special_at"] is None:
                f["first_special_at"] = t - ep["start"]
                f["alive_at_first_special"] = sum(st[s] > 0 for s in SLOTS[:4])
            f["specials"].append(t)
        if "倒された" in what:
            st[slot] = 0
        elif what in ("復活", "スペシャル発動"):
            st[slot] = 1
        elif what == "スペシャル溜まった":
            st[slot] = 2
    sp = f.pop("specials")
    f["synced"] = any(b - a <= SPECIAL_SYNC for a, b in zip(sp, sp[1:]))
    f["n_specials"] = len(sp)
    return f


def _row_after(label, items, w=EARLY):
    """items: [(打開, 最初の w 秒の様子)]。w 秒以降、打開できるまでの残り時間と、その後に取られたカウントを比べる。"""
    if not items:
        return f"  {label:<30}   0回"
    ok = [e for e, _ in items if e["success"]]
    rest = [e["seconds"] - w for e in ok]
    return (f"  {label:<30}{len(items):4d}回  成功{len(ok) / len(items) * 100:4.0f}%"
            f"  そこから打開まで平均{_avg(rest):5.1f}秒")


def _avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _row(label, eps):
    if not eps:
        return f"  {label:<26}   0回"
    ok = [e for e in eps if e["success"]]
    return (f"  {label:<26}{len(eps):4d}回  成功{len(ok) / max(len(eps), 1) * 100:4.0f}%"
            f"  打開までの平均{_avg([e['seconds'] for e in ok]):5.1f}秒  取られたカウント平均{_avg([e['count_lost'] for e in eps]):5.1f}")


def report(matches):
    ms = [m for m in matches if m.get("rule") == "ガチエリア" and m.get("timeline")]
    if not ms:
        print("カウントの記録があるガチエリアの試合がまだありません")
        return
    all_eps, per_match = [], []
    for m in ms:
        eps = [e for e in episodes(m) if e["seconds"] >= 3]
        all_eps += eps
        per_match.append((m, eps))
    print(f"ガチエリア {len(ms)}試合 / 打開 {len(all_eps)}回")
    print("\n■ 全体")
    print(_row("すべて", all_eps))
    print("\n■ 打開の始め方")
    print(_row("確保された時点で味方4人", [e for e in all_eps if e["ally_alive_at_start"] == 4]))
    print(_row("確保された時点で味方3人以下", [e for e in all_eps if e["ally_alive_at_start"] < 4]))
    print(_row("味方が一度全滅した", [e for e in all_eps if e["wiped"]]))
    print(_row("全滅しなかった", [e for e in all_eps if not e["wiped"]]))
    print("\n■ 打開中の撃ち合い")
    print(_row("最初に落ちたのが敵", [e for e in all_eps if e["first_death"] == "敵"]))
    print(_row("最初に落ちたのが味方", [e for e in all_eps if e["first_death"] == "味方"]))
    print("\n■ スペシャル")
    print(_row("味方が2人以上スペシャルを合わせた", [e for e in all_eps if e["specials_synced"]]))
    print(_row("合わせなかった（使ったが単発）", [e for e in all_eps if e["ally_specials_used"] and not e["specials_synced"]]))
    print(_row("誰もスペシャルを使わなかった", [e for e in all_eps if not e["ally_specials_used"]]))
    long_ = [(e, early(e, m)) for m, eps in per_match for e in eps if e["seconds"] > EARLY]
    if long_:
        print(f"\n■ 確保されてから最初の{int(EARLY)}秒に何をしたか（{int(EARLY)}秒以内に終わらなかった打開 {len(long_)}回）")
        print(_row_after("味方が1人も落ちなかった", [x for x in long_ if x[1]["ally_deaths"] == 0]))
        print(_row_after("味方が1人落ちた", [x for x in long_ if x[1]["ally_deaths"] == 1]))
        print(_row_after("味方が2人以上落ちた", [x for x in long_ if x[1]["ally_deaths"] >= 2]))
        print(_row_after("敵を1人以上倒した", [x for x in long_ if x[1]["enemy_deaths"] >= 1]))
        print(_row_after("敵を1人も倒せなかった", [x for x in long_ if x[1]["enemy_deaths"] == 0]))
        print(_row_after("スペシャルを使わなかった", [x for x in long_ if x[1]["n_specials"] == 0]))
        print(_row_after("味方3人以上いるときに最初のスペシャル", [x for x in long_ if (x[1]["alive_at_first_special"] or 0) >= 3]))
        print(_row_after("味方2人以下のときに最初のスペシャル", [x for x in long_ if x[1]["alive_at_first_special"] is not None and x[1]["alive_at_first_special"] <= 2]))
        print(_row_after("2人以上でスペシャルを合わせた", [x for x in long_ if x[1]["synced"]]))
        print(_row_after("スペシャルは単発だった", [x for x in long_ if x[1]["n_specials"] >= 1 and not x[1]["synced"]]))
        mine = [x for x, (m) in [((e, early(e, m)), m) for m, eps in per_match for e in eps if e["seconds"] > EARLY] if m.get("me_slot")]
        print(_row_after("自分は落ちなかった", [x for x in mine if not x[1]["me_died"]]))
        print(_row_after("自分が落ちた", [x for x in mine if x[1]["me_died"]]))
    with_me = [e for e, m in ((e, m) for m, eps in per_match for e in eps) if m.get("me_slot")]
    if with_me:
        print("\n■ 自分")
        print(_row("打開中に自分が倒されなかった", [e for e in with_me if e["me_deaths"] == 0]))
        print(_row("打開中に自分が1回以上倒された", [e for e in with_me if e["me_deaths"] > 0]))
    known = [(m, eps) for m, eps in per_match if m.get("result") in ("勝ち", "負け")]
    if known:
        print("\n■ 勝ち負けとの関係（1試合の平均）")
        for res in ("勝ち", "負け"):
            g = [eps for m, eps in known if m["result"] == res]
            if g:
                print(f"  {res}: 確保された回数 {_avg([len(x) for x in g]):.1f}回 / "
                      f"取られたカウント合計 {_avg([sum(e['count_lost'] for e in x) for x in g]):.1f} / "
                      f"打開までの平均 {_avg([e['seconds'] for x in g for e in x if e['success']]):.1f}秒")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="記録の保存先")
    a = ap.parse_args()
    report([m for m in Store(a.data).matches() if not m.get("interrupted")])


if __name__ == "__main__":
    main()
