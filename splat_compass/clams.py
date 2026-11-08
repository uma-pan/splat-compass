"""ガチアサリのアサリの所持のまとめ。

1秒ごとに読んだ8枠の所持表示（0〜7、ガチアサリは "power"、読めない枠は None）は1回ずつだと揺れるので、
直近の数回分をならしてから使う。
- 枠ごと: 直近の数回のうち2回以上ガチアサリが見えていればガチアサリ。それ以外は読めた数の中央値。何も読めなければ前の値を少しだけ引き継ぐ。
"""
from .hud import SLOT_NAMES

POWER = "power"
WINDOW = 3          # 何回分（約1秒ごと）をならすか
PICK_GAP = 5.0      # ガチアサリを持った回を分ける間隔（秒）
CARRY = 3.0         # 読めなかった枠に、前の値を引き継いでよい秒数


def _slot_value(vals):
    # ガチアサリは1回だけ見えたものは読み違えのことが多いので、ならす範囲で2回以上見えたときだけにする
    if sum(v == POWER for v in vals) >= min(2, len(vals)):
        return POWER
    nums = sorted(v for v in vals if isinstance(v, int))
    if not nums:
        return None
    return nums[len(nums) // 2]


def smooth(rows):
    """所持の記録（[時刻, 8枠…] の並び）を、ならした値の並び [(時刻, [8枠])] にする。"""
    out, last, last_t = [], [None] * len(SLOT_NAMES), [None] * len(SLOT_NAMES)
    for i, row in enumerate(rows):
        t = row[0]
        win = rows[max(0, i - WINDOW + 1):i + 1]
        cur = []
        for k in range(len(SLOT_NAMES)):
            if row[1 + k] == 0:  # 倒されている枠（読み取り側で 0 にしている）は、すぐ 0 にする
                v = 0
            else:
                v = _slot_value([r[1 + k] for r in win])
            if v is None and last[k] is not None and last_t[k] is not None and t - last_t[k] <= CARRY:
                v = last[k]
            elif v is not None:
                last[k], last_t[k] = v, t
            cur.append(v)
        out.append((t, cur))
    return out


def current(rows):
    """試合中の今の所持（ならした値）。記録が無ければ None。"""
    if not rows:
        return None
    return smooth(rows[-(WINDOW + 3):])[-1][1]


def totals(slots):
    """[味方の数, 敵の数, ガチアサリを持っている側（'味方'/'相手'/None）]。ガチアサリは数に入れない。"""
    a = sum(v for v in slots[:4] if isinstance(v, int))
    e = sum(v for v in slots[4:] if isinstance(v, int))
    side = "味方" if POWER in slots[:4] else ("相手" if POWER in slots[4:] else None)
    return a, e, side


def summary(m):
    """保存した試合から、アサリの所持のまとめ。記録が無ければ None。
    {series: [(試合開始からの秒, 味方, 敵, ガチアサリの側)], power: {味方: 秒, 相手: 秒}, picks: {味方: 回, 相手: 回},
     peak: (味方, 敵), me: {peak, power_sec, lost: [(試合開始からの秒, 落とした数)]} または None}"""
    rows = m.get("clam_timeline") or []
    if len(rows) < 5:
        return None
    start = m["video_start"]
    sm = smooth(rows)
    series, power, picks = [], {"味方": 0.0, "相手": 0.0}, {"味方": 0, "相手": 0}
    prev_side, prev_t = None, None
    seen = {"味方": None, "相手": None}  # そのチームがガチアサリを最後に持っていた時刻
    for t, slots in sm:
        a, e, side = totals(slots)
        series.append((round(t - start, 1), a, e, side))
        if prev_t is not None and prev_side is not None and t - prev_t <= 3:
            power[prev_side] += t - prev_t
        if side is not None:
            # 読み取りが途切れただけのものを別の回に数えないよう、5秒以上あいたときだけ新しい回とする
            if seen[side] is None or t - seen[side] > PICK_GAP:
                picks[side] += 1
            seen[side] = t
        prev_side, prev_t = side, t
    me = None
    if m.get("me_slot") in SLOT_NAMES:
        k = SLOT_NAMES.index(m["me_slot"])
        mine = [(t, s[k]) for t, s in sm]
        peak = max((v for _, v in mine if isinstance(v, int)), default=0)
        psec = 0.0
        for (t0, v0), (t1, _) in zip(mine, mine[1:]):
            if v0 == POWER and t1 - t0 <= 3:
                psec += t1 - t0
        lost = []
        for ev in m.get("events") or []:
            t, rel, slot, what = ev[:4]
            if slot != m["me_slot"] or "倒された" not in what:
                continue
            before = [v for tt, v in mine if t - 4 <= tt < t - 0.5 and v is not None]
            if not before:
                continue
            v = before[-1]
            if v == POWER or (isinstance(v, int) and v > 0):
                lost.append((round(rel, 1), v))
        me = {"peak": peak, "power_sec": round(psec), "lost": lost}
    return {"series": series, "power": {k: round(v) for k, v in power.items()}, "picks": picks,
            "peak": (max(s[1] for s in series), max(s[2] for s in series)), "me": me}
