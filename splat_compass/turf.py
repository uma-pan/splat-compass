"""ナワバリバトルの終盤を、保存した出来事とマップから振り返る。"""
from . import clock, winprob


def endgame(m):
    """最後の30秒の開始時点と、その後の両チームの動き。"""
    finish, go = m.get("video_end"), m.get("go")
    if m.get("rule") != clock.TURF_RULE or finish is None or m.get("interrupted"):
        return None
    if go is not None and abs(finish - go - clock.TURF_LENGTH) <= 5:
        finish = min(finish, go + clock.TURF_LENGTH)
    elif not clock.TURF_LENGTH - 5 <= finish - m["video_start"] <= clock.TURF_LENGTH + 30:
        return None

    start = finish - clock.TURF_FINAL_SECONDS
    states = winprob._states_at(m)(start)
    teams = [winprob.SLOTS[:4], winprob.SLOTS[4:]]
    alive = [sum(states[s] > 0 for s in team) for team in teams]
    ready = [sum(states[s] == 2 for s in team) for team in teams]
    deaths, specials = [0, 0], [0, 0]
    me = m.get("me_slot")
    own_deaths = own_specials = 0
    for t, _, slot, what, *_ in m["events"]:
        if not start < t <= finish or slot not in states:
            continue
        side = 0 if slot in teams[0] else 1
        if "倒された" in what:
            deaths[side] += 1
            own_deaths += int(slot == me)
        elif what == "スペシャル発動":
            specials[side] += 1
            own_specials += int(slot == me)
    inks = [row for row in (m.get("inks") or []) if start <= row[0] <= finish]
    return {"window": (start, finish), "alive": alive, "ready": ready, "deaths": deaths, "specials": specials,
            "own": (own_deaths, own_specials) if me else None,
            "ink": max(inks, key=lambda row: row[0]) if inks else None}
