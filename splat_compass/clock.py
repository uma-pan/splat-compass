"""時刻の書き方。試合開始からの秒ではなく、ゲーム内の残り時間で書く。

試合の記録の時刻（rel）は「8人の表示が出てからの秒」。残り時間が減り始める（GO）のはその少しあとなので、
試合ごとに読み取った GO の時刻（記録の go）を使う。無い古い記録はおよその差（GO_OFFSET）で書く。
"""
GO_OFFSET = 3.0   # 8人の表示が出てから残り時間が減り始めるまでの、およその秒数（録画で確かめた）
LENGTH = 300      # 試合の長さ（5分）
TURF_RULE = "ナワバリバトル"
TURF_LENGTH = 180
TURF_FINAL_SECONDS = 30


def remain(rel, go_rel=None, length=LENGTH):
    """試合開始からの秒 → 残り時間の文字（"4:59"、延長なら "延長+0:12"）。"""
    g = GO_OFFSET if go_rel is None else go_rel
    s = rel - g
    if s <= length or length == TURF_LENGTH:
        r = max(0, length - max(int(s), 0))
        return f"{r // 60}:{r % 60:02d}"
    o = int(s - length)
    return f"延長+{o // 60}:{o % 60:02d}"


def go_rel(m):
    """試合の記録（dict）か進行中の試合から、GO が 8人の表示から何秒後か。分からなければ None。"""
    if isinstance(m, dict):
        go, start = m.get("go"), m.get("video_start")
    else:
        go, start = getattr(m, "go", None), getattr(m, "start", None)
    return go - start if go is not None and start is not None else None


def length(m):
    rule = m.get("rule") if isinstance(m, dict) else m.rule
    return TURF_LENGTH if rule == TURF_RULE else LENGTH


def fmt(m, rel):
    return remain(rel, go_rel(m), length(m))
