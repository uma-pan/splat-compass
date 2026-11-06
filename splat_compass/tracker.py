"""コマを順に受け取り、試合の区切りと出来事（倒された・復活・スペシャル）を記録する。

録画を後から処理するときも、キャプボの映像をその場で処理するときも同じものを使う。
"""
from dataclasses import dataclass, field

from .hud import DEAD, SLOT_NAMES, SPECIAL, ScreenReader, read_slots

STABLE = 3           # 状態が変わったとみなす連続コマ数
START_FRAMES = 5     # 試合が始まったとみなす連続コマ数
END_SECONDS = 3.0    # 試合が終わったとみなす、表示が消えている秒数
IGNORE_FIRST = 12.0  # 試合開始直後は表示が安定しないので出来事を記録しない（秒）
RESULT_WAIT = 75.0   # 試合後に勝敗の表示を待つ秒数
RULE_KEEP = 40.0     # 試合前に見たルール表示を覚えておく秒数

KIND_LABEL = {
    "dead": "倒された",
    "dead_with_special": "スペシャルを持ったまま倒された",
    "respawn": "復活",
    "special_ready": "スペシャル溜まった",
    "special_used": "スペシャル発動",
}


@dataclass
class Event:
    t: float          # 映像の中の時刻（秒）
    rel: float        # 試合開始からの時刻（秒）
    slot: str         # 左1〜右4（左1が自分）
    kind: str         # KIND_LABEL のキー
    alive: tuple      # (味方の生存数, 敵の生存数)
    specials: tuple   # (味方のスペシャル準備数, 敵のスペシャル準備数)

    def row(self):
        return [round(self.t, 1), round(self.rel, 1), self.slot, KIND_LABEL[self.kind],
                self.alive[0], self.alive[1], self.specials[0], self.specials[1]]


ROW_HEADER = ["映像内の時刻", "試合開始から", "枠", "出来事", "味方の生存", "敵の生存",
              "味方のスペシャル準備", "敵のスペシャル準備"]


@dataclass
class Match:
    start: float
    end: float = None
    rule: str = None
    result: str = None
    events: list = field(default_factory=list)


class Tracker:
    def __init__(self, fps, on_event=None, on_match_end=None):
        self.fps = fps
        self.on_event = on_event or (lambda m, e: None)
        self.on_match_end = on_match_end or (lambda m: None)
        self.screens = ScreenReader()
        self.match = None          # 進行中の試合
        self.finished = None       # 終わって勝敗待ちの試合
        self.cur = None            # 確定している8人の状態
        self.pend, self.pend_n = None, 0
        self.on_n, self.off_since = 0, None
        self.last_rule = (None, -1e9)   # (ルール名, 見た時刻)
        self.special_since = None  # 自分のスペシャルが溜まった時刻
        self.last_t = 0.0

    # ---- 外から見る用 ----
    @property
    def in_match(self):
        return self.match is not None

    def counts(self):
        if not self.cur:
            return None
        c = self.cur
        return {
            "alive": (sum(x != DEAD for x in c[:4]), sum(x != DEAD for x in c[4:])),
            "specials": (sum(x == SPECIAL for x in c[:4]), sum(x == SPECIAL for x in c[4:])),
            "me": c[0],
        }

    # ---- 1コマごと ----
    def feed(self, t, img):
        self.last_t = t
        slots = read_slots(img)
        known = sum(x is not None for x in slots)
        if self.match is None:
            self._idle(t, img, known)
            return
        # 試合中
        if known < 6:
            if self.off_since is None:
                self.off_since = t
            elif t - self.off_since >= END_SECONDS:
                self._end(self.off_since)
                return
        else:
            self.off_since = None
        self._update_slots(t, slots)

    def _idle(self, t, img, known):
        rule, _ = self.screens.rule(img)
        if rule:
            self.last_rule = (rule, t)
        if self.finished is not None:
            res, _ = self.screens.result(img)
            if res:
                self.finished.result = res
                self._close_finished()
            elif t - self.finished.end > RESULT_WAIT:
                self._close_finished()
        if known >= 6:
            self.on_n += 1
            if self.on_n >= START_FRAMES:
                self._start(t)
        else:
            self.on_n = 0

    def _start(self, t):
        if self.finished is not None:
            self._close_finished()
        self.match = Match(start=t)
        if t - self.last_rule[1] < RULE_KEEP:
            self.match.rule = self.last_rule[0]
        self.cur, self.pend, self.pend_n = None, None, 0
        self.on_n, self.off_since = 0, None
        self.special_since = None

    def _end(self, t):
        self.match.end = t
        self.finished, self.match = self.match, None
        self.cur = None
        self.special_since = None

    def _close_finished(self):
        m, self.finished = self.finished, None
        self.on_match_end(m)

    def flush(self):
        """映像の最後で呼ぶ。"""
        if self.match is not None:
            self._end(self.last_t)
        if self.finished is not None:
            self._close_finished()

    def _update_slots(self, t, slots):
        if self.cur is None:
            if None not in slots:
                self.cur = slots
            return
        st = [self.cur[k] if slots[k] is None else slots[k] for k in range(8)]
        if st == self.cur:
            self.pend, self.pend_n = None, 0
            return
        if st == self.pend:
            self.pend_n += 1
        else:
            self.pend, self.pend_n = st, 1
        if self.pend_n < STABLE:
            return
        old, new = self.cur, self.pend
        self.cur, self.pend, self.pend_n = new, None, 0
        if new[0] == SPECIAL and old[0] != SPECIAL:
            self.special_since = t
        elif new[0] != SPECIAL:
            self.special_since = None
        if t - self.match.start < IGNORE_FIRST:
            return
        alive = (sum(x != DEAD for x in new[:4]), sum(x != DEAD for x in new[4:]))
        sps = (sum(x == SPECIAL for x in new[:4]), sum(x == SPECIAL for x in new[4:]))
        for k in range(8):
            a, b = old[k], new[k]
            if a == b:
                continue
            if b == DEAD:
                kind = "dead_with_special" if a == SPECIAL else "dead"
            elif a == DEAD:
                kind = "respawn"
            elif b == SPECIAL:
                kind = "special_ready"
            else:
                kind = "special_used"
            e = Event(t, t - self.match.start, SLOT_NAMES[k], kind, alive, sps)
            self.match.events.append(e)
            self.on_event(self.match, e)
