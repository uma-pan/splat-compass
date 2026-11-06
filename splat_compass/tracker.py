"""コマを順に受け取り、試合の区切りと出来事（倒された・復活・スペシャル）を記録する。

録画を後から処理するときも、キャプボの映像をその場で処理するときも同じものを使う。
"""
from dataclasses import dataclass, field

from .hud import DEAD, SLOT_NAMES, SPECIAL, ScreenReader, me_respawning, read_slots

STABLE = 3           # 状態が変わったとみなす連続コマ数
START_FRAMES = 5     # 試合が始まったとみなす連続コマ数
END_SECONDS = 3.0    # 試合が終わったとみなす、表示が消えている秒数
IGNORE_FIRST = 12.0  # 試合開始直後は表示が安定しないので出来事を記録しない（秒）
SELF_WINDOW = (0.5, 12.0)  # 復活待ち表示が出る何秒前〜何秒前に倒された枠を自分の候補にするか
RESPAWN_AFTER = 8.0        # 復活待ち表示が出てから何秒以内に復活するか
RESULT_WAIT = 75.0   # 試合後に勝敗の表示を待つ秒数
RULE_KEEP = 40.0     # 試合前に見たルール表示を覚えておく秒数
MASS_DEATH = 5       # 同時にこの人数以上が倒れたら、画面の切り替わりとみなす

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
    slot: str         # 左1〜右4（自分の位置は試合ごとに違う。Match.me_slot を見る）
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
    me_slot: str = None  # 自分の枠（自分が倒されたときの復活待ち表示から推定）
    respawn_marks: list = field(default_factory=list)  # 自分の復活待ち表示が出た時刻
    interrupted: bool = False  # 勝敗の表示が出ないまま終わった（回線落ち・録画の途切れなど）
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
        self.ready_at = [None] * 8  # 各枠のスペシャルが溜まった時刻
        self.resp_prev = False
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
            "me": c[self.me_index] if self.me_index is not None else None,
        }

    @property
    def me_index(self):
        if self.match is None or self.match.me_slot is None:
            return None
        return SLOT_NAMES.index(self.match.me_slot)

    @property
    def special_since(self):
        """自分のスペシャルが溜まった時刻（自分の枠が分かっていて、溜まっているときだけ）。"""
        i = self.me_index
        return self.ready_at[i] if i is not None else None

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
        resp = me_respawning(img)
        if resp and not self.resp_prev:
            self.match.respawn_marks.append(t)
            self._decide_self()
        self.resp_prev = resp
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
        self.ready_at = [None] * 8
        self.resp_prev = False

    def _decide_self(self, final=False):
        """自分の復活待ち表示が出た時刻と、味方の倒された・復活の時刻を照らし合わせて自分の枠を決める。

        候補は「表示の少し前に倒されて、表示の時点でまだ倒れたまま」の枠。
        試合の終わり（final=True）には「表示のあとすぐ復活した」ことも条件に加える。
        候補が複数なら票を分け合う。
        """
        votes = {}
        evs = [e for e in self.match.events if e.slot.startswith("左")]
        for mk in self.match.respawn_marks:
            cands = []
            for slot in SLOT_NAMES[:4]:
                mine = [e for e in evs if e.slot == slot]
                before = [e for e in mine if e.t <= mk]
                if not before or not before[-1].kind.startswith("dead"):
                    continue  # 表示の時点で倒れていない
                if not SELF_WINDOW[0] <= mk - before[-1].t <= SELF_WINDOW[1]:
                    continue
                if final and not any(e.kind == "respawn" and 0 <= e.t - mk <= RESPAWN_AFTER for e in mine):
                    continue
                cands.append(slot)
            for c in cands:
                votes[c] = votes.get(c, 0) + 1 / len(cands)
        if votes:
            self.match.me_slot = max(votes, key=votes.get)

    def _end(self, t):
        self.match.end = t
        self._decide_self(final=True)
        self.finished, self.match = self.match, None
        self.cur = None
        self.ready_at = [None] * 8

    def _close_finished(self):
        m, self.finished = self.finished, None
        m.interrupted = m.result is None
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
        if sum(a != DEAD and b == DEAD for a, b in zip(old, new)) >= MASS_DEATH:
            # 5人以上が同時に倒れることはまずない。回線落ちなどで画面が暗くなったとみなして無視する
            self.pend, self.pend_n = None, 0
            return
        self.cur, self.pend, self.pend_n = new, None, 0
        for k in range(8):
            if new[k] == SPECIAL and old[k] != SPECIAL:
                self.ready_at[k] = t
            elif new[k] != SPECIAL:
                self.ready_at[k] = None
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
