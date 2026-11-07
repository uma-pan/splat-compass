"""コマを順に受け取り、試合の区切りと出来事（倒された・復活・スペシャル）を記録する。

録画を後から処理するときも、キャプボの映像をその場で処理するときも同じものを使う。
"""
from dataclasses import dataclass, field

import cv2

from .hud import (ALIVE, DEAD, SLOT_NAMES, SPECIAL, ScreenReader, asari_goal_open, me_respawning, me_special_full,
                  bar_band, crop_icon, crop_killcam, find_checkpoints, is_map_screen, read_counts, read_objective_pos, read_penalty, read_slots)

STABLE = 3           # 状態が変わったとみなす連続コマ数
START_FRAMES = 5     # 試合が始まったとみなす連続コマ数
END_SECONDS = 3.0    # 試合が終わったとみなす、表示が消えている秒数
ICON_FROM, ICON_TO = 15.0, 60.0  # ブキのアイコンを切り出す時間帯（試合開始からの秒）
IGNORE_FIRST = 12.0  # 試合開始直後は表示が安定しないので出来事を記録しない（秒）
SELF_WINDOW = (0.5, 12.0)  # 復活待ち表示が出る何秒前〜何秒前に倒された枠を自分の候補にするか
GAUGE_MIN = 10            # ゲージが満タンのコマがこれだけ貯まったら、ゲージで自分の枠を決める
GAUGE_MARGIN = 30         # 一番ゲージと合う枠が、二番目よりこれだけ多く合っていること
RESPAWN_AFTER = 8.0        # 復活待ち表示が出てから何秒以内に復活するか
FAKE_MATCH = 60.0    # これより短い区間は試合ではないとみなす（秒）
CP_FROM, CP_TO = 10.0, 50.0  # チェックポイントを探すために棒を集める区間（試合開始からの秒）
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
    me_slot: str = None  # 自分の枠（スペシャルゲージと復活待ち表示から推定）
    gauge_votes: list = field(default_factory=lambda: [0, 0, 0, 0])  # 味方4枠それぞれの、ゲージとの一致度
    gauge_full_frames: int = 0
    me_by_gauge: bool = False
    respawn_marks: list = field(default_factory=list)  # 自分の復活待ち表示が出た時刻
    timeline: list = field(default_factory=list)  # 1秒ごとの [時刻, 味方カウント, 敵カウント, 味方の箱が塗られている, 敵の箱が塗られている, 味方ペナルティ, 敵ペナルティ]
    map_shots: list = field(default_factory=list)  # [(時刻, マップ画面の縮小画像)]
    killcams: list = field(default_factory=list)  # [(時刻, 倒されたときの吹き出しの切り抜き)]
    icons: list = field(default_factory=lambda: [None] * 8)  # 8人のブキのアイコン（生きている時の絵）
    checkpoints: list = None  # ヤグラ・ホコのチェックポイントの位置（-1〜+1）
    objective: list = field(default_factory=list)  # 1秒ごとの [時刻, ヤグラ・ホコの位置(-1〜1), アサリのゴールが開いている]
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
        self.res_n = 0
        self.resp_prev = False
        self.last_t = 0.0
        self.last_count_t = -1e9

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
        if self.finished is not None and t - self.finished.end > RESULT_WAIT:
            self._close_finished()
        slots = read_slots(img)
        known = sum(x is not None for x in slots)
        if self.match is None:
            self._idle(t, img, known)
            return
        # 試合中。勝敗の表示は、上部の表示が消えるより先に出ることがあるので、ここでも見る
        res, _ = self.screens.result(img)
        self.res_n = self.res_n + 1 if res else 0
        if self.res_n >= 2:  # 2コマ続けて見えたら確定（試合中の誤判定を避ける）
            self.res_n = 0
            if self.finished is not None and t - self.match.start < FAKE_MATCH:
                # 試合後の画面を「試合開始」と読み違えていた。勝敗は直前の試合のもの
                self.match = None
                self.cur = None
                self.finished.result = res
                self._close_finished()
            else:
                self._end(t)
                if self.finished is not None:
                    self.finished.result = res
                    self._close_finished()
            return
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
            kc = crop_killcam(img)
            if kc is not None:
                self.match.killcams.append((t, kc))
        self.resp_prev = resp
        if t - self.last_count_t >= 1.0:
            self.last_count_t = t
            c = read_counts(img)
            if c is not None:
                pl, pr = read_penalty(img)
                self.match.timeline.append([round(t, 1), c[0], c[1], int(c[2]), int(c[3]), pl, pr])
            on_map = is_map_screen(img)
            if on_map and not self.map_prev:  # マップ画面が出るたびに1枚残す
                self.match.map_shots.append((t, cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)))
            self.map_prev = on_map
            rule = self.match.rule
            if rule in ("ガチヤグラ", "ガチホコ", None) and self.match.checkpoints is None:
                rel = t - self.match.start
                if CP_FROM <= rel <= CP_TO:
                    self.cp_bands.append(bar_band(img))
                elif rel > CP_TO and len(self.cp_bands) >= 10:
                    self.match.checkpoints = find_checkpoints(self.cp_bands)
                    self.cp_bands = []
            if rule in ("ガチヤグラ", "ガチホコ"):
                p = read_objective_pos(img)
                self.match.objective.append([round(t, 1), None if p is None else round(p, 3), 0])
            elif rule == "ガチアサリ":
                self.match.objective.append([round(t, 1), None, int(asari_goal_open(img))])
            elif rule is None:  # ルールが読めなかった試合は両方記録しておく
                p = read_objective_pos(img)
                self.match.objective.append([round(t, 1), None if p is None else round(p, 3), int(asari_goal_open(img))])
        self._update_slots(t, slots)
        rel = t - self.match.start
        if ICON_FROM <= rel <= ICON_TO and any(i is None for i in self.match.icons):
            for k in range(8):
                if self.match.icons[k] is None and slots[k] == ALIVE:
                    self.match.icons[k] = crop_icon(img, k)
        if self.cur is not None:
            self._gauge_vote(me_special_full(img))

    def _idle(self, t, img, known):
        rule, _ = self.screens.rule(img)
        if rule:
            self.last_rule = (rule, t)
        if self.finished is not None:
            res, _ = self.screens.result(img)
            if res:
                self.finished.result = res
                self._close_finished()
        if known >= 6:
            self.on_n += 1
            if self.on_n >= START_FRAMES:
                self._start(t)
        else:
            self.on_n = 0

    def _start(self, t):
        # 直前の試合は、勝敗の表示を待つためにしばらく閉じずにおく
        self.match = Match(start=t)
        if t - self.last_rule[1] < RULE_KEEP:
            self.match.rule = self.last_rule[0]
        self.cur, self.pend, self.pend_n = None, None, 0
        self.on_n, self.off_since = 0, None
        self.ready_at = [None] * 8
        self.resp_prev = False
        self.cp_bands = []
        self.map_prev = False

    def _gauge_vote(self, full):
        """右上のゲージが満タンのときにスペシャルが溜まっている枠ほど、自分らしい。"""
        v = self.match.gauge_votes
        for k in range(4):
            sp = self.cur[k] == SPECIAL
            if full and sp:
                v[k] += 1
            elif full != sp:
                v[k] -= 1
        if full:
            self.match.gauge_full_frames += 1
        best = max(range(4), key=lambda k: v[k])
        if self.match.gauge_full_frames >= GAUGE_MIN and v[best] - sorted(v)[-2] >= GAUGE_MARGIN:
            self.match.me_slot = SLOT_NAMES[best]
            self.match.me_by_gauge = True

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
        if votes and not self.match.me_by_gauge:  # ゲージで決まっていればそちらを優先
            self.match.me_slot = max(votes, key=votes.get)

    def _end(self, t):
        if t - self.match.start < FAKE_MATCH:
            # 試合後の画面などを試合と読み違えた短い区間。捨てて、直前の試合の勝敗待ちを続ける
            self.match, self.cur = None, None
            self.ready_at = [None] * 8
            return
        if self.finished is not None:
            self._close_finished()
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
