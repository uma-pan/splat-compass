"""コマを順に受け取り、試合の区切りと出来事（倒された・復活・スペシャル）を記録する。

録画を後から処理するときも、キャプボの映像をその場で処理するときも同じものを使う。
"""
from collections import deque
from dataclasses import dataclass, field

import cv2

from .hud import (ALIVE, DEAD, SLOT_NAMES, SPECIAL, ScreenReader, asari_goal_open, me_respawning, me_special_full,
                  bar_band, crop_icon, crop_killcam, hud_visible, read_bubble_counts, read_kill_notices, find_checkpoints, is_map_screen, read_map_ink, team_hues, read_counts, read_objective_pos, read_penalty, read_slots)

STABLE_DEAD = 0.6    # 「倒された」と読める状態がこの秒数続いたら倒されたとみなす
STABLE_BACK = 0.3    # 倒されたあと、普通の表示がこの秒数続いたら復活とみなす
STABLE_SP = 0.5      # スペシャルの溜まった表示が出た・消えたとみなすのに見る秒数
QUIET_SEC = 1.5      # 「溜まってすぐ使った」とみなすには、前後これだけ溜まった表示が出ていないこと
MIN_DEAD = 2.5       # 倒されてからこれより早く復活したら、表示の読み違いとみなして取り消す
START_FRAMES = 5     # 試合が始まったとみなす連続コマ数
END_SECONDS = 10.0   # 試合が終わったとみなす、表示が消えている秒数（やられて復活を待つ間のマップ表示で切れないよう長め）
ICON_FROM, ICON_TO = 3.0, 60.0  # ブキのアイコンを切り出し始める秒と、遅くともここまでに決める秒（試合開始から）
ICON_SHOTS = 5  # 1人あたり何枚切り出して選ぶか
BUBBLE_DROP = 6.0  # ヤグラ・ホコのカウントが1秒に減りうる上限（これより急に減ったら読み違いとみなす。ホコを持って泳ぐと速い）
IGNORE_FIRST = 12.0  # 試合開始直後は表示が安定しないので出来事を記録しない（秒）
SELF_WINDOW = (0.5, 12.0)  # 復活待ち表示が出る何秒前〜何秒前に倒された枠を自分の候補にするか
GAUGE_MIN = 10            # ゲージが満タンのコマがこれだけ貯まったら、ゲージで自分の枠を決める
GAUGE_MARGIN = 30         # 一番ゲージと合う枠が、二番目よりこれだけ多く合っていること
RESPAWN_AFTER = 8.0        # 復活待ち表示が出てから何秒以内に復活するか
FAKE_MATCH = 60.0    # これより短い区間は試合ではないとみなす（秒）
NO_TIMER_START = 25.0  # 試合が始まってこの秒数のうちに残り時間が一度も出なければ、試合ではなかったとみなす
NO_TIMER_END = 20.0    # 残り時間がこの秒数出ていなければ、試合は終わったとみなす（マップを長く開いていても切れないよう長め）
CP_FROM, CP_TO = 10.0, 50.0  # チェックポイントを探すために棒を集める区間（試合開始からの秒）
RESULT_WAIT = 75.0   # 試合後に勝敗の表示を待つ秒数
RULE_KEEP = 40.0     # 試合前に見たルール表示を覚えておく秒数
MASS_DEATH = 5       # 同時にこの人数以上が倒れたら、画面の切り替わりとみなす
MASS_SPECIAL = 3     # 同時にこの人数以上のスペシャルが溜まったら、画面の演出で明るく見えたとみなす
SP_MUTE = 1.5        # そのあとこの秒数は「溜まった」を記録しない
END_TRIM = 3.0       # 試合の終わり（Finish! の演出）でこの秒数のあいだの出来事は読み違いとして捨てる

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
    result_check: tuple = None  # 勝敗を決めた画面の縮小画像と一致度（読み違いを確かめる用）
    me_slot: str = None  # 自分の枠（スペシャルゲージと復活待ち表示から推定）
    gauge_votes: list = field(default_factory=lambda: [0, 0, 0, 0])  # 味方4枠それぞれの、ゲージとの一致度
    gauge_full_frames: int = 0
    me_by_gauge: bool = False
    respawn_marks: list = field(default_factory=list)  # 自分の復活待ち表示が出た時刻
    timeline: list = field(default_factory=list)  # 1秒ごとの [時刻, 味方カウント, 敵カウント, 味方の箱が塗られている, 敵の箱が塗られている, 味方ペナルティ, 敵ペナルティ]
    map_shots: list = field(default_factory=list)  # [(時刻, マップ画面の縮小画像)]
    inks: list = field(default_factory=list)  # [(時刻, マップから読んだ味方の塗りの割合)]
    killcams: list = field(default_factory=list)  # [(時刻, 倒されたときの吹き出しの切り抜き)]
    kills: list = field(default_factory=list)  # 自分が相手を倒した時刻（画面の下の「〇〇 をたおした!」）
    icons: list = field(default_factory=lambda: [None] * 8)  # 8人のブキのアイコン（生きている時の絵）
    checkpoints: list = None  # ヤグラ・ホコのチェックポイントの位置（-1〜+1）
    objective: list = field(default_factory=list)  # 1秒ごとの [時刻, ヤグラ・ホコの位置(-1〜1), アサリのゴールが開いている]
    interrupted: bool = False  # 勝敗の表示が出ないまま終わった（回線落ち・録画の途切れなど）
    events: list = field(default_factory=list)


class Tracker:
    def __init__(self, fps, on_event=None, on_match_end=None, on_log=None):
        self.on_log = on_log or (lambda text: None)  # 試合の始まり・終わりを決めた理由（あとで確かめる用）
        self.fps = fps
        self.on_event = on_event or (lambda m, e: None)
        self.on_match_end = on_match_end or (lambda m: None)
        self.screens = ScreenReader()
        self.match = None          # 進行中の試合
        self.finished = None       # 終わって勝敗待ちの試合
        self.cur = None            # 確定している8人の状態
        self.on_n, self.off_since = 0, None
        self.last_rule = (None, -1e9)   # (ルール名, 見た時刻)
        self.ready_at = [None] * 8  # 各枠のスペシャルが溜まった時刻
        self.res_n, self.res_prev = 0, None
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
        res = self._result_seen(img)
        if res:
            if self.finished is not None and t - self.match.start < FAKE_MATCH:
                # 試合後の画面を「試合開始」と読み違えていた。勝敗は直前の試合のもの
                self.match = None
                self.cur = None
                self._set_result(self.finished, res, img)
            else:
                self.on_log(f"勝敗の表示（{res}）を見たので試合を終えます")
                self._end(t)
                if self.finished is not None:
                    self._set_result(self.finished, res, img)
            return
        if known < 6 and (is_map_screen(img) or me_respawning(img)):
            self.off_since = None  # マップを開いている・復活待ち。試合は続いている
        elif known < 6:
            if self.off_since is None:
                self.off_since = t
            elif t - self.off_since >= END_SECONDS:
                self.on_log("上部の8人の表示が消えたので試合を終えます")
                self._end(self.off_since)
                return
        else:
            self.off_since = None
        # 残り時間の表示で、本当に試合中かを確かめる（試合後の画面を8人の表示と読み違えて続いてしまうのを防ぐ）
        if hud_visible(img):
            self.timer_seen = t
        elif self.timer_seen is None and t - self.match.start > NO_TIMER_START:
            self.on_log("残り時間の表示が一度も出ないので、試合ではなかったとみなします")
            self.match, self.cur = None, None  # 一度も残り時間が出ない＝試合ではなかった
            self.ready_at = [None] * 8
            return
        elif self.timer_seen is not None and t - self.timer_seen > NO_TIMER_END:
            self.on_log(f"残り時間の表示が{int(NO_TIMER_END)}秒出ていないので試合を終えます")
            self._end(self.timer_seen)
            return
        resp = me_respawning(img)
        if resp and not self.resp_prev:
            self.match.respawn_marks.append(t)
            self._decide_self()
            kc = crop_killcam(img)
            if kc is not None:
                self.match.killcams.append((t, kc))
        self.resp_prev = resp
        self._kills(t, img)
        if t - self.last_count_t >= 1.0:
            self.last_count_t = t
            c = read_counts(img)
            if c is not None:
                pl, pr = read_penalty(img)
                self.match.timeline.append([round(t, 1), c[0], c[1], int(c[2]), int(c[3]), pl, pr])
            elif self.match.rule in ("ガチヤグラ", "ガチホコ", None):
                self._bubble_counts(t, img)
            on_map = is_map_screen(img)
            # マップ画面が出るたびに1枚残す。出始めは半透明で描かれている途中なので、次に見たとき（約1秒後）に撮る
            if on_map and self.map_prev and not self.map_taken:
                small = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
                self.match.map_shots.append((t, small))
                ink = read_map_ink(small, team_hues(self.match.icons))
                if ink is not None:
                    self.match.inks.append((round(t, 1), round(ink, 3)))
                self.map_taken = True
            if not on_map:
                self.map_taken = False
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
        if ICON_FROM <= rel and any(i is None for i in self.match.icons) and (rel > ICON_TO or hud_visible(img)):
            self._collect_icons(t, rel, img, slots)
        if self.cur is not None:
            self._gauge_vote(me_special_full(img))

    def _kills(self, t, img):
        """「〇〇 をたおした!」の行が増えたら、増えた分だけ倒したとみなす（2コマ続けて同じ行数のときだけ数える）。
        行は数秒で消え、続けて倒すと上に重なって増える。"""
        n = read_kill_notices(img)
        if n == self.kill_pend:
            self.kill_pend_n += 1
        else:
            self.kill_pend, self.kill_pend_n = n, 1
        if self.kill_pend_n == 2 and n != self.kill_n:
            if n > self.kill_n and t - self.match.start >= IGNORE_FIRST:
                self.match.kills.extend([t] * (n - self.kill_n))
            self.kill_n = n

    def _collect_icons(self, t, rel, img, slots):
        """8人のアイコンを1秒おきに何枚か切り出し、ほかの切り出しと一番似ている1枚を選ぶ。
        試合開始の演出の白い帯や、アサリ・ホコを持ったときの印が重なった1枚を避けるため。"""
        from .weapons import feature
        caps = self.icon_caps
        for k in range(8):
            if self.match.icons[k] is not None:
                continue
            c = caps[k]
            if slots[k] == ALIVE and (not c or t - c[-1][0] >= 1.0):
                ic = crop_icon(img, k)
                c.append((t, ic, feature(ic)))
            if len(c) >= ICON_SHOTS or (rel > ICON_TO and c):
                score = [sum(float(f @ g) for _, _, g in c) for _, _, f in c]
                self.match.icons[k] = c[max(range(len(c)), key=lambda i: score[i])][1]

    def _bubble_counts(self, t, img):
        """ガチヤグラ・ガチホコのカウントを吹き出しから読む。カウントは減るだけで、急には減らない。
        そこで、今より少なく、前に読めたときから急に減りすぎていない値が2秒続けて読めたときだけ更新する
        （吹き出しが重なって1文字だけ読めたときの読み違いを避ける）。見えない側はそれまでの値（始めは100）のまま。"""
        vals = read_bubble_counts(img)
        if vals == (None, None) and not self.bub_seen:
            return
        self.bub_seen = True
        for k, v in enumerate(vals):
            if v is None:
                continue
            if v == self.bub[k]:  # 今の値と同じ＝正しく読めている
                self.bub_t[k], self.bub_pend[k] = t, None
                continue
            # 最後に正しく読めたときから、急に減りすぎていないか
            ok = v < self.bub[k] and self.bub[k] - v <= BUBBLE_DROP * max(t - self.bub_t[k], 1.0)
            if not ok:
                self.bub_pend[k] = None
            elif self.bub_pend[k] == v:
                self.bub[k], self.bub_pend[k], self.bub_t[k] = v, None, t
            else:
                self.bub_pend[k] = v
        self.match.timeline.append([round(t, 1), self.bub[0], self.bub[1], 0, 0, None, None])

    def _idle(self, t, img, known):
        rule, _ = self.screens.rule(img)
        if rule:
            self.last_rule = (rule, t)
        if self.finished is not None:
            res = self._result_seen(img)
            if res:
                self._set_result(self.finished, res, img)
        if known >= 6:
            self.on_n += 1
            if self.on_n >= START_FRAMES:
                self._start(t)
        else:
            self.on_n = 0

    def _start(self, t):
        self.on_log("試合が始まりました")
        # 直前の試合は、勝敗の表示を待つためにしばらく閉じずにおく
        self.match = Match(start=t)
        if t - self.last_rule[1] < RULE_KEEP:
            self.match.rule = self.last_rule[0]
        self.cur = None
        self.on_n, self.off_since = 0, None
        self.ready_at = [None] * 8
        self.resp_prev = False
        self.cp_bands = []
        self.map_prev = False
        self.map_taken = False
        self.icon_caps = [[] for _ in range(8)]
        self.timer_seen = None
        self.kill_n, self.kill_pend, self.kill_pend_n = 0, 0, 0
        self.bub, self.bub_pend, self.bub_t, self.bub_seen = [100, 100], [None, None], [t, t], False

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
            self.on_log(f"{int(t - self.match.start)}秒で終わったので試合ではなかったとみなします")
            self.match, self.cur = None, None
            self.ready_at = [None] * 8
            return
        if self.finished is not None:
            self._close_finished()
        self.match.end = t
        self._trim_end(t)
        self._decide_self(final=True)
        self.finished, self.match = self.match, None
        self.cur = None
        self.ready_at = [None] * 8

    def _trim_end(self, t):
        """試合の終わりの演出（Finish! の帯や上部の表示が消えていく途中）で、倒された・溜まったと読み違えたものを捨てる。"""
        m = self.match
        drop = [e for e in m.events if e.t > t - END_TRIM]
        if drop:
            m.events = [e for e in m.events if e.t <= t - END_TRIM]
            self.on_log(f"試合の終わりの演出中に読んだ出来事を{len(drop)}件取り消しました")

    def end_by_hand(self, t, result=None):
        """画面から試合の終わりを読み取れなかったときに、手で試合を終える。result: '勝ち' / '負け' / None"""
        if self.match is not None:
            m = self.match
            if self.finished is not None:
                self._close_finished()
            m.end = t
            self._decide_self(final=True)
            self.finished, self.match = m, None
            self.cur = None
            self.ready_at = [None] * 8
        if self.finished is None:
            return False
        self.finished.by_hand = True
        self.finished.result = result
        self._close_finished()
        return True

    def _result_seen(self, img):
        """勝敗の表示が同じ読みで2コマ続いたら、その読み（'勝ち'/'負け'）。それまでは None。"""
        res, _ = self.screens.result(img)
        if res and res == self.res_prev:
            self.res_n += 1
        else:
            self.res_n = 1 if res else 0
        self.res_prev = res
        if self.res_n >= 2:
            self.res_n = 0
            return res
        return None

    def _set_result(self, m, res, img):
        m.result = res
        m.result_check = (cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA), self.screens.last_scores)
        self._close_finished()

    def _close_finished(self):
        m, self.finished = self.finished, None
        if m.result is None:
            self.on_log("勝敗の表示を読めないまま次に進んだので、勝敗不明として閉じます")
        m.interrupted = m.result is None
        self.on_match_end(m)

    def flush(self):
        """映像の最後で呼ぶ。"""
        if self.match is not None:
            self._end(self.last_t)
        if self.finished is not None:
            self._close_finished()

    def _quick_special(self, k, t0, t1):
        """溜まってすぐ使ったスペシャル。「溜まった」と「発動」を続けて記録する。"""
        if self.match is None or t0 - self.match.start < IGNORE_FIRST:
            return
        c = self.cur
        alive = (sum(x != DEAD for x in c[:4]), sum(x != DEAD for x in c[4:]))
        sps = (sum(x == SPECIAL for x in c[:4]), sum(x == SPECIAL for x in c[4:]))
        for t, kind in ((t0, "special_ready"), (t1, "special_used")):
            e = Event(t, t - self.match.start, SLOT_NAMES[k], kind, alive, sps)
            self.match.events.append(e)
            self.on_event(self.match, e)

    @staticmethod
    def _want(h, cur, t):
        """枠の読みの並び h（直近1秒）から、今の状態として認めるもの。決めきれなければ None。"""
        def last(sec):
            return [x for tt, x in h if tt >= t - sec]
        if cur == DEAD:
            r = last(STABLE_BACK)
            if len(r) >= 2 and all(x in (ALIVE, SPECIAL) for x in r):
                return SPECIAL if sum(x == SPECIAL for x in r) * 2 > len(r) else ALIVE
            return None
        r = last(STABLE_DEAD)
        if len(r) >= 3 and all(x == DEAD for x in r) and h[0][0] <= t - STABLE_DEAD + 0.05:
            return DEAD
        r = [x for x in last(STABLE_SP) if x is not None]
        if cur == ALIVE and len(r) >= 3 and sum(x == SPECIAL for x in r) >= 0.6 * len(r) \
                and h[0][0] <= t - STABLE_SP + 0.05:
            return SPECIAL
        if cur == SPECIAL and len(r) >= 3 and all(x == ALIVE for x in r) and h[0][0] <= t - STABLE_SP + 0.05:
            return ALIVE
        return None

    def _update_slots(self, t, slots):
        """8枠の状態を、枠ごとに少し前からの読みを見て更新する。
        - 倒された: 0.6秒ずっと「倒された」と読めたとき（画面の演出で一瞬暗くなるのを除く）
        - スペシャルが溜まった: 0.5秒のうち6割以上「溜まった」と読めたとき（溜まった表示は明るさが脈打つので）
        - 溜まった表示が消えた: 0.5秒ずっと普通の表示のとき
        - 溜まってすぐ使った（エナジースタンドなど）: 溜まった表示が一瞬だけ出て、前後しばらく出ていないとき"""
        if self.cur is None:
            if None not in slots:
                # 始まった時点でスペシャルが溜まっている人はいない（開始の演出で明るく見えるのを溜まったと読まない）
                self.cur = [ALIVE if x == SPECIAL else x for x in slots]
                self.sp_mute = -1e9
                self.hist = [deque() for _ in range(8)]
                self.quick = [None] * 8   # 一瞬出た溜まった表示（始め, 終わり）。このあと出なければ「すぐ使った」
                self.last_sp = [-1e9] * 8  # 最後に「溜まった」と読めた時刻
                self.last_death = [None] * 8
                self.bursts = []  # 終わった一瞬の光り [(枠, (始め, 終わり))]
            return
        commit = []
        for k in range(8):
            h = self.hist[k]
            h.append((t, slots[k]))
            while h and h[0][0] < t - 1.0:
                h.popleft()
            c = self.cur[k]
            if slots[k] == SPECIAL:
                if c == ALIVE and self.quick[k] is None and t - self.last_sp[k] > QUIET_SEC:
                    self.quick[k] = [t, t]
                elif self.quick[k] is not None:
                    self.quick[k][1] = t
                self.last_sp[k] = t
            q = self.quick[k]
            if q is not None and t - q[1] > QUIET_SEC:  # 一瞬だけ出て、そのあと出ていない
                # 何人もの枠が同時に一瞬光ったのは、画面全体が明るくなる演出とみなす
                others = [(a, b) for kk, (a, b) in self.bursts if kk != k] + \
                         [tuple(self.quick[kk]) for kk in range(8) if kk != k and self.quick[kk] is not None]
                flash = any(a <= q[1] + 0.3 and q[0] <= b + 0.3 for a, b in others)
                if self.cur[k] == ALIVE and q[1] - q[0] < STABLE_SP and not flash:
                    self._quick_special(k, q[0], q[1])
                self.bursts.append((k, (q[0], q[1])))
                self.quick[k] = None
            self.bursts = [(kk, ab) for kk, ab in self.bursts if ab[1] > t - 5]
            want = self._want(h, c, t)
            if want is not None and want != c:
                commit.append((k, want))
        dark = sum(slots[k] == DEAD and self.cur[k] != DEAD for k in range(8))
        if dark >= MASS_DEATH:
            # 5人以上が同時に倒れることはまずない。回線落ちなどで画面が暗くなったとみなし、倒れたことにはしない
            commit = [(k, w) for k, w in commit if w != DEAD]
        if t - self.match.start < IGNORE_FIRST:
            commit = [(k, w) for k, w in commit if w != SPECIAL]  # 始まってすぐ溜まることはない
        rising = [k for k, w in commit if w == SPECIAL and self.cur[k] != SPECIAL]
        recent = sum(1 for e in self.match.events[-8:] if e.kind == "special_ready" and t - e.t < 0.5)
        if len(rising) + recent >= MASS_SPECIAL:
            # 何人ものスペシャルが同時に溜まることはまずない。画面全体が明るくなる演出とみなす
            self.sp_mute = t + SP_MUTE
        if t < self.sp_mute:
            commit = [(k, w) for k, w in commit if not (w == SPECIAL and self.cur[k] != SPECIAL)]
        if not commit:
            return
        old = list(self.cur)
        new = list(self.cur)
        for k, w in commit:
            new[k] = w
            if w == SPECIAL:
                self.quick[k] = None  # ちゃんと溜まった（一瞬ではなかった）
        self.cur = new
        for k, _ in commit:
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
                d = self.last_death[k]
                if d is not None and t - d.t < MIN_DEAD and d in self.match.events:
                    # すぐ戻った＝倒されてはいなかった（画面の演出などで暗く見えただけ）
                    self.match.events.remove(d)
                    self.last_death[k] = None
                    continue
            elif b == SPECIAL:
                kind = "special_ready"
            else:
                kind = "special_used"
            e = Event(t, t - self.match.start, SLOT_NAMES[k], kind, alive, sps)
            self.match.events.append(e)
            if b == DEAD:
                self.last_death[k] = e
            self.on_event(self.match, e)
