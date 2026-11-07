"""1コマの画面（1920x1080、RGB）から読み取れることをまとめる。

- 画面上部の8人のアイコン: 生きている / 倒されている / スペシャルが溜まっている
- 自分が倒されたときの右下の「復活まであと〇秒」の黄色い帯
- 試合後の WIN! / LOSE... と、試合前のルール表示
"""
import os

import cv2
import numpy as np

W, H = 1920, 1080
SLOT_NAMES = ["左1", "左2", "左3", "左4", "右1", "右2", "右3", "右4"]  # 左が自分のチーム
SLOT_X = [560, 650, 740, 830, 1085, 1172, 1258, 1345]
SLOT_Y = 72
HALF = 30
RESPAWN_BOX = (1610, 1910, 960, 1020)  # x0, x1, y0, y1
RESULT_BOX = (0, 0, 300, 160)          # x, y, w, h（左上の WIN! / LOSE...）
RULE_BOX = (820, 330, 280, 290)        # 画面中央の黒いしぶきに出るルール名

DEAD, ALIVE, SPECIAL = 0, 1, 2

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
RULE_NAMES = {"area": "ガチエリア", "yagura": "ガチヤグラ", "hoko": "ガチホコ", "asari": "ガチアサリ"}


def _mx_sat(rgb):
    rgb = rgb.astype(np.int16)
    mx = rgb.max(axis=2)
    return mx, mx - rgb.min(axis=2)


def read_slots(img):
    """各アイコンを SPECIAL / ALIVE / DEAD / None（読めない）に分ける。"""
    out = []
    for x in SLOT_X:
        v, s = _mx_sat(img[SLOT_Y - HALF:SLOT_Y + HALF, x - HALF:x + HALF])
        colored = ((s > 90) & (v > 110)).mean()           # チーム色
        dark = ((v < 80) & (s < 40)).mean()               # 倒されたときの灰色・黒
        pale = ((v > 170) & (s > 25) & (s <= 90)).mean()  # スペシャルが溜まると白っぽく光る
        if pale > 0.25 and colored < 0.25 and dark < 0.3:
            out.append(SPECIAL)
        elif colored >= 0.04 and dark < 0.3:
            out.append(ALIVE)
        elif colored < 0.03 and 0.2 < dark < 0.65:
            out.append(DEAD)
        else:
            out.append(None)
    return out


def me_respawning(img):
    x0, x1, y0, y1 = RESPAWN_BOX
    p = img[y0:y1, x0:x1].astype(np.int16)
    return bool(((p[..., 0] > 200) & (p[..., 1] > 200) & (p[..., 2] < 90)).mean() > 0.1)


GAUGE_CENTER = (1780, 115)  # 右上のスペシャルゲージの中心
_yy, _xx = np.mgrid[-95:96, -95:96]
_RING = (np.hypot(_yy, _xx) > 60) & (np.hypot(_yy, _xx) < 90)


def me_special_full(img):
    """自分のスペシャルゲージが満タンか。満タンになると目盛りが白っぽく光る。"""
    cx, cy = GAUGE_CENTER
    v, s = _mx_sat(img[cy - 95:cy + 96, cx - 95:cx + 96])
    return bool(((v > 200) & (s < 90))[_RING].mean() > 0.12)


# カウント（のこり〇〇）の箱。上部アイコンの下、残り時間の左右にある
COUNT_Y = (152, 208)                      # 数字の行
COUNT_BOXES = [(793, 905), (1015, 1128)]  # 左（味方）・右（敵）の x 範囲
_DIGITS = None


def _digit_protos():
    global _DIGITS
    if _DIGITS is None:
        d = np.load(os.path.join(TEMPLATE_DIR, "digits.npz"))
        _DIGITS = {k[1:]: d[k] for k in d.files}  # "0"〜"9" と "x"（数字でないもの）
    return _DIGITS


def _read_box(b):
    """箱の切り抜きから (数, 箱が色で塗られているか)。読めなければ (None, None)。"""
    b = b.astype(np.int16)
    bg = np.median(b.reshape(-1, 3), axis=0)
    filled = bool(bg.max() - bg.min() > 80 and bg.max() > 100)
    fg = ((np.abs(b - bg).max(axis=2) > 70) & (b.max(axis=2) > 150)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    protos = _digit_protos()
    chars = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h >= 28 and a >= 60 and w <= 45 and x > 0 and x + w < fg.shape[1] and y > 0:
            g = cv2.resize((lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255, (20, 30),
                           interpolation=cv2.INTER_AREA).flatten().astype(np.float32) / 255
            d = min(protos, key=lambda k: float(((protos[k] - g) ** 2).sum()))
            if d != "x":
                chars.append((x, d))
    if not chars or len(chars) > 3:
        return None, None
    v = int("".join(c for _, c in sorted(chars)))
    return (v, filled) if v <= 100 else (None, None)


PENALTY_Y = (214, 260)                       # ペナルティ（+〇〇）の行
PENALTY_BOXES = [(800, 925), (995, 1115)]    # 左（味方）・右（敵）


_PEN = None


def _pen_protos():
    global _PEN
    if _PEN is None:
        d = np.load(os.path.join(TEMPLATE_DIR, "pen_digits.npz"))
        _PEN = {k[1:]: d[k] for k in d.files}  # "0"〜"9"、"plus"（+）、"x"（数字でないもの）
    return _PEN


def _read_penalty_box(b):
    """「+18」のような文字を読む。出ていなければ 0、読めなければ None。
    文字は白いときと灰色のときがあるので、明るさの境目は切り抜きごとに自動で決める（大津の方法）。"""
    v = b.max(axis=2).astype(np.uint8)
    _, fg = cv2.threshold(v, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    protos = _pen_protos()
    chars = []
    for i in sorted(range(1, n), key=lambda i: st[i][0]):
        x, y, w, h, a = st[i]
        if not (16 <= h <= 40 and a >= 30 and w <= 35 and y > 0 and y + h < fg.shape[0]):
            continue
        g = cv2.resize((lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255, (20, 30),
                       interpolation=cv2.INTER_AREA).flatten().astype(np.float32) / 255
        chars.append(min(protos, key=lambda k: float(((protos[k] - g) ** 2).sum())))
    if "plus" not in chars:
        return 0  # 「+」が無い＝ペナルティは出ていない
    digits = [c for c in chars[chars.index("plus") + 1:] if c not in ("plus", "x")]
    if not digits or len(digits) > 2:
        return None
    return int("".join(digits))


def read_penalty(img):
    """(味方のペナルティ, 敵のペナルティ)。ガチエリアで確保し返した側に付く、カウントが減り始めるまでの残り。"""
    y0, y1 = PENALTY_Y
    return tuple(_read_penalty_box(img[y0:y1, x0:x1]) for x0, x1 in PENALTY_BOXES)


def read_counts(img):
    """(味方のカウント, 敵のカウント, 味方の箱が塗られているか, 敵の箱が塗られているか)。
    カウントが出ていない画面やルールでは None。ガチエリアでは、箱が塗られている側がエリアを確保している。"""
    y0, y1 = COUNT_Y
    (l, lf), (r, rf) = (_read_box(img[y0:y1, x0:x1]) for x0, x1 in COUNT_BOXES)
    if l is None or r is None:
        return None
    return l, r, lf, rf


# ヤグラ・ホコの位置を示す、画面上部の点線の棒
BAR_Y = 155
BAR_LEFT, BAR_RIGHT, BAR_CENTER = 515, 1405, 960


def read_objective_pos(img):
    """ヤグラ・ホコの位置を -1（左＝味方のゴール）〜 +1（右＝敵のゴール）で返す。見つからなければ None。
    棒の上の丸い目印（ヤグラ・ホコ）を円として探す。四角い目印（チェックポイント）は棒からわずかに上下にずれるので外す。"""
    band = img[BAR_Y - 30:BAR_Y + 30, BAR_LEFT + 25:BAR_RIGHT - 25]
    g = cv2.medianBlur(cv2.cvtColor(np.ascontiguousarray(band), cv2.COLOR_RGB2GRAY), 3)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=120, param2=22,
                          minRadius=11, maxRadius=34)
    if cs is None:
        return None
    for x, y, r in cs[0]:
        if 25 <= y <= 34:  # 棒の高さにちょうど乗っている円だけ
            return float((x + BAR_LEFT + 25 - BAR_CENTER) / (BAR_RIGHT - BAR_CENTER))
    return None


def bar_band(img):
    """チェックポイント探し用に、棒のまわりを白黒で切り出す。"""
    return cv2.cvtColor(np.ascontiguousarray(img[BAR_Y - 30:BAR_Y + 30, 480:1440]), cv2.COLOR_RGB2GRAY)


def find_checkpoints(bands):
    """試合序盤の棒の切り抜きを何枚も重ねて中央値をとると、動く背景がぼけて棒の上の目印だけがくっきり残る。
    目印（チェックポイント）の位置を -1〜+1 の並びで返す。両端のゴールと中央のヤグラ・ホコは除く。"""
    med = np.median(np.stack(bands), axis=0).astype(np.float32)
    mag = np.hypot(cv2.Sobel(med, cv2.CV_32F, 1, 0), cv2.Sobel(med, cv2.CV_32F, 0, 1))
    rows = list(range(17, 23)) + list(range(38, 44))  # 点線より少し上と下（点は小さいので引っかからない）
    e = np.convolve((mag[rows] > 80).sum(axis=0).astype(float), np.ones(5) / 5, mode="same")
    e[:540 - 480] = 0   # 両端のゴールの丸は先に消しておく（近くの目印とくっつかないように）
    e[1380 - 480:] = 0
    blobs = []
    for c in np.where(e >= 2.5)[0]:
        if blobs and c - blobs[-1][-1] <= 25:  # ホコの山形の目印は2つに割れやすいのでまとめる
            blobs[-1].append(c)
        else:
            blobs.append([c])
    out = []
    for bl in blobs:
        if len(bl) < 10:
            continue
        p = (480 + float(np.mean(bl)) - BAR_CENTER) / (BAR_RIGHT - BAR_CENTER)
        if 0.1 < abs(p) < 0.93:
            out.append(round(p, 3))
    return out


_yy, _xx = np.mgrid[0:80, 0:100]
_PIE_RING = (np.hypot(_yy - 40, _xx - 50) > 26) & (np.hypot(_yy - 40, _xx - 50) < 33)


def asari_goal_open(img):
    """ガチアサリで、どちらかのゴールが割られて開いているか。
    開いている間は、カウントの間の小さな目印が大きな円グラフ（残り時間）に変わる。"""
    c = img[120:200, 910:1010].astype(np.int16)
    mx = c.max(axis=2)
    s = mx - c.min(axis=2)
    return bool(((mx < 60) | ((s > 90) & (mx > 110)))[_PIE_RING].mean() > 0.75)


def _crop(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def white(img):
    return img.min(axis=2) > 200


def dark(img):
    return img.max(axis=2) < 50


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def _load_mask(name, width=None):
    data = np.fromfile(os.path.join(TEMPLATE_DIR, name), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)[..., ::-1]
    m = white(img)
    return m[:, :width] if width else m


class ScreenReader:
    """勝敗とルールの表示を、見本の白文字の形と比べて読む。"""

    def __init__(self):
        w = RESULT_BOX[2]
        self.win = _load_mask("win.png", w)
        self.lose = _load_mask("lose.png", w)
        self.rules = {RULE_NAMES[k]: _load_mask(f"rule_{k}.png") for k in RULE_NAMES}

    def result(self, img):
        """('勝ち'|'負け', 一致度) か (None, 一致度)。"""
        m = white(_crop(img, RESULT_BOX))
        a, b = iou(m, self.win), iou(m, self.lose)
        if max(a, b) < 0.45:
            return None, max(a, b)
        return ("勝ち" if a > b else "負け"), max(a, b)

    def rule(self, img):
        """ルール表示が出ているコマなら (ルール名, 一致度)、出ていなければ (None, 0)。"""
        c = _crop(img, RULE_BOX)
        if dark(c).mean() < 0.45:
            return None, 0.0
        m = white(c)
        best, score = None, 0.0
        for k, t in self.rules.items():
            q = iou(m, t)
            if q > score:
                best, score = k, q
        return (best, score) if score >= 0.4 else (None, score)
