"""1コマの画面（1920x1080、RGB）から読み取れることをまとめる。

- 画面上部の8人のアイコン: 生きている / 倒されている / スペシャルが溜まっている
- 自分が倒されたときの右下の「復活まであと〇秒」の黄色い帯
- 試合後の WIN! / LOSE... と、試合前のルール表示
"""
import os

import cv2
import numpy as np

W, H = 1920, 1080
SLOT_NAMES = ["左1", "左2", "左3", "左4", "右1", "右2", "右3", "右4"]  # 左1 が自分
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
