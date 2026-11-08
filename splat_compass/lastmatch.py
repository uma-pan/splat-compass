"""試合履歴の一覧と、1試合の詳しい振り返り。

試合が終わったら、その試合だけを調べて出す。全試合を集計するまとめのページより軽い。
- 結果と、自分のキル・デス・スペシャルを普段と比べた数字
- 勝率の推移（自分が倒した・倒された・スペシャルを使った時刻の印つき）とカウントの推移
- 振り返りの指摘
- 倒された1回ごとの状況（録画があればその場面を再生）
- 勝率が大きく動いた場面
- 8人のブキと、それぞれが倒された回数
- マップ画面と塗りの割合
"""
import html
import json
import os

from . import clips, clock, coach, feedback, learning, weapons, winprob
from .hud import SLOT_NAMES, STAGES
from .storage import MODES, mode_of

ESC = html.escape


def _mmss(sec):
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def saved_matches(store):
    return sorted((m for m in store.matches(all=True) if m.get("video_end") is not None),
                  key=lambda m: (m.get("played_at", ""), m["_file"]))


def result_of(m):
    return "中断" if m.get("interrupted") else m.get("result") or "勝敗不明"


def history_html(store, filters):
    ms = saved_matches(store)
    if not ms:
        return '<p class="muted">保存された試合がまだありません。「試合中」で映像を読み取ると、ここに追加されます。</p>'
    start, end = filters.get("from", ""), filters.get("to", "")
    if start and end and end < start:
        return '<p class="muted">終了日は開始日以降を選んでください。</p>'
    rule = filters.get("rule", "").strip()
    stage = filters.get("stage", "").strip()
    result, mode = filters.get("result"), filters.get("mode")
    matches = []
    for m in reversed(ms):
        day = m.get("played_at", "")[:10]
        if (start and day < start) or (end and day > end):
            continue
        if rule not in (m.get("rule") or "ルール不明") or stage not in (m.get("stage") or "ステージ不明"):
            continue
        if (result and result_of(m) != result) or (mode and mode_of(m) != mode):
            continue
        matches.append(m)
    if not matches:
        return '<p class="muted">条件に合う試合がありません。</p>'
    page_size = 15
    try:
        page = int(filters.get("page", 0))
    except ValueError:
        page = 0
    page = min(max(page, 0), (len(matches) - 1) // page_size)
    first = page * page_size
    rows = []
    for m in matches[first:first + page_size]:
        arg = ESC(json.dumps(m["_file"]))
        when = ESC(m.get("played_at", "")[:16].replace("T", " "))
        rows.append(f'<tr><td>{when}</td><td>{ESC(m.get("rule") or "ルール不明")}</td>'
                    f'<td>{ESC(m.get("stage") or "ステージ不明")}</td><td>{ESC(result_of(m))}</td><td>{ESC(mode_of(m))}</td>'
                    f'<td><button class="btn" aria-label="{when} の試合を開く" onclick="loadLast({arg}, true)">開く</button></td></tr>')
    prev = f'<button class="btn" onclick="loadHistory({page - 1})">前の15件</button> ' if page else ""
    following = f' <button class="btn" onclick="loadHistory({page + 1})">次の15件</button>' if first + page_size < len(matches) else ""
    return (f'<div class="card"><table><tr><th>日時</th><th>ルール</th><th>ステージ</th><th>勝敗</th><th>分類</th><th>詳細</th></tr>'
            + "".join(rows) + f'</table><p>{prev}{len(matches)}件中 {first + 1}〜{first + len(rows)}件{following}</p></div>')


def _chart(m, pr, me_events, kills, W=720, H=200):
    """勝率の線と、自分の出来事の印。カウントの進み（100 − 残り）も薄く描く。"""
    dur = max(m["video_end"] - m["video_start"], 1)
    x = lambda rel: 40 + (W - 50) * min(max(rel / dur, 0), 1)
    y = lambda p: 10 + (H - 40) * (1 - p)
    parts = [f'<line x1="40" y1="{y(0.5):.1f}" x2="{W - 10}" y2="{y(0.5):.1f}" stroke="#ccc" stroke-dasharray="4 4"/>',
             f'<text x="34" y="{y(1) + 4:.1f}" text-anchor="end" font-size="11" fill="#888">100%</text>',
             f'<text x="34" y="{y(0.5) + 4:.1f}" text-anchor="end" font-size="11" fill="#888">50%</text>',
             f'<text x="34" y="{y(0) + 4:.1f}" text-anchor="end" font-size="11" fill="#888">0%</text>']
    g = clock.go_rel(m)
    g = clock.GO_OFFSET if g is None else g
    for mm in range(5, -1, -1):  # 残り 5:00, 4:00, … の位置に目盛り
        rel = g + (5 - mm) * 60
        if rel <= dur:
            parts.append(f'<text x="{x(rel):.1f}" y="{H - 8}" text-anchor="middle" font-size="11" fill="#888">{mm}:00</text>')
    tl = m.get("timeline") or []
    if tl:  # カウントの進み（残りが減るほど上に行く。味方は青、敵は橙）
        for col, color in ((1, "#9ec5fe"), (2, "#ffd0a8")):
            pts = " ".join(f"{x(r[0] - m['video_start']):.1f},{y(1 - r[col] / 100):.1f}" for r in tl if r[col] is not None)
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    if pr is not None:
        tt, pp = pr
        pts = " ".join(f"{x(t - m['video_start']):.1f},{y(p):.1f}" for t, p in zip(tt, pp))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#1f6feb" stroke-width="2"/>')
    for rel, kind in me_events:
        color = {"倒された": "#c0392b", "スペシャル": "#d4a000"}[kind]
        parts.append(f'<circle cx="{x(rel):.1f}" cy="{H - 26}" r="4" fill="{color}"><title>{clock.fmt(m, rel)} {kind}</title></circle>')
    for rel in kills:
        parts.append(f'<circle cx="{x(rel):.1f}" cy="{H - 34}" r="4" fill="#1a7f37"><title>{clock.fmt(m, rel)} 倒した</title></circle>')
    legend = ('<div class="muted" style="font-size:12px">' + ('濃い青線は推定勝率。' if pr is not None else '')
              + '薄い線はカウントの進み（青が味方・橙が相手）。 '
              '<span style="color:#1a7f37">●</span>倒した　<span style="color:#c0392b">●</span>倒された　'
              '<span style="color:#d4a000">●</span>スペシャル</div>')
    return f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;background:#fff;border-radius:8px">{"".join(parts)}</svg>{legend}'


def exported_clips(store, m, link):
    items = []
    for name, path in clips.existing(store, m):
        url = ESC(f"{link(path)}?v={os.stat(path).st_mtime_ns}")
        ext = "WebM" if path.endswith(".webm") else "MP4"
        items.append(f'<div class="clip-item"><strong>{ESC(name)}</strong> <span class="muted">{ext}</span> '
                     f'<button class="btn" type="button" data-src="{url}" onclick="playExport(this)">アプリ内で再生</button> '
                     f'<a href="{url}" download="{ESC(os.path.basename(path))}">ファイルを保存</a>'
                     '<p class="muted" role="status"></p><video class="clip-video" controls preload="none" hidden></video></div>')
    return "".join(items) or '<p class="muted">書き出した動画はここに表示されます。</p>'


def html_for(store, link, video_dir=None, file=None):
    ms = saved_matches(store)
    if not ms:
        return '<p class="muted">保存された試合がまだありません</p>'
    m = next((x for x in ms if x["_file"] == file), None) if file else ms[-1]
    if m is None:
        return '<p class="muted">選んだ試合が見つかりません。</p>'
    i = ms.index(m)
    base = feedback.baseline([x for x in ms[:i] if not x.get("interrupted")])
    me = m.get("me_slot")
    dur = m["video_end"] - m["video_start"]
    res = result_of(m)
    model = winprob.load_model(store) if m.get("rule") == "ガチエリア" else None
    pr = learning.predict_match(store, m, model)
    wp = coach.wp_func(store, m)

    file_arg = ESC(json.dumps(m["_file"]))
    out = []
    names = weapons.weapons_of(store, m) if m.get("weapon_icons") else None
    my_weapon = names[SLOT_NAMES.index(me)] if names and me and names[SLOT_NAMES.index(me)] else None
    stage = m.get("stage")
    out.append(f'<h2 class="lm-title" data-match="{ESC(m["_file"])}">{ESC(m.get("played_at", "")[:16].replace("T", " "))}　{ESC(m.get("rule") or "ルール不明")}　'
               f'{ESC(stage or "")}　<span class="res-{ESC(res)}">{ESC(res)}</span>　<span class="muted">{_mmss(dur)}'
               f'{"　" + ESC(my_weapon) if my_weapon else ""}</span></h2>')
    mode = mode_of(m)
    btns = "".join(f'<button class="btn lm-mode{" on" if mode == k else ""}" onclick="setMatchMode({ESC(json.dumps(m["_file"]))}, {ESC(json.dumps(k))})">{k}</button> '
                   for k in MODES)
    out.append(f'<div class="lm-stage">この試合の記録: {btns}</div>')
    if not m.get("interrupted"):
        corrected = ' <span class="muted">手で直した勝敗です</span>' if m.get("result_fixed") else ''
        rc = (m.get("result_check") or {}).get("image")
        evidence = f' <a href="{ESC(link(os.path.join(store.root, "results", rc)))}" target="_blank">判定に使った画像</a>' if rc else ''
        out.append(f'<div class="lm-stage">勝敗を訂正: <button class="btn" onclick="fixResult({file_arg}, \'勝ち\')">勝ち</button> '
                   f'<button class="btn" onclick="fixResult({file_arg}, \'負け\')">負け</button>{evidence}{corrected}</div>')
    stem = os.path.splitext(m["_file"])[0]
    crop = os.path.join(store.root, "stages", stem + ".png")
    dl = "".join(f'<option value="{ESC(n)}">' for n in STAGES)
    image = f'<img src="{ESC(link(crop))}" alt="ステージ判定に使った画像" height="30"> ' if os.path.isfile(crop) else ''
    out.append(f'<form class="lm-stage" onsubmit="setStage({file_arg}, this.elements.stage.value); return false">{image}'
               f'<label>ステージ <input name="stage" list="stagenames" value="{ESC(stage or "")}" placeholder="ステージ不明"></label> '
               f'<datalist id="stagenames">{dl}</datalist><button class="btn" type="submit">訂正を保存</button></form>')

    # 数字
    n = feedback._numbers(m)
    if n is not None:
        mins = n["mins"]

        def cell(label, got, key, more_is_good):
            exp = base.get(key) * mins if base and base.get(key) is not None else None
            mark = ""
            if exp is not None and exp >= 0.5:
                if got >= exp * 1.4 + 0.5:
                    mark = "good" if more_is_good else "bad"
                elif got <= exp * 0.6 - 0.5:
                    mark = "bad" if more_is_good else "good"
            usual = f'<div class="muted">普段 {exp:.1f}</div>' if exp is not None else ""
            return f'<div class="lm-num {mark}"><div class="muted">{label}</div><div class="v">{got}</div>{usual}</div>'
        cells = []
        if n["kills"] is not None:
            cells.append(cell("倒した", len(n["kills"]), "kills", True))
        cells += [cell("倒された", len(n["deaths"]), "deaths", False),
                  cell("スペシャル", len(n["specials"]), "specials", True),
                  cell("溜めたまま倒された", len(n["held"]), "held", False)]
        out.append('<div class="lm-nums">' + "".join(cells) + "</div>")
        if base:
            out.append('<p class="muted">「普段」は、この試合より前のガチ・エンジョイ両方の記録を、今回の試合時間に合わせた値です。</p>')
    else:
        out.append('<p class="muted">この試合は自分の枠が分からなかったので、自分の数字は出せません</p>')

    # 勝率の推移
    me_events = []
    for t, rel, slot, what, *_ in m["events"]:
        if slot == me and "倒された" in what:
            me_events.append((rel, "倒された"))
        elif slot == me and what == "スペシャル発動":
            me_events.append((rel, "スペシャル"))
    kills = [k["rel"] for k in m.get("kills") or []]
    if pr is None:
        title, note = "カウントと出来事の推移", "この試合では推定勝率を表示できません。"
    elif model is not None:
        title, note = "推定勝率の推移", "ガチエリア用モデルによる推定です。"
    else:
        title, note = "推定勝率の推移（参考値）", "ガチエリアで学習した簡易モデルを使っています。"
    out.append(f'<div class="card wide"><h3>{title}</h3><p class="muted">{note}</p>' + _chart(m, pr, me_events, kills) + "</div>")

    # 振り返り
    lis = "".join(f'<li class="k-{ESC(k)}"><b>{ESC(k)}</b> {ESC(t)}</li>' for k, t in feedback.match_feedback(m, base, store))
    out.append(f'<div class="card wide"><h3>振り返り</h3><ul class="fb">{lis}</ul></div>')

    # 録画があれば場面を再生できるようにする
    src, off = None, 0.0
    loc = clips.locate(m, video_dir)  # 録画（カメラから読んだ試合は、同じ時間の OBS の録画）
    if loc is not None:
        src, off = link(loc[0]), loc[1]
        out.append('<div class="card wide" id="scene-player"><h3>元の録画から場面を再生</h3>'
                   '<p id="lvnow" class="muted" role="status">下の場面一覧で「再生」を押すと、ここで流れます。</p>'
                   '<video class="clip-video" id="lv" controls preload="none" hidden></video></div>')
    else:
        out.append('<div class="card wide"><h3>試合の録画</h3><p class="muted">この試合に対応する録画が見つかりません。</p>'
                   '<button class="btn" onclick="document.querySelector(\'nav button[data-tab=scenes]\').click()">録画フォルダの設定を開く</button></div>')

    btns = ""
    if loc is not None:
        btns = "".join(f'<button class="btn" type="button" onclick="makeClips(this, {file_arg}, \'{k}\')">{n}</button> '
                       for k, n in clips.KINDS.items())
        btns = f'<fieldset class="clip-actions"><legend>動画にまとめる場面</legend>{btns}</fieldset>'
    out.append(f'<div class="card wide clip-exports" data-file="{ESC(m["_file"])}"><h3>書き出した動画</h3>{btns}'
               f'<p class="muted clip-status" role="status"></p><div class="clip-list">{exported_clips(store, m, link)}</div></div>')

    def play_btn(t_cam, label):
        """t_cam: 読み取ったときの時刻（試合開始が m["video_start"]）。録画の時刻に直して再生する。"""
        if not src:
            return ""
        t_video = off + (t_cam - m["video_start"])
        a, b = t_video - 8, t_video + 4
        return (f'<button class="btn" onclick="{ESC(f"lmPlay({json.dumps(src)}, {a:.1f}, {b:.1f}, {json.dumps(label, ensure_ascii=False)})")}">再生</button>')

    # 倒された場面
    ds = coach.deaths(m, wp) or []
    if ds:
        trs = "".join(f'<tr><td>{clock.fmt(m, d["rel"])}</td><td style="text-align:left">{ESC(coach.describe(d))}</td>'
                      f'<td>{play_btn(d["t"], clock.fmt(m, d["rel"]) + " 倒された")}</td></tr>' for d in ds)
        out.append(f'<div class="card wide"><h3>倒された場面 {len(ds)}回</h3>'
                   f'<table><tr><th>時刻</th><th style="text-align:left">そのときの状況</th><th></th></tr>{trs}</table></div>')

    # 勝率が大きく動いた場面
    if pr is not None:
        tt, pp = pr
        moves = []
        for k in range(len(tt)):
            j = k
            while j + 1 < len(tt) and tt[j + 1] - tt[k] <= 5:
                j += 1
            if j > k:
                moves.append((pp[j] - pp[k], tt[k], tt[j]))
        moves.sort(key=lambda v: -abs(v[0]))
        picked = []
        for d, a, b in moves:
            if abs(d) < 0.08 or len(picked) >= 6:
                break
            if all(abs(a - p[1]) > 10 for p in picked):
                picked.append((d, a, b))
        if picked:
            rows = []
            for d, a, b in sorted(picked, key=lambda v: v[1]):
                what = [f"{e[2]} {e[3]}" for e in m["events"] if a - 1 <= e[0] <= b and ("倒された" in e[3] or e[3] == "スペシャル発動")]
                what += ["自分が倒した" for k in m.get("kills") or [] if a - 1 <= k["t"] <= b]
                rel = a - m["video_start"]
                rows.append(f'<tr><td>{clock.fmt(m, rel)}</td><td class="{"good" if d > 0 else "bad"}">{d * 100:+.0f}pt</td>'
                            f'<td style="text-align:left">{ESC("、".join(what[:6]) or "（出来事なし。カウントや確保の動き）")}</td>'
                            f'<td>{play_btn(a + 2, clock.fmt(m, rel) + " 推定勝率 " + format(d * 100, "+.0f") + "pt")}</td></tr>')
            out.append('<div class="card wide"><h3>推定勝率が大きく動いた場面（5秒以内に8ポイント以上）</h3>'
                       '<p class="muted">カウントの変化や、味方・相手の動きも含めた推定値です。</p>'
                       '<table><tr><th>時刻</th><th>変化</th><th style="text-align:left">そのとき起きたこと</th><th></th></tr>'
                       + "".join(rows) + "</table></div>")

    # 8人（ブキの名前はここで直せる）
    rv = weapons.review(store, only=m)[0]
    if rv:
        row = rv[0][1]
        deaths_by = {s: 0 for s in SLOT_NAMES}
        sp_by = {s: 0 for s in SLOT_NAMES}
        for e in m["events"]:
            if "倒された" in e[3]:
                deaths_by[e[2]] += 1
            elif e[3] == "スペシャル発動":
                sp_by[e[2]] += 1
        strip = link(os.path.join(store.root, "weapons", m["weapon_icons"]))
        _, all_names = weapons.review(store, last=0)
        dl = '<datalist id="lmwnames">' + "".join(f'<option value="{ESC(n)}">' for n in all_names) + "</datalist>"

        def cell(k):
            s = SLOT_NAMES[k]
            w = row[k]
            if w is None:
                return '<div class="wcell"></div>'
            key, n, conf, side = w
            return (f'<div class="wcell{" me" if s == me else ""}">'
                    f'<div class="wimg" style="background-image:url(\'{strip}\');background-position:-{k * 60}px 0"></div>'
                    f'<input class="{conf}" list="lmwnames" value="{ESC(n or "")}" placeholder="{"分からない" if conf == "none" else ""}" data-key="{ESC(key)}">'
                    f'<div class="saved"></div><div class="muted">倒された {deaths_by[s]}・SP {sp_by[s]}</div></div>')
        out.append('<div class="card wide"><h3>8人</h3>' + dl
                   + '<div class="wrow"><span class="side">味方</span>' + "".join(cell(k) for k in range(4)) + "</div>"
                   + '<div class="wrow"><span class="side">相手</span>' + "".join(cell(k) for k in range(4, 8)) + "</div>"
                   + '<p class="muted">ブキの名前が違っていたら直してください（次から見本として使います）。黄色は候補どまり、赤は分からないブキ、'
                     '太字は自分です。倒した数は自分の分しか読めないので、倒された回数とスペシャルの回数を出しています。</p></div>')

    # マップ
    shots = m.get("map_shots") or []
    if shots:
        inks = {round(x[1]): x[2] for x in m.get("inks") or []}
        imgs = []
        for x in shots:
            ink = inks.get(round(x["rel"]))
            cap = f'{clock.fmt(m, x["rel"])}' + (f' 塗り {ink * 100:.0f}%' if ink is not None else "")
            u = link(os.path.join(store.root, "maps", x["image"]))
            imgs.append(f'<figure style="display:inline-block;margin:4px"><a href="{u}" target="_blank">'
                        f'<img loading="lazy" src="{u}" height="110"></a><figcaption class="muted">{ESC(cap)}</figcaption></figure>')
        out.append('<div class="card wide"><h3>マップ画面</h3>' + "".join(imgs) + "</div>")
    return "".join(out)
