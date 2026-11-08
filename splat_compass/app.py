"""splat-compass の画面。映像の読み取り・試合中の表示・まとめのページを1つの窓にまとめる。

  python -m splat_compass.app --camera 1        … カメラ1を読みながら画面を開く
  python -m splat_compass.app --video 録画.mp4  … 録画を実際の速さで流して動作確認
  python -m splat_compass.app                   … 映像は読まず、まとめだけ見る

自分の PC の中だけで動く小さなサーバー（127.0.0.1）を立て、その画面を窓に出す。
pywebview が入っていれば普通のアプリの窓（小さくすると常に手前に出る）、無ければブラウザで開く。
"""
import argparse
import json
import mimetypes
import os
import pathlib
import queue
import threading
import time
import urllib.parse
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import killcam, live, report, weapons
from .sources import camera_frames, video_frames
from .storage import Store

FULL_SIZE = (1200, 880)
COMPACT_SIZE = (380, 600)


# ---------- 状態 ----------
class State:
    """映像を読む側から届いたものを貯めて、画面に渡す。"""

    def __init__(self, log_dir=None):
        self.log_dir = log_dir  # 記録の欄の内容を日ごとのファイルにも残す（あとで確かめる用）
        self.lock = threading.Lock()
        self.state = {"in_match": False}
        self.wp, self.wp_kind = [], None
        self.weapons, self.weapons_file = None, None  # この試合の8人のブキ、保存された試合の名前
        self.last_saved = None  # 最後に保存した試合（次の試合が始まっても残す）
        self.notices = deque(maxlen=20)
        self.logs = deque(maxlen=300)
        self.seq = 0
        self.version = 0      # まとめが変わるたびに増える（画面が読み直す合図）
        self.reading = False  # 映像を読んでいるか
        self.can_start = False  # 画面からカメラを選んで読み始められるか
        self.mode, self.view = "ガチ", "ガチ"  # これから保存する試合の種類、まとめで見る記録

    def _write(self, text):
        if not self.log_dir:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            day = time.strftime("%Y%m%d")
            with open(os.path.join(self.log_dir, f"{day}.txt"), "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {text}\n")
        except OSError:
            pass

    def _push(self, box, text):
        self.seq += 1
        box.append((self.seq, text, time.time()))

    def consume(self, q):
        while True:
            kind, msg = q.get()
            with self.lock:
                if kind == "state":
                    self.state = msg
                elif kind == "wp":
                    rel, p, k = msg
                    self.wp.append((round(rel, 1), round(p, 4)))
                    self.wp_kind = k
                elif kind == "wp_trim":
                    self.wp = [x for x in self.wp if x[0] <= msg]
                elif kind == "wp_reset":
                    self.wp, self.wp_kind = [], None
                    self.weapons, self.weapons_file = None, None
                elif kind == "weapons":
                    self.weapons = msg
                elif kind == "weapons_saved":
                    self.weapons_file = msg
                    self.last_saved = msg
                elif kind == "notice":
                    self._push(self.notices, msg)
                elif kind == "log":
                    self._push(self.logs, msg)
                    self._write(msg)
                elif kind == "report":  # 試合後の学習とまとめが終わったら、画面を読み直させる
                    self.version += 1
                elif kind == "end":
                    self.reading = False

    def snapshot(self, since):
        with self.lock:
            return {"state": self.state, "wp": self.wp, "wp_kind": self.wp_kind, "reading": self.reading,
                    "can_start": self.can_start, "mode": self.mode, "view": self.view,
                    "weapons": self.weapons, "weapons_file": self.weapons_file, "last_saved": self.last_saved,
                    "version": self.version, "seq": self.seq,
                    "logs": [(i, t) for i, t, _ in self.logs if i > since],
                    "notices": [(i, t) for i, t, at in self.notices if i > since and time.time() - at < 6]}


# ---------- 画面の HTML ----------
LIVE_HTML = """
<div class="lv-top">
  <div><div id="wp" class="wpbig">勝率 —</div><div id="wpsub" class="muted"></div></div>
  <button id="compact" class="btn">小さくする</button>
</div>
<div id="notice" class="notice"></div>
<div class="modebar">この試合の記録: <button class="btn mode" data-mode="ガチ" onclick="setConfig({mode:'ガチ'})">ガチ</button>
  <button class="btn mode" data-mode="エンジョイ" onclick="setConfig({mode:'エンジョイ'})">エンジョイ</button>
  <span class="muted">（本気でやる記録と、そうでない記録を分けて集計します）</span></div>
<button id="recam" class="btn" style="float:right" onclick="recam()">カメラを選び直す</button>
<label class="muted" style="display:block;margin-bottom:8px"><input type="checkbox" id="sound" onchange="try{localStorage.setItem('sc-sound', this.checked ? '1' : '0')}catch(e){}"> 通知で音を鳴らす</label>
<div id="reloadbar" class="reloadbar">まとめが新しくなりました <button class="btn" onclick="location.reload()">読み直す</button></div>
<div id="campick" class="card campick"><h3>読み取るカメラ</h3>
  <p class="muted">キャプチャーボードや OBS の仮想カメラを選んでください。スプラトゥーンの画面が映っているものを選びます。</p>
  <button class="btn" onclick="findCameras()">カメラを探す</button>
  <div id="camlist" class="camlist"></div></div>
<svg id="wpchart" class="wpchart" viewBox="0 0 600 150" preserveAspectRatio="none"></svg>
<div class="lv-grid">
  <div class="card"><h3>試合</h3><div id="matchinfo" class="info">試合を待っています</div>
    <div id="endbox" class="endbox">試合が終わったのに続いているときは:
      <button class="btn" onclick="endMatch('勝ち')">勝ちで終える</button>
      <button class="btn" onclick="endMatch('負け')">負けで終える</button>
      <button class="btn" onclick="endMatch(null)">勝敗不明で終える</button></div>
    <div id="lastbox" class="endbox"><span id="lastlabel">直前の試合</span>の勝敗を直す:
      <button class="btn" onclick="fixLast('勝ち')">勝ち</button>
      <button class="btn" onclick="fixLast('負け')">負け</button></div></div>
  <div class="card"><h3>8人</h3><div id="slots" class="slots"></div><div id="alive" class="info"></div></div>
</div>
<div class="card lv-log"><h3>記録</h3><ul id="logs" class="logs"></ul></div>
<div class="card lv-weapons"><h3>8人のブキ</h3><div id="weapons" class="muted">試合が始まると読み取ります</div></div>
"""

APP_CSS = """<style>
.lm-head { margin-bottom:8px; } .lm-head select { font-size:14px; padding:4px; }
.lm-title { font-size:20px; margin:6px 0 10px; }
.lm-stage { margin:-4px 0 10px; } .lm-stage img { vertical-align:middle; background:#333; border-radius:4px; }
.lm-nums { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; }
.lm-num { background:#fff; border-radius:10px; padding:8px 14px; min-width:110px; box-shadow:0 1px 2px rgba(0,0,0,.06); }
.lm-num .v { font-size:28px; font-weight:bold; }
.lm-num.good .v, td.good { color:#1a7f37; } .lm-num.bad .v, td.bad { color:#c0392b; }
#last .card { margin-bottom:12px; } #last td, #last th { white-space:nowrap; } #last td[style*='left'] { white-space:normal; }
.btn { border:1px solid #d1d5db; background:#fff; border-radius:6px; padding:6px 12px; cursor:pointer; font-family:inherit; font-size:13px; }
.lv-top { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px; }
.wpbig { font-size:40px; font-weight:bold; line-height:1.1; }
nav { overflow-x:auto; } nav button { white-space:nowrap; flex:0 0 auto; }
header { flex-wrap:wrap; } .viewsel { white-space:nowrap; }
.modebar { margin-bottom:10px; } .btn.lm-mode.on, .btn.mode.on { background:#1f6feb; color:#fff; border-color:#1f6feb; }
.viewsel { position:absolute; right:16px; top:14px; font-size:13px; color:#fff; } .viewsel select { font-size:13px; }
.reloadbar { display:none; margin-bottom:10px; color:#555; } .reloadbar.on { display:block; }
.campick { display:none; margin-bottom:12px; } .campick.on { display:block; }
.camlist { display:flex; flex-wrap:wrap; gap:10px; margin-top:8px; }
.camlist button { border:1px solid #d0d7de; background:#fff; border-radius:8px; padding:6px; cursor:pointer; font-family:inherit; }
.camlist img { display:block; width:240px; height:135px; object-fit:cover; background:#000; border-radius:4px; margin-bottom:4px; }
.notice { min-height:0; font-size:18px; font-weight:bold; color:#fff; background:#c0392b; border-radius:8px; padding:0 12px; margin:4px 0;
          max-height:0; overflow:hidden; transition:max-height .2s, padding .2s; }
.notice.on { max-height:80px; padding:8px 12px; }
.wpchart { width:100%; height:150px; background:#fff; border-radius:10px; box-shadow:0 1px 2px rgba(0,0,0,.06); margin-bottom:12px; display:block; }
.lv-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; }
.info { font-size:14px; line-height:1.8; }
.slots { display:flex; gap:6px; align-items:center; margin-bottom:6px; }
.slot { width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:13px; color:#fff; font-weight:bold; }
.slot.ally { background:#3d7bd9; } .slot.enemy { background:#e67e22; }
.slot.dead { background:#c8ccd2; color:#6b7280; } .slot.sp { box-shadow:0 0 0 3px #f1c40f; }
.slot.me { outline:2px solid #111; outline-offset:2px; }
.slot.gap { width:10px; background:none; }
.wrow { display:flex; gap:6px; margin:4px 0; flex-wrap:wrap; align-items:flex-start; }
.wrow .side { width:34px; font-size:12px; color:#6b7280; padding-top:28px; }
.wcell { width:118px; font-size:12px; text-align:center; }
.wcell img, .wcell .wimg { width:60px; height:63px; display:block; margin:0 auto 2px; border-radius:6px; }
.wimg { background-repeat:no-repeat; background-size:480px 63px; }
.wcell input { width:100%; font-size:12px; padding:2px 4px; border:1px solid #d1d5db; border-radius:4px; font-family:inherit; }
.wcell input.guess { background:#fff7d6; } .wcell input.none { background:#fde2e1; }
.wcell input.label { border-color:#3d7bd9; } .wcell.me input { font-weight:bold; }
.wcell .saved { color:#1a7f37; font-size:11px; height:14px; }
table.info { border-collapse:collapse; background:#fff; border-radius:10px; font-size:12px; }
table.info th, table.info td { padding:3px 8px; border-top:1px solid #eee; text-align:left; }
table.info tr.empty td:first-child { color:#c0392b; }
table.info input, table.info select { font-size:12px; font-family:inherit; padding:1px 4px; border:1px solid #d1d5db; border-radius:4px; }
.wmatch { background:#fff; border-radius:10px; padding:8px 12px; margin:8px 0; }
.wmatch h2 { font-size:13px; margin:0 0 4px; }
.endbox { display:none; margin-top:8px; font-size:12px; color:#6b7280; }
.endbox.on { display:block; } .endbox .btn { padding:3px 8px; font-size:12px; margin:2px 2px 0 0; }
body.compact .endbox { display:none !important; }
.lv-weapons { margin-top:12px; } body.compact .lv-weapons { display:none; }
.logs { list-style:none; margin:0; padding:0; font-size:13px; max-height:260px; overflow:auto; }
.logs li { padding:3px 0; border-top:1px solid #eee; }
body.compact header, body.compact nav { display:none; }
body.compact main { padding:8px; }
body.compact .tab { display:none !important; } body.compact #live { display:block !important; }
body.compact .wpbig { font-size:32px; }
body.compact .wpchart { height:110px; }
body.compact .lv-grid { grid-template-columns:1fr; gap:8px; margin-bottom:8px; }
body.compact .card { padding:8px 10px; } body.compact .card h3 { display:none; }
body.compact .logs { max-height:120px; }
</style>"""

APP_JS = """<script>
let since = 0, version = null, inMatch = false, lastShown = null;
async function recam() {
  if (inMatch && !confirm("試合の途中です。読むのをやめると、この試合は途中までで保存されます。よいですか？")) return;
  await fetch("/api/stop_camera", { method: "POST", body: "{}" });
  findCameras();
}
async function makeClips(file, kind) {
  const r = await fetch("/api/clips", { method: "POST", body: JSON.stringify({ file: file, kind: kind }) });
  if (!r.ok) alert(await r.text());
}
async function setConfig(c) {
  const r = await fetch("/api/config", { method: "POST", body: JSON.stringify(c) });
  if (!r.ok) { alert(await r.text()); return; }
  if (c.view !== undefined) {  // 見る記録を変えたら、まとめを作り直す
    document.querySelectorAll(".tab").forEach(t => { if (t.id !== "live") t.innerHTML = '<p class="muted">見る記録を切り替えています…</p>'; });
    location.reload();
  }
}
async function setMatchMode(file, mode) {
  const r = await fetch("/api/match_mode", { method: "POST", body: JSON.stringify({ file: file, mode: mode }) });
  if (r.ok) loadLast(file); else alert(await r.text());
}
(function () {  // 見る記録の切り替えを、ページの上に置く
  const h = document.querySelector("header");
  if (!h) return;
  h.style.position = "relative";
  const d = document.createElement("div"); d.className = "viewsel";
  d.innerHTML = '見る記録 <select id="viewsel" onchange="setConfig({view: this.value})">' +
    '<option value="ガチ">ガチだけ</option><option value="エンジョイ">エンジョイだけ</option><option value="すべて">すべて</option></select>';
  h.appendChild(d);
})();
async function setStage(file, name) {
  const r = await fetch("/api/stage", { method: "POST", body: JSON.stringify({ file: file, name: name.trim() }) });
  if (r.ok) loadLast(file); else alert(await r.text());
}
async function loadLast(file) {
  const box = $("lastbody");
  box.innerHTML = '<p class="muted">読み込んでいます…</p>';
  const r = await fetch("/api/last" + (file ? "?file=" + encodeURIComponent(file) : ""));
  box.innerHTML = await r.text();
}
let lmStop = null;
function lmPlay(src, a, b, label) {
  const v = $("lv"); if (!v) return;
  if (v.getAttribute("src") !== src) v.setAttribute("src", src);
  const go = () => { v.currentTime = Math.max(0, a); v.play(); };
  if (v.readyState >= 1) go(); else v.addEventListener("loadedmetadata", go, { once: true });
  lmStop = b; $("lvnow").textContent = label;
  if (!v._lm) { v._lm = true; v.addEventListener("timeupdate", () => { if (lmStop !== null && v.currentTime >= lmStop) { v.pause(); lmStop = null; } }); }
}
document.addEventListener("click", e => {
  const b = e.target.closest && e.target.closest('nav button[data-tab="last"]');
  if (b && lastShown === null) { lastShown = "first"; loadLast(); }
  if (e.target.closest && e.target.closest('nav button[data-tab="wcheck"]')) loadWcheck();  // 開くたびに新しい試合を読む
});
async function loadWcheck() {
  const r = await fetch("/api/wcheck");
  $("wcheckbody").innerHTML = await r.text();
}
async function findCameras() {
  $("camlist").textContent = "探しています…（数秒かかります）";
  const cams = await (await fetch("/api/cameras")).json();
  if (!cams.length) { $("camlist").textContent = "カメラが見つかりませんでした。キャプチャーボードや OBS の仮想カメラが動いているか確かめてください。"; return; }
  $("camlist").innerHTML = cams.map(c => `<button onclick="startCamera(${c.index})"><img src="${c.img}">カメラ ${c.index}（${c.w}×${c.h}）</button>`).join("");
}
async function startCamera(i) {
  const r = await fetch("/api/start_camera", { method: "POST", body: JSON.stringify({ index: i }) });
  if (!r.ok) alert(await r.text());
}
const $ = id => document.getElementById(id);
const mmss = s => Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");

function setCompact(on) {
  document.body.classList.toggle("compact", on);
  $("compact").textContent = on ? "大きくする" : "小さくする";
  try { localStorage.setItem("sc-compact", on ? "1" : ""); } catch (e) {}
  if (window.pywebview && pywebview.api) pywebview.api.set_compact(on);
}
$("compact").addEventListener("click", () => setCompact(!document.body.classList.contains("compact")));
window.addEventListener("pywebviewready", () => {
  try { if (localStorage.getItem("sc-compact")) setCompact(true); } catch (e) {}
});
try { if (localStorage.getItem("sc-compact")) document.body.classList.add("compact"); } catch (e) {}

function beep() {
  let on = false; try { on = localStorage.getItem("sc-sound") === "1"; } catch (e) {}
  if (!on) return;  // 通知の音は、画面で選んだときだけ鳴らす
  try {
    const a = new (window.AudioContext || window.webkitAudioContext)(), o = a.createOscillator(), g = a.createGain();
    o.frequency.value = 880; o.connect(g); g.connect(a.destination); g.gain.value = 0.1;
    o.start(); o.stop(a.currentTime + 0.15);
  } catch (e) {}
}

function drawChart(wp) {
  const W = 600, H = 150, svg = $("wpchart");
  let s = `<line x1="0" y1="${H / 2}" x2="${W}" y2="${H / 2}" stroke="#bbb" stroke-dasharray="4 4"/>`;
  const span = Math.max(300, wp.length ? wp[wp.length - 1][0] + 10 : 0);
  for (let m = 60; m < span; m += 60) s += `<line x1="${m / span * W}" y1="0" x2="${m / span * W}" y2="${H}" stroke="#eee"/>`;
  if (wp.length > 1) {
    const pts = wp.map(([t, p]) => `${(t / span * W).toFixed(1)},${((1 - p) * H).toFixed(1)}`).join(" ");
    s += `<polyline points="${pts}" fill="none" stroke="#3d7bd9" stroke-width="2.5" vector-effect="non-scaling-stroke"/>`;
  }
  svg.innerHTML = s;
}

function showState(d) {
  const st = d.state || {};
  inMatch = !!st.in_match;
  if (!d.reading) {
    $("matchinfo").textContent = "映像を読んでいません（start.bat から開くと、試合中の様子が出ます）";
  } else if (!st.in_match && st.pending) {
    $("matchinfo").textContent = "試合が終わりました。勝敗の表示（左上の WIN! / LOSE...）を待っています";
  } else if (!st.in_match) {
    $("matchinfo").textContent = "試合を待っています";
  } else {
    const c = st.counts ? `のこり 味方 ${st.counts[0]} ／ 敵 ${st.counts[1]}<br>` : "";
    let me = "自分の枠はまだ分かりません（一度倒されると分かります）";
    if (st.me !== null && st.me !== undefined && st.slots) {
      const x = st.slots[st.me];
      me = "自分 " + (x === 0 ? "復活待ち" : x === 2 ? "スペシャルあり" + (st.hold ? `（${Math.floor(st.hold)}秒）` : "") : "生きている");
    }
    let k = st.kills !== undefined ? `<br>倒した ${st.kills}人` : "";
    const pc = st.pace;
    if (pc) {
      const warn = pc.deaths_usual !== null && pc.deaths >= 3 && pc.deaths >= pc.deaths_usual * 1.5 + 1;
      const usual = pc.deaths_usual !== null ? `（普段のペースなら ${pc.deaths_usual.toFixed(1)}回）` : "";
      k = `<br>倒した ${pc.kills}人 ／ 倒された <span style="${warn ? "color:#c0392b;font-weight:bold" : ""}">${pc.deaths}回</span>${usual}`;
    }
    const ink = st.ink ? `<br>塗り 味方 ${Math.round(st.ink[0] * 100)}%（${st.ink[1]}秒前のマップ）` : "";
    $("matchinfo").innerHTML = `${st.rule || "ルール不明"}${st.stage ? "　" + st.stage : ""}　${st.clock ? (st.clock.startsWith("延長") ? st.clock : "残り " + st.clock) : mmss(st.elapsed)}<br>${c}${me}${k}${ink}`;
  }
  $("campick").classList.toggle("on", !d.reading && d.can_start);
  $("recam").style.display = d.reading && d.can_start ? "" : "none";
  document.querySelectorAll(".btn.mode").forEach(b => b.classList.toggle("on", b.dataset.mode === d.mode));
  if ($("viewsel") && document.activeElement !== $("viewsel")) $("viewsel").value = d.view || "すべて";
  $("endbox").classList.toggle("on", !!(d.reading && (st.in_match || st.pending)));
  $("lastbox").classList.toggle("on", !!(d.last_saved && !st.in_match && !st.pending));
  if (lastShown === null && $("last") && $("last").classList.contains("on")) { lastShown = d.last_saved || "first"; loadLast(); }
  if (!window._wc && $("wcheck") && $("wcheck").classList.contains("on")) { window._wc = true; loadWcheck(); }
  if ($("sound") && !window._snd) { window._snd = true; try { $("sound").checked = localStorage.getItem("sc-sound") === "1"; } catch (e) {} }
  // 新しい試合が保存されたら、「直前の試合」タブをその試合に入れ替える（試合中は作らない）
  if (d.last_saved && lastShown !== null && lastShown !== d.last_saved && !st.in_match) {
    lastShown = d.last_saved; if ($("last").classList.contains("on")) loadLast(); else lastShown = null;
  }
  if (d.last_saved) $("lastlabel").textContent = "直前の試合（" + d.last_saved.slice(9, 11) + ":" + d.last_saved.slice(11, 13) + " 開始）";
  let html = "", alive = "";
  if (st.in_match && st.slots) {
    st.slots.forEach((x, k) => {
      if (k === 4) html += '<span class="slot gap"></span>';
      const cls = ["slot", k < 4 ? "ally" : "enemy"];
      if (x === 0) cls.push("dead");
      if (x === 2) cls.push("sp");
      if (k === st.me) cls.push("me");
      html += `<span class="${cls.join(" ")}">${x === 0 ? "×" : ""}</span>`;
    });
    const a = st.slots.slice(0, 4).filter(x => x !== 0).length, e = st.slots.slice(4).filter(x => x !== 0).length;
    const sa = st.slots.slice(0, 4).filter(x => x === 2).length, se = st.slots.slice(4).filter(x => x === 2).length;
    alive = `人数 ${a} 対 ${e}${a < e ? "（不利）" : a > e ? "（有利）" : ""}<br>スペシャル準備 味方 ${sa} ／ 敵 ${se}`;
  }
  $("slots").innerHTML = html; $("alive").innerHTML = alive;
  if (d.wp.length) {
    const p = d.wp[d.wp.length - 1][1];
    $("wp").textContent = (d.wp_kind === "エリア" ? "勝率 " : "勝率の目安 ") + Math.round(p * 100) + "%";
    $("wp").style.color = p >= 0.6 ? "#1a7f37" : p <= 0.4 ? "#c0392b" : "";
    $("wpsub").textContent = d.wp_kind === "エリア" ? "" : "エリアの記録から作った簡単なモデルの値";
  } else {
    $("wp").textContent = "勝率 —"; $("wp").style.color = ""; $("wpsub").textContent = "";
  }
  drawChart(d.wp);
}

let wsig = "";
function weaponCell(w, k, file, me) {
  const conf = w.conf || "none";
  const val = w.name || "";
  const ph = conf === "none" ? "分からない" : "";
  const dis = file ? "" : 'disabled title="試合が保存されたら直せます"';
  const key = file ? file + ":" + k : "";
  return `<div class="wcell${k === me ? " me" : ""}"><img src="${w.img}"><input class="${conf}" list="wnames" value="${val.replace(/"/g, "&quot;")}" placeholder="${ph}" data-key="${key}" ${dis}><div class="saved"></div></div>`;
}
function showWeapons(d) {
  const st = d.state || {};
  const sig = JSON.stringify([d.weapons && d.weapons.map(w => [w.name, w.conf]), d.weapons_file, st.me]);
  if (sig === wsig || (document.activeElement && document.activeElement.closest && document.activeElement.closest("#weapons"))) return;
  wsig = sig;
  if (!d.weapons) { $("weapons").innerHTML = '<span class="muted">試合が始まると読み取ります</span>'; return; }
  const me = st.me === undefined ? null : st.me;
  const row = (a, b, label) => `<div class="wrow"><span class="side">${label}</span>` +
    d.weapons.slice(a, b).map((w, i) => weaponCell(w, a + i, d.weapons_file, me)).join("") + "</div>";
  $("weapons").innerHTML = row(0, 4, "味方") + row(4, 8, "相手") +
    `<div class="muted">${d.weapons_file ? "違っていたら名前を直してください（次から見本として使われます）" : "黄色は候補どまり、赤は分からないブキ。試合が保存されたら直せます"}</div>`;
}
document.addEventListener("change", async e => {
  const el = e.target;
  if (el.dataset && el.dataset.info) {  // ブキの情報の表
    const r = await fetch("/api/weapon_info", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: el.dataset.info, field: el.dataset.field, value: el.value.trim()})});
    el.style.borderColor = r.ok ? "#1a7f37" : "#c0392b";
    return;
  }
  if (!el.dataset || !el.dataset.key) return;
  const r = await fetch("/api/weapon", {method: "POST", headers: {"Content-Type": "application/json"},
                                        body: JSON.stringify({key: el.dataset.key, name: el.value.trim()})});
  const msg = el.parentElement.querySelector(".saved");
  if (r.ok) { el.className = "label"; if (msg) msg.textContent = "保存しました"; }
  else if (msg) msg.textContent = "保存できませんでした";
});

async function poll() {
  try {
    const d = await (await fetch("/api/live?since=" + since)).json();
    window._last = d;
    showState(d);
    showWeapons(d);
    for (const [i, t] of d.logs) {
      const li = document.createElement("li"); li.textContent = t; $("logs").appendChild(li);
    }
    if (d.logs.length) $("logs").scrollTop = $("logs").scrollHeight;
    if (d.notices.length) {
      $("notice").textContent = d.notices[d.notices.length - 1][1]; $("notice").classList.add("on"); beep();
      clearTimeout(window._nt); window._nt = setTimeout(() => $("notice").classList.remove("on"), 4000);
    }
    since = Math.max(since, d.seq);
    if (version === null) version = d.version;
    // 試合が保存されたら、まとめを読み直すボタンを出す（自動で読み直すと、ページ作りと画像の読み込みが次の試合の読み取りと重なるため）
    $("reloadbar").classList.toggle("on", d.version !== version);
  } catch (e) {}
  setTimeout(poll, 500);
}
poll();

async function endMatch(result) {
  if (!confirm((result || "勝敗不明") + "でこの試合を終えますか？")) return;
  const r = await fetch("/api/end", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({result})});
  if (!r.ok) alert("終えられませんでした: " + await r.text());
}

async function fixLast(result) {
  const d = window._last;
  if (!d || !d.last_saved) return;
  if (!confirm("直前の試合を「" + result + "」にしますか？")) return;
  await fixResult(d.last_saved + ".json", result);
}

async function fixResult(file, result) {
  const r = await fetch("/api/result", {method: "POST", headers: {"Content-Type": "application/json"},
                                        body: JSON.stringify({file, result})});
  if (!r.ok) alert("直せませんでした: " + await r.text());
}
</script>"""


def review_html(store, link):
    """「ブキの確認」タブ。直近の試合の8人のブキの読み取りを並べ、その場で直せるようにする。"""
    rows, names = weapons.review(store, 20)
    kl, _ = killcam.load_labels(store)
    names = sorted(set(names) | {n for n in kl.values() if n and killcam.kind_of(n) == "メイン"})
    esc = report.ESC
    out = ['<datalist id="wnames">' + "".join(f'<option value="{esc(n)}">' for n in names) + "</datalist>",
           '<p class="muted">直近20試合の読み取り結果です。黄色は候補どまり、赤は分からないブキ、青い枠は手で付けた名前です。'
           '違っていたら名前を直すと、その絵が次から見本として使われます（まだ出たことのないブキも、ここで名前を付ければ覚えます）。</p>']
    for m, row in rows:
        strip = link(os.path.join(store.root, "weapons", m["weapon_icons"]))
        me = None
        cells = []
        for k, w in enumerate(row):
            if w is None:
                cells.append('<div class="wcell"></div>')
                continue
            key, n, conf, side = w
            if side == "me":
                me = k
            cells.append(f'<div class="wcell{" me" if side == "me" else ""}">'
                         f'<div class="wimg" style="background-image:url(\'{strip}\');background-position:-{k * 60}px 0"></div>'
                         f'<input class="{conf}" list="wnames" value="{esc(n or "")}" placeholder="{"分からない" if conf == "none" else ""}" '
                         f'data-key="{esc(key)}"><div class="saved"></div></div>')
        res = m.get("result") or ""
        out.append(f'<div class="wmatch"><h2>{esc(m.get("played_at", "")[:16].replace("T", " "))}　{esc(m.get("rule") or "ルール不明")}　'
                   f'<span class="res-{esc(res)}">{esc(res)}</span></h2>'
                   f'<div class="wrow"><span class="side">味方</span>{"".join(cells[:4])}</div>'
                   f'<div class="wrow"><span class="side">相手</span>{"".join(cells[4:])}</div></div>')
    if not rows:
        out.append('<p class="muted">ブキのアイコンの記録がまだありません</p>')
    out.append(info_table(store, names))
    return "".join(out)


def info_table(store, names):
    """ブキの情報（種類・射程・サブ・スペシャル）の表。その場で直せる。"""
    from . import weapon_info as wi
    info = wi.Info(store)
    esc = report.ESC
    seen = sorted({wi.norm(n) for n in weapons.load_labels(store).values() if n and n != weapons.UNKNOWN} | set(names))
    subs = sorted({v[0] for v in wi.KITS.values()})
    sps = sorted({v[1] for v in wi.KITS.values()})
    h = ['<h2 style="font-size:15px;margin-top:24px">ブキの情報</h2>',
         '<p class="muted">編成でまとめた勝率に使います。射程はおおまかな目安です。空欄や違うところがあれば直してください（すぐ保存されます）。</p>',
         '<datalist id="wsubs">' + "".join(f'<option value="{esc(x)}">' for x in subs) + "</datalist>",
         '<datalist id="wsps">' + "".join(f'<option value="{esc(x)}">' for x in sps) + "</datalist>",
         '<table class="info"><tr><th>ブキ</th><th>種類</th><th>射程</th><th>サブ</th><th>スペシャル</th></tr>']

    def select(name, field, options, value):
        opts = '<option value=""></option>' + "".join(
            f'<option{" selected" if o == value else ""}>{esc(o)}</option>' for o in options)
        return f'<select data-info="{esc(name)}" data-field="{field}">{opts}</select>'
    for n in seen:
        d = info.get(n)
        empty = ' class="empty"' if not (d["サブ"] and d["スペシャル"] and d["種類"]) else ""
        h.append(f'<tr{empty}><td>{esc(n)}</td>'
                 f'<td>{select(n, "種類", wi.CLASSES, d["種類"])}</td><td>{select(n, "射程", wi.RANGES, d["射程"])}</td>'
                 f'<td><input list="wsubs" data-info="{esc(n)}" data-field="サブ" value="{esc(d["サブ"] or "")}"></td>'
                 f'<td><input list="wsps" data-info="{esc(n)}" data-field="スペシャル" value="{esc(d["スペシャル"] or "")}"></td></tr>')
    h.append("</table>")
    return "".join(h)


# ---------- サーバー ----------
def make_handler(store, get_video_dir, st, cmds, start_camera=None):
    root = pathlib.Path(store.root).resolve()
    cache = {"version": None, "html": None}

    def link(path):
        p = pathlib.Path(path).resolve()
        vd = get_video_dir()
        for prefix, base in (("/data/", root), ("/video/", pathlib.Path(vd).resolve() if vd else None)):
            if base is not None:
                try:
                    return prefix + urllib.parse.quote(p.relative_to(base).as_posix())
                except ValueError:
                    pass
        return p.as_uri()

    def page():
        with st.lock:
            info_p = os.path.join(store.root, "weapon_info.json")
            from . import statink
            v = (st.version, weapons.labels_mtime(store),  # 試合が増えたか、ブキの名前・情報や stat.ink の集計が変わったら作り直す
                 os.path.getmtime(info_p) if os.path.exists(info_p) else 0, statink.summary_mtime(store))
        if cache["version"] != v:
            cache["html"] = report.render(store, get_video_dir(), link=link,
                                          first_tabs=[("live", "試合中", LIVE_HTML),
                                                      ("last", "直前の試合", '<div id="lastbody"><p class="muted">読み込んでいます…</p></div>'),
                                                      ("wcheck", "ブキの確認", '<div id="wcheckbody"><p class="muted">読み込んでいます…</p></div>')],
                                          head=APP_CSS, script=APP_JS, lazy=True)
            cache["version"] = v
            cache["tabs"] = {}
        return cache["html"]

    def warm():
        """起動したら、試合が始まる前に集計のタブを先に作っておく（初めて開いたときに待たされないように）。"""
        time.sleep(3)
        for k, _, _ in report.tab_list(store, get_video_dir(), link, True):
            if st.state.get("in_match"):
                return  # 試合中は読み取りを優先する
            try:
                tab(k)
            except Exception:
                pass
    threading.Thread(target=warm, daemon=True).start()

    def tab(key):
        """集計のタブの中身。同じ版の間は作ったものを使い回す。"""
        page()  # 版が変わっていれば作り直しの合図を受ける
        if key not in cache.setdefault("tabs", {}):
            cache["tabs"][key] = report.tab_body(store, key, get_video_dir(), link, app=True)
        return cache["tabs"][key]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            try:
                if u.path == "/":
                    self._send(200, page().encode("utf-8"), "text/html; charset=utf-8")
                elif u.path == "/api/cameras":
                    import base64
                    import tempfile
                    import cv2
                    from .sources import list_cameras
                    d = tempfile.mkdtemp()
                    out = []
                    for i, w, h in list_cameras(snapshot_dir=d):
                        img = cv2.imread(os.path.join(d, f"camera_{i}.png"))
                        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (320, 180))) if img is not None else (False, None)
                        out.append({"index": i, "w": w, "h": h,
                                    "img": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""})
                    self._send(200, json.dumps(out).encode("utf-8"), "application/json")
                elif u.path == "/api/tab":
                    k = urllib.parse.parse_qs(u.query).get("k", [""])[0]
                    self._send(200, tab(k).encode("utf-8"), "text/html; charset=utf-8")
                elif u.path == "/api/wcheck":
                    try:
                        body = review_html(store, link)
                    except Exception as e:
                        body = f'<p class="muted">作れませんでした: {report.ESC(str(e))}</p>'
                    self._send(200, body.encode("utf-8"), "text/html; charset=utf-8")
                elif u.path == "/api/last":
                    from . import lastmatch
                    f = urllib.parse.parse_qs(u.query).get("file", [None])[0]
                    try:
                        body = lastmatch.html_for(store, link, get_video_dir(), f)
                    except Exception as e:
                        body = f'<p class="muted">作れませんでした: {report.ESC(str(e))}</p>'
                    self._send(200, body.encode("utf-8"), "text/html; charset=utf-8")
                elif u.path == "/api/live":
                    since = int(urllib.parse.parse_qs(u.query).get("since", ["0"])[0])
                    self._send(200, json.dumps(st.snapshot(since), ensure_ascii=False).encode("utf-8"),
                               "application/json; charset=utf-8")
                elif u.path.startswith("/data/"):
                    self._file(root, u.path[6:])
                elif u.path.startswith("/video/") and get_video_dir():
                    self._file(pathlib.Path(get_video_dir()).resolve(), u.path[7:])
                else:
                    self._send(404, b"not found", "text/plain")
            except (ConnectionError, BrokenPipeError):
                pass
            except Exception as e:  # 画面に理由を出す
                self._send(500, f"エラー: {e}".encode("utf-8"), "text/plain; charset=utf-8")

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if u.path == "/api/weapon" and ":" in str(body.get("key", "")):
                    weapons.set_label(store, os.path.basename(body["key"]), str(body.get("name", "")).strip())
                    with st.lock:
                        st._push(st.logs, f"ブキの名前を直しました: {body['key']} → {body.get('name') or '（消した）'}")
                    self._send(200, b"ok", "text/plain")
                elif u.path == "/api/weapon_info" and body.get("field") in ("種類", "射程", "サブ", "スペシャル"):
                    from . import weapon_info
                    weapon_info.set_override(store, str(body.get("name", "")), body["field"], str(body.get("value", "")).strip())
                    self._send(200, b"ok", "text/plain")
                elif u.path == "/api/stop_camera":
                    if start_camera is None or not hasattr(start_camera, "stop"):
                        raise ValueError("ここからは止められません")
                    start_camera.stop()
                    self._send(200, b"ok", "text/plain")
                elif u.path == "/api/start_camera":
                    err = start_camera(int(body.get("index", 0))) if start_camera else "ここからは始められません"
                    if err:
                        self._send(400, err.encode("utf-8"), "text/plain; charset=utf-8")
                    else:
                        self._send(200, b"ok", "text/plain")
                elif u.path == "/api/config":
                    from .storage import MODES
                    cfg = store.load_config()
                    if body.get("mode") in MODES:
                        cfg["mode"] = body["mode"]
                    if body.get("view") in MODES + ("すべて",):
                        cfg["view"] = body["view"]
                    store.save_config(cfg)
                    with st.lock:
                        st.mode, st.view = store.mode(), store.view() or "すべて"
                        if "view" in body:
                            st.version += 1  # まとめを作り直させる
                    self._send(200, b"ok", "text/plain")
                elif u.path == "/api/match_mode" and body.get("file"):
                    from .storage import MODES
                    if body.get("mode") not in MODES:
                        raise ValueError("種類が違います")
                    store.update(os.path.basename(str(body["file"])), {"mode": body["mode"]})
                    with st.lock:
                        st._push(st.logs, f"試合の種類を直しました: {body['file']} → {body['mode']}")
                    self._send(200, b"ok", "text/plain")
                elif u.path == "/api/clips" and body.get("file"):
                    from . import clips
                    kind = body.get("kind")
                    if kind not in clips.KINDS:
                        raise ValueError("種類が違います")
                    m = store.load(os.path.join(store.mdir, os.path.basename(str(body["file"]))))

                    def job():
                        def log(text):
                            with st.lock:
                                st._push(st.logs, text)
                        try:
                            if not clips.make(store, [m], kind, get_video_dir(), log):
                                log(f"{clips.KINDS[kind]}がありませんでした")
                        except Exception as e:
                            log(f"動画にまとめられませんでした: {e}")
                    threading.Thread(target=job, daemon=True).start()
                    self._send(200, b"ok", "text/plain")
                elif u.path == "/api/stage" and body.get("file"):
                    from .hud import save_stage_label
                    f, name = os.path.basename(str(body["file"])), str(body.get("name", "")).strip() or None
                    store.update(f, {"stage": name})
                    if name:
                        save_stage_label(store, os.path.splitext(f)[0], name)
                    with st.lock:
                        st._push(st.logs, f"ステージを直しました: {f} → {name or '（消した）'}")
                    self._send(200, b"ok", "text/plain")
                elif u.path == "/api/end":
                    if cmds is None:
                        self._send(400, "映像を読んでいません".encode("utf-8"), "text/plain; charset=utf-8")
                    else:
                        r = body.get("result")
                        cmds.put(("end", r if r in ("勝ち", "負け") else None))
                        self._send(200, b"ok", "text/plain")
                elif u.path == "/api/result" and body.get("result") in ("勝ち", "負け"):
                    store.set_result(body["file"], body["result"])
                    with st.lock:
                        st.version += 1  # 画面を読み直させる
                        st._push(st.logs, f"勝敗を手で直しました: {body['file']} → {body['result']}")
                    self._send(200, b"ok", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:
                self._send(500, f"{e}".encode("utf-8"), "text/plain; charset=utf-8")

        def _file(self, base, rel):
            p = (base / urllib.parse.unquote(rel)).resolve()
            if base not in p.parents or not p.is_file():  # 決めたフォルダの外は出さない
                self._send(404, b"not found", "text/plain")
                return
            size = p.stat().st_size
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            a, b = 0, size - 1
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):  # 動画の途中から再生できるように、一部だけ送る
                s, _, e = rng[6:].split(",")[0].partition("-")
                if s:
                    a = int(s)
                    b = int(e) if e else size - 1
                else:
                    a = max(size - int(e), 0)
                b = min(b, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(b - a + 1))
            self.end_headers()
            with open(p, "rb") as f:
                f.seek(a)
                left = b - a + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

    Handler.page = staticmethod(page)
    return Handler


def serve(store, get_video_dir, st, port, cmds=None, start_camera=None):
    handler = make_handler(store, get_video_dir, st, cmds, start_camera)
    for p in range(port, port + 20):  # 使われていたら次の番号
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("サーバーを立てられませんでした")
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    threading.Thread(target=handler.page, daemon=True).start()  # まとめは作るのに数秒かかるので先に作っておく
    return f"http://127.0.0.1:{httpd.server_address[1]}/"


# ---------- 窓 ----------
def open_in_browser(url):
    webbrowser.open(url)
    print(f"ブラウザで開きました: {url}")
    print("終わるときはこの窓で Ctrl+C を押してください")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def open_window(url):
    try:
        import webview
    except Exception:  # 入っていない・入れられなかった
        webview = None
    if webview is None:
        print("（pip install pywebview を入れると、専用の窓で開き、小さくすると常に手前に出せます）")
        open_in_browser(url)
        return
    holder = {}

    class Api:
        def set_compact(self, on):
            w = holder["w"]
            w.resize(*(COMPACT_SIZE if on else FULL_SIZE))
            try:
                w.on_top = bool(on)  # 小さいときはゲームの画面の上に出しておく
            except Exception:
                pass

    try:
        holder["w"] = webview.create_window("splat-compass", url, width=FULL_SIZE[0], height=FULL_SIZE[1], js_api=Api())
        webview.start()
    except Exception as e:
        print(f"専用の窓を開けませんでした（{e}）。ブラウザで開きます")
        open_in_browser(url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, help="カメラ番号（python -m splat_compass.live --list で調べる）")
    ap.add_argument("--video", help="録画を実際の速さで流して試す")
    ap.add_argument("--fast", action="store_true", help="--video を待たずに最速で流す")
    ap.add_argument("--data", help="記録の保存先")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    store = Store(a.data)
    st = State(os.path.join(store.root, "logs"))
    files = sorted(f for f in os.listdir(store.mdir) if f.endswith(".json"))
    st.last_saved = files[-1][:-5] if files else None  # 起動前に保存した一番新しい試合も直せるように
    st.mode, st.view = store.mode(), store.view() or "すべて"
    q = queue.Queue()
    threading.Thread(target=st.consume, args=(q,), daemon=True).start()
    stop = threading.Event()  # アプリを閉じるとき
    cmds = queue.Queue()
    lock = threading.Lock()
    run_stop = {"ev": threading.Event(), "th": None}

    def begin(frames, source):
        ev = threading.Event()
        run_stop["ev"] = ev

        class Both:  # アプリを閉じたときか、カメラを選び直すときに止める
            def is_set(self):
                return stop.is_set() or ev.is_set()

        def run():
            try:
                live.worker(frames, store, source, q, Both(), cmds)
            finally:
                close = getattr(frames, "close", None)
                if close:
                    close()
                q.put(("end", None))
        st.reading = True
        th = threading.Thread(target=run, daemon=True)
        run_stop["th"] = th
        th.start()

    def stop_camera():
        """今のカメラで読むのをやめる（カメラを選び直すため）。"""
        run_stop["ev"].set()
        th = run_stop["th"]
        if th is not None:
            th.join(timeout=5)
        return None

    def start_camera(index):
        """画面で選んだカメラで読み始める。選んだ番号は次から自動で使う。"""
        with lock:
            if st.reading:
                return "もう読んでいます"
            try:
                frames = camera_frames(index, live.FPS)
                first = next(frames)  # 開けるかここで確かめる
            except Exception as e:
                return f"カメラ {index} を開けませんでした: {e}"

            def again():
                yield first
                yield from frames
            cfg = store.load_config()
            cfg["camera"] = index
            store.save_config(cfg)
            begin(again(), "camera")
            return None

    if a.video:
        begin(video_frames(a.video, live.FPS, realtime=not a.fast), os.path.basename(a.video))
    else:
        cam = a.camera if a.camera is not None else store.load_config().get("camera")
        st.can_start = True
        if cam is not None:
            err = start_camera(int(cam))
            if err:
                st._push(st.logs, err + "。カメラを選び直してください")
    start_camera.stop = stop_camera
    url = serve(store, lambda: store.load_config().get("video_dir"), st, a.port, cmds, start_camera)
    open_window(url)
    stop.set()


if __name__ == "__main__":
    main()
