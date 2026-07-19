"""Build the AgentFuse observability dashboard from run traces.

Reads every ``runs/*.jsonl`` trace and emits self-contained, dependency-free
artifacts with the data embedded inline (no server, no CDN, no build step):

  * ``dashboard/index.html``    — full standalone page; open directly in a browser
  * ``dashboard/artifact.html`` — body-only fragment for a Claude Artifact
  * ``docs/index.html``         — GitHub Pages copy

Design language: a technical-brutalist *engineering schematic*. High-contrast
editorial serif (Fraunces) for display, raw monospace (Space Mono) for data;
hard edges, hairline rules, corner registration ticks, section index marks, a
titleblock on every panel, an annotated plotted chart, and one signal-orange
accent — the fuse.

Usage:
    python dashboard/build_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
OUT_FULL = ROOT / "dashboard" / "index.html"
OUT_ARTIFACT = ROOT / "dashboard" / "artifact.html"
OUT_PAGES = ROOT / "docs" / "index.html"

CATALOG = {
    "loop_trap.jsonl": ("Infinite Tool Loop", "Agent repeats a doomed tool call; the breaker forces a new path.", "loop"),
    "drift.jsonl": ("Goal Drift", "Agent wanders off-objective; the breaker re-anchors it.", "drift"),
    "escalation.jsonl": ("Human Escalation", "An unrecoverable failure; the breaker hands control to a human.", "escalate"),
    "real_agentkit.jsonl": ("Real AgentKit Run", "Live openai-agents Runner + real hooks — a real run that self-healed.", "real"),
    "real_gpt.jsonl": ("Real GPT Run", "Live GPT model driving a real agent; supervised and self-healed.", "real"),
}

TITLE = "AgentFuse — Observability Dashboard"

STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Space+Mono:wght@400;700&display=swap');
:root{
  --bg:#0B0A09; --ink:#F1EEE6; --dim:#98938A; --faint:#6E6A61;
  --rule:#2B2925; --rule2:#403C35;
  --fuse:#FF5D1F; --live:#54DE8B; --stop:#FF4D4D; --cool:#7FB4D6;
  --serif:'Fraunces',Georgia,'Times New Roman',serif;
  --mono:'Space Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{color-scheme:dark}
body{margin:0;color:var(--ink);font-family:var(--mono);font-size:13px;-webkit-font-smoothing:antialiased;
  background:
    repeating-linear-gradient(90deg,transparent 0 63px,rgba(255,255,255,.015) 63px 64px),
    var(--bg);}
::selection{background:var(--fuse);color:#0B0A09}
a{color:var(--fuse)}

/* ---------- masthead ---------- */
.mast{display:flex;align-items:center;gap:16px;padding:15px 26px 14px;
  border-bottom:2px solid var(--ink);position:relative}
.mast::after{content:"";position:absolute;left:0;right:0;bottom:-5px;height:1px;background:var(--rule2)}
.mb{display:flex;align-items:center;gap:9px}
.mb svg{color:var(--fuse);filter:drop-shadow(0 0 5px rgba(255,93,31,.55))}
.mb .wm{font-weight:700;font-size:16px;letter-spacing:1px}
.spec{margin-left:auto;font-size:10.5px;letter-spacing:1px;color:var(--faint);display:flex;gap:14px;flex-wrap:wrap}
.spec b{color:var(--dim);font-weight:400}
.spec .on{color:var(--live)}

/* ---------- shell ---------- */
.app{display:grid;grid-template-columns:252px 1fr;min-height:calc(100vh - 58px)}
.rail{border-right:1px solid var(--rule);padding:16px 0 20px}
.rail-hd{font-size:10px;letter-spacing:2px;color:var(--faint);padding:2px 18px 12px;
  border-bottom:1px solid var(--rule)}
.run{display:block;width:100%;text-align:left;font:inherit;color:inherit;cursor:pointer;background:none;
  border:0;border-bottom:1px solid var(--rule);padding:13px 18px 13px 26px;position:relative}
.run::before{content:attr(data-ix);position:absolute;left:6px;top:13px;font-size:9.5px;color:var(--faint)}
.run:hover .nm{text-decoration:underline;text-underline-offset:3px}
.run:focus-visible{outline:2px solid var(--fuse);outline-offset:-2px}
.run.on{background:linear-gradient(90deg,rgba(255,93,31,.07),transparent 80%)}
.run.on::after{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--fuse)}
.run .r1{display:flex;align-items:baseline;gap:8px}
.run .nm{font-family:var(--serif);font-size:16px;font-weight:600;letter-spacing:-.2px}
.run.on .nm{color:var(--fuse)}
.run .tg{margin-left:auto;font-size:9px;letter-spacing:1px;color:var(--faint)}
.run .tg.esc{color:var(--stop)} .run .tg.ok{color:var(--live)}
.run .ds{color:var(--dim);font-size:11px;line-height:1.4;margin:5px 0 7px}
.run .mt{font-size:10px;color:var(--faint)}
.run .mt b{color:var(--dim);font-weight:700}

/* ---------- main ---------- */
.main{padding:22px 30px 40px;max-width:1180px}
.hero{display:grid;grid-template-columns:1.7fr 1fr;gap:30px;align-items:end;
  padding-bottom:20px;border-bottom:2px solid var(--rule2)}
.eyebrow{font-size:10.5px;letter-spacing:2px;color:var(--fuse)}
.htitle{font-family:var(--serif);font-weight:600;font-size:52px;line-height:.96;letter-spacing:-1.4px;
  margin:12px 0 12px;text-wrap:balance}
.hobj{color:var(--dim);font-size:12.5px;line-height:1.55;max-width:560px}
.hmeta{font-size:10.5px;color:var(--faint);margin-top:10px;letter-spacing:.3px}
.oc{text-align:right;border-top:2px solid var(--rule2);padding-top:12px}
.oc .w{font-family:var(--serif);font-weight:600;font-size:33px;line-height:1;letter-spacing:-.8px}
.oc .w.heal{color:var(--live)} .oc .w.esc{color:var(--stop)} .oc .w.plain{color:var(--ink)}
.oc .s{font-size:10.5px;color:var(--faint);margin-top:9px;letter-spacing:1px}

/* corner-tick plate */
.plate{border:1px solid var(--rule);position:relative;padding:15px 17px}
.plate::before,.plate::after{content:"";position:absolute;width:9px;height:9px;border:2px solid var(--rule2)}
.plate::before{top:-1px;left:-1px;border-width:2px 0 0 2px}
.plate::after{bottom:-1px;right:-1px;border-width:0 2px 2px 0}
.tb{display:flex;align-items:baseline;gap:10px;border-bottom:1px solid var(--rule);
  padding-bottom:9px;margin-bottom:14px}
.tb .ix{color:var(--fuse);font-weight:700;font-size:11px}
.tb .lb{font-size:10.5px;letter-spacing:2px;color:var(--dim)}
.tb .rt{margin-left:auto;font-size:10px;color:var(--faint);letter-spacing:.5px}

/* readout spec-table */
.readout{display:grid;grid-template-columns:1.5fr 1.6fr 1fr .8fr 1.1fr 1fr;
  border:1px solid var(--rule2);margin:22px 0 20px}
.cell{padding:13px 15px;border-right:1px solid var(--rule)}
.cell:last-child{border-right:0}
.cell .k{font-size:9.5px;letter-spacing:1.5px;color:var(--faint);display:flex;align-items:center;gap:6px}
.cell .v{font-size:27px;font-weight:700;margin-top:9px;font-variant-numeric:tabular-nums;letter-spacing:-1px;line-height:1}
.cell .u{font-size:9.5px;color:var(--faint);margin-top:4px;letter-spacing:1px}
.cell.hi{background:rgba(255,93,31,.05)} .cell.hi .v{color:var(--fuse)}
.cell.hg{background:rgba(84,222,139,.05)} .cell.hg .v{color:var(--live)}

.grid2{display:grid;grid-template-columns:1.55fr 1fr;gap:16px;margin-bottom:16px}
.chart{width:100%;height:172px;display:block}
.chart text{font-family:var(--mono);fill:var(--faint)}
.chart .plot{stroke:var(--fuse);stroke-width:1.75;fill:none;stroke-dasharray:1600;stroke-dashoffset:1600;animation:draw 1.15s ease forwards}
@keyframes draw{to{stroke-dashoffset:0}}
.dets{display:flex;flex-direction:column;gap:0}
.det{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--faint);
  padding:8px 0;border-bottom:1px solid var(--rule)}
.det:last-child{border-bottom:0}
.det .bx{width:9px;height:9px;border:1px solid var(--rule2);flex:none}
.det .st{margin-left:auto;font-size:9.5px;letter-spacing:1px}
.det.hot{color:var(--fuse)} .det.hot .bx{background:var(--fuse);border-color:var(--fuse);box-shadow:0 0 7px var(--fuse)}
.det.hot .st{color:var(--fuse)}
.route{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px;margin-top:2px}
.route .n{border:1px solid var(--rule2);padding:4px 9px}
.route .w{color:var(--faint)}

/* timeline ledger */
.tl{margin-top:2px}
.ln{display:grid;grid-template-columns:74px 96px 1fr;gap:0;border-bottom:1px solid var(--rule);
  padding:8px 0;align-items:baseline}
.ln .g{color:var(--faint);font-size:10.5px;border-right:1px solid var(--rule);padding-right:12px;text-align:right}
.ln .t{color:var(--dim);font-size:10.5px;padding-left:12px}
.ln .d{font-size:12.5px;line-height:1.45;padding-left:6px}
.notice{margin:2px 0;padding:12px 15px;border:1px solid var(--rule2);position:relative}
.notice.trip{border-left:3px solid var(--fuse);background:linear-gradient(90deg,rgba(255,93,31,.06),transparent 70%)}
.notice.heal{border-left:3px solid var(--live);background:linear-gradient(90deg,rgba(84,222,139,.055),transparent 70%)}
.notice .h{display:flex;align-items:center;gap:9px;font-size:12px;font-weight:700;letter-spacing:.5px}
.notice.trip .h{color:var(--fuse)} .notice.heal .h{color:var(--live)}
.notice .h svg{filter:drop-shadow(0 0 4px currentColor)}
.tag{margin-left:auto;font-size:9.5px;font-weight:700;letter-spacing:1px;padding:2px 7px;border:1px solid currentColor}
.notice .rz{font-size:12px;line-height:1.5;margin-top:8px;color:var(--ink)}
.notice .rz b{color:var(--dim)}
.notice .instr{font-size:11.5px;line-height:1.55;margin-top:9px;color:var(--dim);
  border-top:1px solid var(--rule);padding-top:9px;white-space:pre-wrap}
.foot{color:var(--faint);font-size:10.5px;margin-top:26px;border-top:1px solid var(--rule);
  padding-top:14px;letter-spacing:.5px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
@media(max-width:940px){.app{grid-template-columns:1fr}.hero{grid-template-columns:1fr}.oc{text-align:left}
  .readout{grid-template-columns:repeat(3,1fr)}.grid2{grid-template-columns:1fr}}
</style>"""

BOLT = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M13 2 4 14h6l-1 8 9-12h-6z"/></svg>'
BOLT_SM = '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M13 2 4 14h6l-1 8 9-12h-6z"/></svg>'
STEER = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v5h-5"/></svg>'

BODY = f"""<header class="mast">
  <div class="mb">{BOLT}<span class="wm">AGENTFUSE</span></div>
  <div class="spec"><span>SUPERVISORY CIRCUIT BREAKER</span><span><b>REV</b> 0.1</span>
    <span><b>BUS</b> <span id="nrun">0</span> RUNS</span><span class="on">&#9679; ONLINE</span></div>
</header>
<div class="app">
  <aside class="rail">
    <div class="rail-hd">INDEX &mdash; SUPERVISED RUNS</div>
    <div id="runlist"></div>
  </aside>
  <main class="main" id="main"></main>
</div>"""

SCRIPT = """<script id="data" type="application/json">__DATA__</script>
<script>
const BOLT='__BOLT__', STEER='__STEER__';
const RUNS = JSON.parse(document.getElementById('data').textContent);
let active = 0;
const sumOf=r=>r.records.find(x=>x.kind==='summary')||{};
const metaOf=r=>r.records.find(x=>x.kind==='meta')||{};
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
document.getElementById('nrun').textContent = RUNS.length;

function outcome(s){
  if(s.status==='complete') return s.recoveries>0?{c:'heal',w:'SELF-HEALED'}:{c:'plain',w:'COMPLETED'};
  if(s.status==='escalated') return {c:'esc',w:'ESCALATED'};
  return {c:'plain',w:(s.status||'—').toUpperCase()};
}

function renderList(){
  document.getElementById('runlist').innerHTML = RUNS.map((r,i)=>{
    const s=sumOf(r), st=s.status||'x';
    const tg = st==='escalated'?'esc':(st==='complete'?'ok':'');
    return `<button class="run ${i===active?'on':''}" data-ix="${String(i+1).padStart(2,'0')}" onclick="select(${i})">
      <div class="r1"><span class="nm">${esc(r.title)}</span><span class="tg ${tg}">${st.toUpperCase()}</span></div>
      <div class="ds">${esc(r.subtitle)}</div>
      <div class="mt"><b>${s.trips??0}</b> TRIP · <b>${s.recoveries??0}</b> HEAL · <b>${(s.total_tokens??0).toLocaleString()}</b> TOK</div>
    </button>`;
  }).join('');
}

function chart(records){
  let cum=0, pts=[], tripAt=null, ev=0;
  for(const e of records){
    if(e.kind==='event'){ cum+=(e.tokens_in||0)+(e.tokens_out||0); pts.push(cum); ev++; }
    else if(e.kind==='trip' && tripAt===null) tripAt=Math.max(0,ev-1);
  }
  if(pts.length<2) return '<div style="color:var(--faint)">no token data</div>';
  const W=600,H=172,L=48,R=14,T=16,B=26, maxY=Math.max(...pts)||1;
  const X=i=>L+(i/(pts.length-1))*(W-L-R), Y=v=>H-B-(v/maxY)*(H-T-B);
  const line=pts.map((v,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  const area=`${line} L${X(pts.length-1).toFixed(1)},${H-B} L${L},${H-B} Z`;
  // gridlines + y labels (0, mid, max)
  let grid='';
  [0,.5,1].forEach(f=>{ const y=(H-B)-(f)*(H-T-B); const val=Math.round(maxY*f);
    grid+=`<line x1="${L}" y1="${y.toFixed(1)}" x2="${W-R}" y2="${y.toFixed(1)}" stroke="#2B2925" stroke-width="1"/>`;
    grid+=`<text x="${L-6}" y="${(y+3).toFixed(1)}" font-size="9" text-anchor="end">${val.toLocaleString()}</text>`; });
  // trip annotation
  let ann='';
  if(tripAt!==null){ const tx=X(tripAt);
    ann=`<line x1="${tx.toFixed(1)}" y1="${T}" x2="${tx.toFixed(1)}" y2="${H-B}" stroke="#FF5D1F" stroke-width="1" stroke-dasharray="3 3"/>
      <circle cx="${tx.toFixed(1)}" cy="${Y(pts[tripAt]).toFixed(1)}" r="3.5" fill="#FF5D1F"/>
      <text x="${(tx+5).toFixed(1)}" y="${T+9}" font-size="9" fill="#FF5D1F">TRIP</text>`; }
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
    <defs><linearGradient id="fl" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#FF5D1F" stop-opacity=".22"/><stop offset="1" stop-color="#FF5D1F" stop-opacity="0"/></linearGradient></defs>
    ${grid}
    <line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="#403C35" stroke-width="1"/>
    <path d="${area}" fill="url(#fl)"/><path class="plot" d="${line}"/>
    ${pts.map((v,i)=>`<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="1.8" fill="#FF5D1F"/>`).join('')}
    ${ann}
    <text x="${L}" y="${H-8}" font-size="9">STEP 1</text>
    <text x="${W-R}" y="${H-8}" font-size="9" text-anchor="end">STEP ${pts.length}</text>
  </svg>`;
}

function detectors(r){
  const hot=new Set(r.records.filter(x=>x.kind==='trip').map(x=>x.detector));
  return '<div class="dets">'+['loop','drift','progress','spend'].map(d=>
    `<div class="det ${hot.has(d)?'hot':''}"><span class="bx"></span>${d.toUpperCase()}<span class="st">${hot.has(d)?'TRIPPED':'nominal'}</span></div>`).join('')+'</div>';
}
function routeView(s){
  const p=(s.route||'').split('->').map(x=>x.trim()).filter(Boolean);
  if(!p.length) return '<div style="color:var(--faint)">—</div>';
  return '<div class="route">'+p.map((n,i)=>`<span class="n">${esc(n)}</span>${i<p.length-1?'<span class="w">&rarr;</span>':''}`).join('')+'</div>';
}
function tb(ix,lb,rt){return `<div class="tb"><span class="ix">§${ix}</span><span class="lb">${lb}</span><span class="rt">${rt||''}</span></div>`;}

function timeline(r){
  let h='<div class="tl">';
  for(const rec of r.records){
    if(rec.kind==='meta'||rec.kind==='summary') continue;
    if(rec.kind==='event'){
      const t=rec.type; let d='';
      if(t==='tool_call') d=`<span style="color:var(--cool)">${esc(rec.tool_name)}</span><span style="color:var(--faint)">(${esc(JSON.stringify(rec.tool_args||{}))})</span>`;
      else if(t==='tool_result') d=`<span style="color:var(--dim)">${esc(rec.tool_name||'')}</span> &rarr; ${esc((rec.text||'').slice(0,78))}`;
      else if(t==='llm_call') d=rec.text?esc(rec.text.slice(0,108)):'<span style="color:var(--faint)">model turn</span>';
      else if(t==='state_update') d='<span style="color:var(--live)">state advanced</span>';
      else if(t==='resume') d=`<span style="color:var(--live)">${esc(rec.text||'resumed')}</span>`;
      else if(t==='complete') d=`<span style="color:var(--live)">${esc(rec.text||'objective complete')}</span>`;
      else d=esc(rec.text||rec.node||'');
      h+=`<div class="ln"><span class="g">STEP ${rec.step}</span><span class="t">${t}</span><span class="d">${d}</span></div>`;
    }
    if(rec.kind==='trip')
      h+=`<div class="notice trip"><div class="h">${BOLT} BREAKER TRIP — ${esc(rec.detector.toUpperCase())} · ${esc(rec.severity)}</div><div class="rz">${esc(rec.reason)}</div></div>`;
    if(rec.kind==='recovery')
      h+=`<div class="notice heal"><div class="h">${STEER} STEERING RECOVERY<span class="tag">${esc(rec.action).toUpperCase()}</span></div><div class="rz"><b>WHY</b> ${esc(rec.rationale)} <span style="color:var(--faint)">· conf ${rec.confidence} · ${esc(rec.backend)}</span></div><div class="instr">${esc(rec.instruction)}</div></div>`;
  }
  return h+'</div>';
}

function select(i){active=i;render();}
function render(){
  renderList();
  const r=RUNS[active], s=sumOf(r), m=metaOf(r), cfg=m.config||{}, oc=outcome(s);
  document.getElementById('main').innerHTML=`
    <div class="hero">
      <div>
        <div class="eyebrow">RUN ${String(active+1).padStart(2,'0')} / ${String(RUNS.length).padStart(2,'0')} &nbsp;·&nbsp; ${esc((r.kind||'run').toUpperCase())} DETECTOR</div>
        <div class="htitle">${esc(r.title)}</div>
        <div class="hobj">${esc(m.original_goal||r.subtitle)}</div>
        <div class="hmeta">BACKEND ${esc((m.recovery_backend||'—').toUpperCase())} &nbsp;/&nbsp; LOOP&ge;${cfg.loop_threshold??'—'} &nbsp;/&nbsp; DRIFT&lt;${cfg.drift_threshold??'—'} &nbsp;/&nbsp; ${esc(r.file)}</div>
      </div>
      <div class="oc"><div class="w ${oc.c}">${oc.w}</div><div class="s">${s.trips??0} TRIP DETECTED · ${s.recoveries??0} RECOVERED</div></div>
    </div>
    <div class="readout">
      <div class="cell hi"><div class="k">${BOLT} TRIPS</div><div class="v">${s.trips??0}</div><div class="u">breaker fires</div></div>
      <div class="cell hg"><div class="k">${STEER} RECOVERIES</div><div class="v">${s.recoveries??0}</div><div class="u">steer + resume</div></div>
      <div class="cell"><div class="k">STATUS</div><div class="v" style="font-size:16px;padding-top:7px">${esc(s.status||'—')}</div></div>
      <div class="cell"><div class="k">STEPS</div><div class="v">${s.steps??'—'}</div><div class="u">actions</div></div>
      <div class="cell"><div class="k">TOKENS</div><div class="v">${(s.total_tokens??0).toLocaleString()}</div><div class="u">consumed</div></div>
      <div class="cell"><div class="k">COST</div><div class="v">$${s.total_cost_usd??0}</div><div class="u">est. usd</div></div>
    </div>
    <div class="grid2">
      <div class="plate">${tb('01','CUMULATIVE TOKEN SPEND','TOKENS × STEP')}${chart(r.records)}</div>
      <div class="plate">${tb('02','DETECTORS','4-CHANNEL')}${detectors(r)}
        ${tb('03','GRAPH ROUTE','NODE PATH').replace('margin-bottom:14px','')}${routeView(s)}</div>
    </div>
    <div class="plate">${tb('04','EXECUTION TIMELINE',(s.steps??0)+' STEPS')}${timeline(r)}</div>
    <div class="foot"><span>AGENTFUSE · LOGICAL CIRCUIT BREAKER FOR LONG-RANGE AGENTS</span><span>GENERATED FROM JSONL TRACES · SELF-CONTAINED</span></div>`;
}
render();
</script>""".replace("__BOLT__", BOLT_SM).replace("__STEER__", STEER)


def load_runs() -> list[dict]:
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for path in sorted(RUNS_DIR.glob("*.jsonl")):
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not records:
            continue
        title, subtitle, kind = CATALOG.get(
            path.name, (path.stem.replace("_", " ").title(), "Supervised agent run.", "run"))
        runs.append({"file": path.name, "title": title, "subtitle": subtitle, "kind": kind, "records": records})
    return runs


def main() -> None:
    runs = load_runs()
    if not runs:
        print("No traces in runs/. Run an example first, e.g. python examples/demo_loop_trap.py")
        return
    data = json.dumps(runs).replace("</", "<\\/")
    script = SCRIPT.replace("__DATA__", data)
    inner = f"<title>{TITLE}</title>\n{STYLE}\n{BODY}\n{script}\n"
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\"/>\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n"
            f"<title>{TITLE}</title>\n{STYLE}\n</head>\n<body>\n{BODY}\n{script}\n</body>\n</html>\n")
    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    OUT_FULL.write_text(full, encoding="utf-8")
    OUT_ARTIFACT.write_text(inner, encoding="utf-8")
    OUT_PAGES.parent.mkdir(parents=True, exist_ok=True)
    OUT_PAGES.write_text(full, encoding="utf-8")
    total = sum(len(r["records"]) for r in runs)
    print(f"Standalone : {OUT_FULL}")
    print(f"Artifact   : {OUT_ARTIFACT}")
    print(f"Pages      : {OUT_PAGES}")
    print(f"  runs: {len(runs)} · records: {total}")


if __name__ == "__main__":
    main()
