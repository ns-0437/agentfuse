"""Build the AgentFuse observability dashboard from run traces.

Reads every ``runs/*.jsonl`` trace and emits self-contained, dependency-free
artifacts with the data embedded inline (no server, no CDN, no build step):

  * ``dashboard/index.html``    — full standalone page; open directly in a browser
  * ``dashboard/artifact.html`` — body-only fragment for a Claude Artifact
  * ``docs/index.html``         — GitHub Pages copy

Design language: an "instrument panel" — engineered IBM Plex type, hardware
status LEDs, an amber live-wire accent (the fuse), and a hero outcome readout.

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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root{
  --bg:#0A0D12; --surf:#111621; --surf2:#161C29; --surf3:#1B2331;
  --hair:#232B3A; --hair2:#2E3849;
  --text:#EAEEF5; --dim:#9AA6B7; --faint:#606C7E;
  --amber:#FFB020; --amber-hi:#FFC658; --teal:#22C48E; --cyan:#4CB8FF; --red:#FF5C77; --violet:#B69CFF;
  --sans:'IBM Plex Sans',system-ui,-apple-system,Segoe UI,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{color-scheme:dark}
body{margin:0;color:var(--text);font-family:var(--sans);-webkit-font-smoothing:antialiased;
  background:
    radial-gradient(900px 500px at 88% -8%, rgba(255,176,32,.06), transparent 60%),
    radial-gradient(700px 500px at -5% 108%, rgba(34,196,142,.05), transparent 55%),
    linear-gradient(0deg, rgba(255,255,255,.014) 1px, transparent 1px) 0 0/100% 44px,
    linear-gradient(90deg, rgba(255,255,255,.014) 1px, transparent 1px) 0 0/44px 100%,
    var(--bg);}
.app{display:grid;grid-template-columns:296px 1fr;min-height:100vh}

/* ---- rail ---- */
.rail{border-right:1px solid var(--hair);background:linear-gradient(180deg,#0C1017,#0A0D12);
  padding:22px 16px 18px;display:flex;flex-direction:column;gap:16px}
.brand{display:flex;align-items:center;gap:10px}
.brand svg{color:var(--amber);filter:drop-shadow(0 0 6px rgba(255,176,32,.5))}
.brand .wm{font-family:var(--mono);font-weight:600;font-size:17px;letter-spacing:.2px}
.brand .tag{font-family:var(--mono);font-size:9px;color:var(--faint);letter-spacing:2px;margin-left:auto;
  border:1px solid var(--hair2);border-radius:4px;padding:2px 6px}
.status{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;color:var(--dim);
  letter-spacing:.5px}
.led{width:8px;height:8px;border-radius:50%;background:currentColor;box-shadow:0 0 8px currentColor;flex:none}
.led.live{color:var(--teal);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.rail-label{font-family:var(--mono);font-size:10px;letter-spacing:2px;color:var(--faint);
  padding-top:2px;border-top:1px solid var(--hair)}
.runlist{display:flex;flex-direction:column;gap:9px;overflow:auto}
.run{position:relative;text-align:left;width:100%;font:inherit;color:inherit;cursor:pointer;
  background:var(--surf);border:1px solid var(--hair);border-radius:11px;padding:12px 13px 12px 15px;
  transition:transform .12s ease,border-color .12s ease,background .12s ease}
.run:hover{border-color:var(--hair2);transform:translateY(-1px)}
.run:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
.run::before{content:"";position:absolute;left:0;top:12px;bottom:12px;width:3px;border-radius:3px;background:transparent}
.run.active{background:var(--surf2);border-color:rgba(255,176,32,.5)}
.run.active::before{background:var(--amber);box-shadow:0 0 10px rgba(255,176,32,.6)}
.run .top{display:flex;align-items:center;gap:8px}
.run .nm{font-weight:600;font-size:13.5px}
.run .st{margin-left:auto;font-family:var(--mono);font-size:9.5px;letter-spacing:.5px;
  text-transform:uppercase;padding:2px 7px;border-radius:20px;border:1px solid transparent}
.st-complete{color:var(--teal);border-color:rgba(34,196,142,.35);background:rgba(34,196,142,.08)}
.st-escalated{color:var(--red);border-color:rgba(255,92,119,.35);background:rgba(255,92,119,.08)}
.st-incomplete,.st-max_turns{color:var(--amber);border-color:rgba(255,176,32,.35);background:rgba(255,176,32,.08)}
.run .ds{color:var(--dim);font-size:11.5px;line-height:1.35;margin:6px 0 8px}
.run .mt{display:flex;gap:12px;font-family:var(--mono);font-size:10.5px;color:var(--faint)}
.run .mt b{color:var(--dim);font-weight:600}
.rail-foot{margin-top:auto;font-family:var(--mono);font-size:10px;color:var(--faint);
  display:flex;align-items:center;gap:7px;border-top:1px solid var(--hair);padding-top:12px}

/* ---- main ---- */
.main{padding:26px 30px 34px;overflow:auto}
.hero{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;
  padding-bottom:18px;border-bottom:1px solid var(--hair)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:2px;color:var(--amber);
  display:flex;align-items:center;gap:8px}
.eyebrow .node{width:6px;height:6px;border-radius:50%;background:var(--amber);box-shadow:0 0 8px var(--amber)}
.h-title{font-size:27px;font-weight:700;letter-spacing:-.3px;margin:11px 0 8px;text-wrap:balance}
.h-obj{color:var(--dim);font-size:13.5px;line-height:1.5;max-width:640px}
.h-meta{font-family:var(--mono);font-size:11px;color:var(--faint);margin-top:9px}
.outcome{flex:none;text-align:right;min-width:190px}
.outcome .big{font-family:var(--mono);font-size:22px;font-weight:600;letter-spacing:-.3px;
  display:flex;align-items:center;gap:9px;justify-content:flex-end}
.outcome .sub{font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:7px}
.oc-heal{color:var(--teal)} .oc-esc{color:var(--red)} .oc-plain{color:var(--dim)}

.readouts{display:grid;grid-template-columns:1.3fr 1.3fr repeat(4,1fr);gap:1px;margin:20px 0;
  background:var(--hair);border:1px solid var(--hair);border-radius:12px;overflow:hidden}
.ro{background:var(--surf);padding:14px 15px}
.ro .k{font-family:var(--mono);font-size:10px;letter-spacing:1.2px;color:var(--faint);display:flex;align-items:center;gap:6px}
.ro .v{font-family:var(--mono);font-size:23px;font-weight:600;margin-top:7px;font-variant-numeric:tabular-nums;letter-spacing:-.5px}
.ro.key{background:var(--surf2)}
.ro.trip .v{color:var(--amber)} .ro.heal .v{color:var(--teal)}

.grid2{display:grid;grid-template-columns:1.5fr 1fr;gap:16px;margin-bottom:16px}
.panel{background:var(--surf);border:1px solid var(--hair);border-radius:13px;padding:16px 18px}
.p-hd{font-family:var(--mono);font-size:10.5px;letter-spacing:1.6px;color:var(--dim);text-transform:uppercase;
  display:flex;align-items:center;gap:8px;margin-bottom:14px}
.p-hd::before{content:"";width:5px;height:5px;border-radius:50%;background:var(--amber);box-shadow:0 0 7px var(--amber)}
.spark{width:100%;height:132px;display:block}
.spark path.line{stroke-dasharray:1400;stroke-dashoffset:1400;animation:draw 1.2s ease forwards}
@keyframes draw{to{stroke-dashoffset:0}}
.peak{font-family:var(--mono);font-size:11.5px;color:var(--dim);margin-top:6px}
.peak b{color:var(--amber-hi)}
.dets{display:flex;flex-wrap:wrap;gap:8px}
.det{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:11.5px;color:var(--faint);
  border:1px solid var(--hair);border-radius:8px;padding:6px 10px}
.det .led{width:7px;height:7px;color:#33405a;box-shadow:none}
.det.hot{color:var(--amber);border-color:rgba(255,176,32,.45);background:rgba(255,176,32,.07)}
.det.hot .led{color:var(--amber)}
.route{display:flex;flex-wrap:wrap;gap:7px;align-items:center;font-family:var(--mono);font-size:12px}
.route .n{background:var(--surf3);border:1px solid var(--hair2);border-radius:7px;padding:5px 10px}
.route .w{color:var(--faint)}

.tl{position:relative;margin-top:4px;padding-left:26px}
.tl::before{content:"";position:absolute;left:8px;top:6px;bottom:6px;width:2px;
  background:linear-gradient(180deg,var(--hair),transparent)}
.ev{position:relative;margin-bottom:13px}
.mk{position:absolute;left:-22px;top:3px;width:12px;height:12px;border-radius:50%;
  background:var(--bg);border:2px solid var(--faint)}
.mk.tool{border-color:var(--violet)} .mk.llm{border-color:var(--cyan)}
.mk.done{border-color:var(--teal);background:var(--teal);box-shadow:0 0 8px rgba(34,196,142,.6)}
.mk.trip{border-color:var(--amber);background:var(--amber);box-shadow:0 0 9px rgba(255,176,32,.7)}
.mk.heal{border-color:var(--teal);background:var(--teal);box-shadow:0 0 9px rgba(34,196,142,.6)}
.row{display:flex;gap:11px;align-items:baseline;flex-wrap:wrap}
.stp{font-family:var(--mono);color:var(--faint);font-size:10.5px;min-width:52px;font-variant-numeric:tabular-nums}
.typ{font-family:var(--mono);font-size:10.5px;color:var(--dim);min-width:88px}
.dsc{font-size:13px;line-height:1.45}
.mono{font-family:var(--mono)}
.blk{border-radius:11px;padding:12px 14px;margin:8px 0 12px}
.blk.trip{background:linear-gradient(180deg,rgba(255,176,32,.09),rgba(255,176,32,.03));border:1px solid rgba(255,176,32,.4)}
.blk.heal{background:linear-gradient(180deg,rgba(34,196,142,.08),rgba(34,196,142,.02));border:1px solid rgba(34,196,142,.38)}
.blk .bt{display:flex;align-items:center;gap:8px;font-weight:600;font-size:12.5px;letter-spacing:.3px;font-family:var(--mono)}
.blk.trip .bt{color:var(--amber-hi)} .blk.heal .bt{color:var(--teal)}
.blk .bt svg{filter:drop-shadow(0 0 4px currentColor)}
.pill{margin-left:auto;font-family:var(--mono);font-size:10px;font-weight:600;padding:2px 8px;border-radius:6px;letter-spacing:.5px}
.pill.inject{background:rgba(34,196,142,.16);color:var(--teal)}
.pill.escalate,.pill.abort{background:rgba(255,92,119,.16);color:var(--red)}
.blk .rz{font-size:12.5px;margin-top:7px;line-height:1.5;color:var(--text)}
.blk .rz b{color:var(--dim);font-weight:600}
.blk .instr{font-family:var(--mono);font-size:12px;line-height:1.55;margin-top:9px;color:var(--text);
  background:rgba(0,0,0,.28);border:1px solid var(--hair);border-radius:8px;padding:10px 12px;white-space:pre-wrap}
.foot{color:var(--faint);font-family:var(--mono);font-size:11px;margin-top:24px;
  border-top:1px solid var(--hair);padding-top:14px}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
@media(max-width:920px){.app{grid-template-columns:1fr}.readouts{grid-template-columns:repeat(3,1fr)}.grid2{grid-template-columns:1fr}}
</style>"""

BOLT = '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M13 2 4 14h6l-1 8 9-12h-6z"/></svg>'
BOLT_SM = '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M13 2 4 14h6l-1 8 9-12h-6z"/></svg>'
STEER = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v5h-5"/></svg>'

BODY = f"""<div class="app">
  <aside class="rail">
    <div class="brand">{BOLT}<span class="wm">AgentFuse</span><span class="tag">SUPERVISOR</span></div>
    <div class="status"><span class="led live"></span><span id="statusline">ONLINE</span></div>
    <div class="rail-label">SUPERVISED RUNS</div>
    <div class="runlist" id="runlist"></div>
    <div class="rail-foot">{BOLT_SM}<span>github.com/ns-0437/agentfuse</span></div>
  </aside>
  <main class="main" id="main"></main>
</div>"""

SCRIPT = """<script id="data" type="application/json">__DATA__</script>
<script>
const BOLT = '__BOLT__', STEER = '__STEER__';
const RUNS = JSON.parse(document.getElementById('data').textContent);
let active = 0;
const sumOf = r => r.records.find(x=>x.kind==='summary')||{};
const metaOf = r => r.records.find(x=>x.kind==='meta')||{};
const esc = s => String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

document.getElementById('statusline').textContent = 'ONLINE  ·  ' + RUNS.length + ' RUNS';

function outcome(s){
  if(s.status==='complete') return (s.recoveries>0)
    ? {cls:'oc-heal', label:'SELF-HEALED', led:'var(--teal)'}
    : {cls:'oc-plain', label:'COMPLETED', led:'var(--teal)'};
  if(s.status==='escalated') return {cls:'oc-esc', label:'ESCALATED', led:'var(--red)'};
  return {cls:'oc-plain', label:(s.status||'—').toUpperCase(), led:'var(--faint)'};
}

function renderList(){
  document.getElementById('runlist').innerHTML = RUNS.map((r,i)=>{
    const s=sumOf(r), st=s.status||'x';
    return `<button class="run ${i===active?'active':''}" onclick="select(${i})">
      <div class="top"><span class="nm">${esc(r.title)}</span><span class="st st-${st}">${st}</span></div>
      <div class="ds">${esc(r.subtitle)}</div>
      <div class="mt"><span><b>${s.trips??0}</b> trips</span><span><b>${s.recoveries??0}</b> heals</span><span><b>${(s.total_tokens??0).toLocaleString()}</b> tok</span></div>
    </button>`;
  }).join('');
}

function sparkline(records){
  let cum=0, pts=[], step=0;
  records.forEach(e=>{ if(e.kind==='event'){ cum+=(e.tokens_in||0)+(e.tokens_out||0); step++; pts.push(cum);} });
  if(pts.length<2) return '<div class="peak">no token data</div>';
  const W=560,H=124,pad=8,maxY=Math.max(...pts)||1;
  const X=i=>pad+(i/(pts.length-1))*(W-2*pad), Y=v=>H-pad-(v/maxY)*(H-2*pad);
  const line=pts.map((v,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  const area=`${line} L${X(pts.length-1).toFixed(1)},${H-pad} L${pad},${H-pad} Z`;
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#FFB020" stop-opacity=".28"/><stop offset="1" stop-color="#FFB020" stop-opacity="0"/></linearGradient></defs>
    <path d="${area}" fill="url(#fill)"/>
    <path class="line" d="${line}" fill="none" stroke="#FFB020" stroke-width="2" stroke-linejoin="round"/>
    <circle cx="${X(pts.length-1).toFixed(1)}" cy="${Y(maxY).toFixed(1)}" r="3.5" fill="#FFC658"/>
  </svg><div class="peak">peak cumulative &nbsp;<b>${maxY.toLocaleString()}</b>&nbsp; tokens</div>`;
}

function detectors(r){
  const hot=new Set(r.records.filter(x=>x.kind==='trip').map(x=>x.detector));
  return '<div class="dets">'+['loop','drift','progress','spend'].map(d=>
    `<span class="det ${hot.has(d)?'hot':''}"><span class="led"></span>${d}</span>`).join('')+'</div>';
}
function routeView(s){
  const p=(s.route||'').split('->').map(x=>x.trim()).filter(Boolean);
  if(!p.length) return '<div class="peak">—</div>';
  return '<div class="route">'+p.map((n,i)=>`<span class="n">${esc(n)}</span>${i<p.length-1?'<span class="w">&rarr;</span>':''}`).join('')+'</div>';
}

function timeline(r){
  let h='<div class="tl">';
  for(const rec of r.records){
    if(rec.kind==='meta'||rec.kind==='summary') continue;
    if(rec.kind==='event'){
      const t=rec.type; let d='',cls='';
      if(t==='tool_call'){cls='tool';d=`<span class="mono" style="color:var(--violet)">${esc(rec.tool_name)}</span><span class="mono" style="color:var(--faint)">(${esc(JSON.stringify(rec.tool_args||{}))})</span>`;}
      else if(t==='tool_result'){cls='tool';d=`<span class="mono" style="color:var(--dim)">${esc(rec.tool_name||'')}</span> &rarr; ${esc((rec.text||'').slice(0,80))}`;}
      else if(t==='llm_call'){cls='llm';d=rec.text?esc(rec.text.slice(0,110)):'<span style="color:var(--faint)">model turn</span>';}
      else if(t==='route'){cls='';d=`<span class="mono">${esc(rec.text||rec.node)}</span>`;}
      else if(t==='state_update'){cls='done';d='<span class="mono" style="color:var(--teal)">state advanced</span>';}
      else if(t==='resume'){cls='heal';d=`<span style="color:var(--teal)">${esc(rec.text||'resumed')}</span>`;}
      else if(t==='complete'){cls='done';d=`<span style="color:var(--teal)">${esc(rec.text||'objective complete')}</span>`;}
      else d=esc(rec.text||'');
      h+=`<div class="ev"><div class="mk ${cls}"></div><div class="row"><span class="stp">step ${rec.step}</span><span class="typ">${t}</span><span class="dsc">${d}</span></div></div>`;
    }
    if(rec.kind==='trip')
      h+=`<div class="ev"><div class="mk trip"></div><div class="blk trip"><div class="bt">${BOLT} CIRCUIT BREAKER TRIPPED — ${esc(rec.detector.toUpperCase())} · ${esc(rec.severity)}</div><div class="rz">${esc(rec.reason)}</div></div></div>`;
    if(rec.kind==='recovery')
      h+=`<div class="ev"><div class="mk heal"></div><div class="blk heal"><div class="bt">${STEER} STEERING RECOVERY<span class="pill ${esc(rec.action)}">${esc(rec.action)}</span></div><div class="rz"><b>Why</b> ${esc(rec.rationale)} <span class="mono" style="color:var(--faint)">· conf ${rec.confidence} · ${esc(rec.backend)}</span></div><div class="instr">${esc(rec.instruction)}</div></div></div>`;
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
        <div class="eyebrow"><span class="node"></span>RUN ${String(active+1).padStart(2,'0')} / ${String(RUNS.length).padStart(2,'0')}  ·  ${esc((r.kind||'run').toUpperCase())}</div>
        <div class="h-title">${esc(r.title)}</div>
        <div class="h-obj">${esc(m.original_goal||r.subtitle)}</div>
        <div class="h-meta">backend ${esc(m.recovery_backend||'—')} &nbsp;·&nbsp; loop&ge;${cfg.loop_threshold??'—'} &nbsp;·&nbsp; drift&lt;${cfg.drift_threshold??'—'} &nbsp;·&nbsp; ${esc(r.file)}</div>
      </div>
      <div class="outcome">
        <div class="big ${oc.cls}"><span class="led" style="color:${oc.led}"></span>${oc.label}</div>
        <div class="sub">${s.trips??0} trips detected · ${s.recoveries??0} recoveries</div>
      </div>
    </div>
    <div class="readouts">
      <div class="ro key trip"><div class="k">${BOLT} TRIPS</div><div class="v">${s.trips??0}</div></div>
      <div class="ro key heal"><div class="k">${STEER} RECOVERIES</div><div class="v">${s.recoveries??0}</div></div>
      <div class="ro"><div class="k">STATUS</div><div class="v" style="font-size:15px;padding-top:6px">${esc(s.status||'—')}</div></div>
      <div class="ro"><div class="k">STEPS</div><div class="v">${s.steps??'—'}</div></div>
      <div class="ro"><div class="k">TOKENS</div><div class="v">${(s.total_tokens??0).toLocaleString()}</div></div>
      <div class="ro"><div class="k">COST</div><div class="v">$${s.total_cost_usd??0}</div></div>
    </div>
    <div class="grid2">
      <div class="panel"><div class="p-hd">Cumulative token spend</div>${sparkline(r.records)}</div>
      <div class="panel"><div class="p-hd">Detectors</div>${detectors(r)}
        <div class="p-hd" style="margin-top:18px">Graph route</div>${routeView(s)}</div>
    </div>
    <div class="panel"><div class="p-hd">Execution timeline</div>${timeline(r)}</div>
    <div class="foot">AgentFuse · logical circuit breaker for long-range agents — generated from JSONL run traces, fully self-contained.</div>`;
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
