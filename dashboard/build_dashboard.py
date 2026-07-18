"""Build the AgentFuse observability dashboard from run traces.

Reads every ``runs/*.jsonl`` trace and emits two self-contained, dependency-free
artifacts with the data embedded inline (no server, no CDN, no build step):

  * ``dashboard/index.html``    — full standalone page; open directly in a browser
  * ``dashboard/artifact.html`` — body-only fragment for publishing as a
                                  Claude Artifact (no <!doctype>/<html>/<head>/<body>)

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
OUT_PAGES = ROOT / "docs" / "index.html"  # GitHub Pages serves from /docs

CATALOG = {
    "loop_trap.jsonl": ("Infinite Tool Loop", "Agent repeats a doomed tool call; the breaker detects it and forces a new path.", "loop"),
    "drift.jsonl": ("Goal Drift", "Agent slowly wanders off-objective; the breaker re-anchors it to the original goal.", "drift"),
    "escalation.jsonl": ("Human Escalation", "An unrecoverable failure; the breaker hard-stops and hands control to a human.", "escalate"),
    "real_agentkit.jsonl": ("Real AgentKit Run", "Live openai-agents Runner + real hooks; a real run that self-healed.", "real"),
    "real_gpt.jsonl": ("Real GPT Run", "Live GPT model driving a real agent; supervised and self-healed.", "real"),
}

TITLE = "AgentFuse — Observability Dashboard"

STYLE = """<style>
:root{
  --bg:#0b0f17; --bg2:#111826; --panel:#151d2c; --panel2:#1b2536;
  --line:#243044; --text:#e6edf6; --dim:#8a99ad; --dim2:#5f6f85;
  --trip:#f5c542; --heal:#38bdf8; --ok:#34d399; --bad:#fb7185; --tool:#a78bfa; --llm:#60a5fa;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
html{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#16203400,#0b0f17),var(--bg);
  color:var(--text);font-family:Inter,Segoe UI,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--heal)}
.app{display:grid;grid-template-columns:320px 1fr;min-height:100vh}
.side{border-right:1px solid var(--line);background:linear-gradient(180deg,#0e1524,#0b0f17);padding:22px 16px}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.bolt{font-size:22px}
.brand h1{font-size:18px;margin:0;letter-spacing:.3px}
.tag{color:var(--dim);font-size:12px;margin:0 0 20px 32px}
.runcard{display:block;width:100%;text-align:left;border:1px solid var(--line);background:var(--panel);
  border-radius:12px;padding:12px 13px;margin-bottom:10px;cursor:pointer;transition:.15s;color:inherit;font:inherit}
.runcard:hover{border-color:#3a4a63;transform:translateY(-1px)}
.runcard:focus-visible{outline:2px solid var(--heal);outline-offset:2px}
.runcard.active{border-color:var(--heal);box-shadow:0 0 0 1px var(--heal) inset}
.runcard .rt{font-weight:600;font-size:14px;display:flex;justify-content:space-between;align-items:center;gap:8px}
.runcard .rs{color:var(--dim);font-size:12px;margin-top:4px;line-height:1.35}
.badge{font-size:11px;padding:2px 8px;border-radius:999px;font-weight:600;white-space:nowrap}
.b-complete{background:rgba(52,211,153,.14);color:var(--ok)}
.b-escalated{background:rgba(251,113,133,.14);color:var(--bad)}
.b-incomplete,.b-max_turns{background:rgba(245,197,66,.14);color:var(--trip)}
.main{padding:26px 30px;overflow:auto}
.hdr h2{margin:0 0 6px;font-size:22px;text-wrap:balance}
.obj{color:var(--dim);font-size:13.5px;max-width:900px;line-height:1.5;
  border-left:3px solid var(--line);padding-left:12px;margin:10px 0 4px}
.meta-row{color:var(--dim2);font-size:12px;margin-top:6px;font-family:var(--mono)}
.tiles{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:22px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px 14px}
.tile .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.6px}
.tile .v{font-size:22px;font-weight:700;margin-top:5px;font-family:var(--mono);font-variant-numeric:tabular-nums}
.tile.trip .v{color:var(--trip)} .tile.heal .v{color:var(--heal)}
.grid2{display:grid;grid-template-columns:1.4fr 1fr;gap:18px;margin-bottom:20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.card h3{margin:0 0 12px;font-size:13px;color:var(--dim);text-transform:uppercase;letter-spacing:.6px}
.spark{width:100%;height:120px}
.route{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-family:var(--mono);font-size:13px}
.node{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:5px 10px}
.arrow{color:var(--dim2)}
.detchips{display:flex;flex-wrap:wrap;gap:8px}
.chip{font-family:var(--mono);font-size:12px;border:1px solid var(--line);border-radius:999px;padding:5px 11px;color:var(--dim)}
.chip.hot{border-color:var(--trip);color:var(--trip);background:rgba(245,197,66,.08)}
.tl{position:relative;margin-top:6px;padding-left:26px}
.tl:before{content:"";position:absolute;left:9px;top:4px;bottom:4px;width:2px;background:var(--line)}
.ev{position:relative;margin-bottom:12px}
.dot{position:absolute;left:-21px;top:3px;width:12px;height:12px;border-radius:50%;
  background:var(--panel);border:2px solid var(--dim2)}
.ev .row{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.step{font-family:var(--mono);color:var(--dim2);font-size:11px;min-width:54px;font-variant-numeric:tabular-nums}
.etype{font-family:var(--mono);font-size:11px;color:var(--dim);min-width:92px}
.edesc{font-size:13.5px}
.mono{font-family:var(--mono)}
.dot.tool{border-color:var(--tool)} .dot.llm{border-color:var(--llm)}
.dot.route{border-color:var(--dim)} .dot.state{border-color:var(--ok)}
.dot.complete{border-color:var(--ok);background:var(--ok)}
.panel-trip,.panel-heal{border-radius:12px;padding:13px 15px;margin:10px 0 14px}
.panel-trip{background:rgba(245,197,66,.07);border:1px solid rgba(245,197,66,.4)}
.panel-heal{background:rgba(56,189,248,.07);border:1px solid rgba(56,189,248,.4)}
.panel-trip .t,.panel-heal .t{font-weight:700;font-size:13px;letter-spacing:.3px;display:flex;gap:8px;align-items:center}
.panel-trip .t{color:var(--trip)} .panel-heal .t{color:var(--heal)}
.panel .reason{color:var(--text);font-size:13px;margin-top:7px;line-height:1.5}
.panel .instr{color:var(--text);font-size:13px;margin-top:9px;line-height:1.55;
  background:#0d1420;border:1px solid var(--line);border-radius:8px;padding:10px 12px;white-space:pre-wrap}
.pill{font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px;margin-left:auto}
.pill.inject{background:rgba(56,189,248,.16);color:var(--heal)}
.pill.escalate,.pill.abort{background:rgba(251,113,133,.16);color:var(--bad)}
.foot{color:var(--dim2);font-size:12px;margin-top:26px;border-top:1px solid var(--line);padding-top:14px}
@media(prefers-reduced-motion:reduce){.runcard{transition:none}.runcard:hover{transform:none}}
@media(max-width:900px){.app{grid-template-columns:1fr}.tiles{grid-template-columns:repeat(3,1fr)}.grid2{grid-template-columns:1fr}}
</style>"""

BODY = """<div class="app">
  <aside class="side">
    <div class="brand"><span class="bolt">⚡</span><h1>AgentFuse</h1></div>
    <p class="tag">Logical circuit breaker · run observability</p>
    <div id="runlist"></div>
  </aside>
  <main class="main" id="main"></main>
</div>"""

SCRIPT = """<script id="data" type="application/json">__DATA__</script>
<script>
const RUNS = JSON.parse(document.getElementById('data').textContent);
let active = 0;

function summaryOf(r){return r.records.find(x=>x.kind==='summary')||{};}
function metaOf(r){return r.records.find(x=>x.kind==='meta')||{};}

function renderList(){
  const el = document.getElementById('runlist');
  el.innerHTML = RUNS.map((r,i)=>{
    const s = summaryOf(r); const st = s.status||'—';
    return `<button class="runcard ${i===active?'active':''}" onclick="select(${i})">
      <div class="rt">${r.title}<span class="badge b-${st}">${st}</span></div>
      <div class="rs">${r.subtitle}</div>
      <div class="rs mono" style="margin-top:6px;color:var(--dim2)">
        ⚡ ${s.trips??0} trips · 🧭 ${s.recoveries??0} recoveries · ${s.total_tokens??0} tok</div>
    </button>`;
  }).join('');
}

function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function sparkline(events){
  let cum=0; const pts=[]; let step=0;
  events.forEach(e=>{ if(e.kind==='event'){ cum += (e.tokens_in||0)+(e.tokens_out||0); step++; pts.push([step,cum]); }});
  if(pts.length<2) return '<div class="rs">no token data</div>';
  const W=520,H=110,pad=6; const maxX=pts.length, maxY=Math.max(...pts.map(p=>p[1]))||1;
  const X=i=>pad+(i/(maxX-1))*(W-2*pad), Y=v=>H-pad-(v/maxY)*(H-2*pad);
  const line=pts.map((p,i)=>`${i?'L':'M'}${X(i).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ');
  const area=`${line} L${X(pts.length-1).toFixed(1)},${H-pad} L${X(0).toFixed(1)},${H-pad} Z`;
  const ex=X(pts.length-1).toFixed(1), ey=Y(maxY).toFixed(1);
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#38bdf8" stop-opacity=".35"/><stop offset="1" stop-color="#38bdf8" stop-opacity="0"/>
    </linearGradient></defs>
    <path d="${area}" fill="url(#g)"/><path d="${line}" fill="none" stroke="#38bdf8" stroke-width="2"/>
    <circle cx="${ex}" cy="${ey}" r="3.5" fill="#38bdf8"/>
  </svg><div class="rs mono">peak cumulative: ${maxY.toLocaleString()} tokens</div>`;
}

function routeView(s){
  const parts=(s.route||'').split('->').map(x=>x.trim()).filter(Boolean);
  if(!parts.length) return '<div class="rs">—</div>';
  return '<div class="route">'+parts.map((n,i)=>
    `<span class="node">${esc(n)}</span>${i<parts.length-1?'<span class="arrow">→</span>':''}`).join('')+'</div>';
}

function detectorChips(r){
  const tripped=new Set(r.records.filter(x=>x.kind==='trip').map(x=>x.detector));
  const all=['loop','drift','progress','spend'];
  return '<div class="detchips">'+all.map(d=>
    `<span class="chip ${tripped.has(d)?'hot':''}">${d}${tripped.has(d)?' ⚡':''}</span>`).join('')+'</div>';
}

function timeline(r){
  let html='<div class="tl">';
  for(const rec of r.records){
    if(rec.kind==='meta') continue;
    if(rec.kind==='summary') continue;
    if(rec.kind==='event'){
      const t=rec.type; let desc='';
      if(rec.tool_name && t==='tool_call') desc=`<span class="mono" style="color:var(--tool)">${esc(rec.tool_name)}</span><span class="mono" style="color:var(--dim2)">(${esc(JSON.stringify(rec.tool_args||{}))})</span>`;
      else if(t==='tool_result') desc=`<span class="mono" style="color:var(--dim)">${esc(rec.tool_name||'')}</span> → ${esc((rec.text||'').slice(0,80))}`;
      else if(t==='llm_call') desc=rec.text?esc(rec.text.slice(0,110)):'<span class="rs">model turn</span>';
      else if(t==='route') desc=`<span class="mono">${esc(rec.text||rec.node)}</span>`;
      else if(t==='state_update') desc='<span class="mono" style="color:var(--ok)">state advanced ✓</span>';
      else if(t==='resume') desc=`<span style="color:var(--heal)">▸ ${esc(rec.text||'resumed')}</span>`;
      else if(t==='complete') desc=`<span style="color:var(--ok)">✔ ${esc(rec.text||'objective complete')}</span>`;
      else desc=esc(rec.text||'');
      const cls=['tool_call','tool_result'].includes(t)?'tool':t==='llm_call'?'llm':t==='route'?'route':t==='state_update'?'state':t==='complete'?'complete':'';
      html+=`<div class="ev"><div class="dot ${cls}"></div><div class="row">
        <span class="step">step ${rec.step}</span><span class="etype">${t}</span>
        <span class="edesc">${desc}</span></div></div>`;
    }
    if(rec.kind==='trip'){
      html+=`<div class="ev"><div class="dot" style="border-color:var(--trip);background:var(--trip)"></div>
        <div class="panel-trip"><div class="t">⚡ CIRCUIT BREAKER TRIPPED — ${esc(rec.detector.toUpperCase())} · ${esc(rec.severity)}</div>
        <div class="reason">${esc(rec.reason)}</div></div></div>`;
    }
    if(rec.kind==='recovery'){
      html+=`<div class="ev"><div class="dot" style="border-color:var(--heal);background:var(--heal)"></div>
        <div class="panel-heal"><div class="t">🧭 STEERING RECOVERY
          <span class="pill ${esc(rec.action)}">${esc(rec.action)}</span></div>
        <div class="reason"><b>Why:</b> ${esc(rec.rationale)} <span class="mono" style="color:var(--dim2)">(conf ${rec.confidence}, via ${esc(rec.backend)})</span></div>
        <div class="instr">${esc(rec.instruction)}</div></div></div>`;
    }
  }
  return html+'</div>';
}

function select(i){active=i;render();}
function render(){
  renderList();
  const r=RUNS[active]; const s=summaryOf(r); const m=metaOf(r);
  const cfg=m.config||{};
  document.getElementById('main').innerHTML=`
    <div class="hdr">
      <h2>${r.title}</h2>
      <div class="obj">${esc(m.original_goal||r.subtitle)}</div>
      <div class="meta-row">recovery backend: ${esc(m.recovery_backend||'—')} · loop≥${cfg.loop_threshold??'—'} · drift&lt;${cfg.drift_threshold??'—'} · max_tokens ${cfg.max_tokens??'—'} · trace: runs/${r.file}</div>
    </div>
    <div class="tiles">
      <div class="tile"><div class="k">Status</div><div class="v" style="font-size:16px">${s.status||'—'}</div></div>
      <div class="tile"><div class="k">Steps</div><div class="v">${s.steps??'—'}</div></div>
      <div class="tile"><div class="k">Tokens</div><div class="v">${(s.total_tokens??0).toLocaleString()}</div></div>
      <div class="tile"><div class="k">Cost</div><div class="v">$${s.total_cost_usd??0}</div></div>
      <div class="tile trip"><div class="k">Trips</div><div class="v">${s.trips??0}</div></div>
      <div class="tile heal"><div class="k">Recoveries</div><div class="v">${s.recoveries??0}</div></div>
    </div>
    <div class="grid2">
      <div class="card"><h3>Cumulative token spend</h3>${sparkline(r.records)}</div>
      <div class="card"><h3>Detectors</h3>${detectorChips(r)}
        <h3 style="margin-top:16px">Graph route</h3>${routeView(s)}</div>
    </div>
    <div class="card"><h3>Execution timeline</h3>${timeline(r)}</div>
    <div class="foot">AgentFuse — a logical circuit breaker for long-range agents. This dashboard is generated
      from JSONL run traces (<span class="mono">runs/*.jsonl</span>) and is fully self-contained.</div>`;
}
render();
</script>"""


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
            path.name, (path.stem.replace("_", " ").title(), "Supervised agent run.", "run")
        )
        runs.append({"file": path.name, "title": title, "subtitle": subtitle,
                     "kind": kind, "records": records})
    return runs


def main() -> None:
    runs = load_runs()
    if not runs:
        print("No traces found in runs/. Run an example first, e.g. "
              "`python examples/demo_loop_trap.py`.")
        return
    data = json.dumps(runs).replace("</", "<\\/")  # keep </script> from closing early
    script = SCRIPT.replace("__DATA__", data)

    inner = f"<title>{TITLE}</title>\n{STYLE}\n{BODY}\n{script}\n"
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\"/>\n"
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
    print(f"  runs: {len(runs)} ({', '.join(r['file'] for r in runs)}) · records: {total}")
    print(f"  open standalone: file:///{OUT_FULL.as_posix()}")


if __name__ == "__main__":
    main()
