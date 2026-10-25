"""
page.py: the slate as one self-contained HTML page.

The table is rendered on the server in the slate's own order, so the page reads fine with scripts off;
a little inline script adds search, a market filter and click-to-sort. It makes no external requests.
It holds real sportsbook lines, so it is for the private data repo only, never docs/.
"""

from __future__ import annotations

from html import escape

import pandas as pd

from src.serving.slate import EASTERN, MARKET_LABEL, SlateResult

NOTE = (
    "Projection is the model's number and Diff = Projection - Line. There is no injury or lineup "
    "information, so large gaps often reflect news the model doesn't have. Not advice."
)

STYLE = """
:root{--bg:#fafaf8;--fg:#1c1c1a;--muted:#6b6b66;--line:#dcdcd6;--card:#fff;--warn-bg:#fff4d6;--warn-fg:#6b4e00;--accent:#2d5be3}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ececE8;--muted:#9a9a93;--line:#34342f;--card:#1e1e1b;--warn-bg:#3a3118;--warn-fg:#f0d98a;--accent:#8fa8ff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}
.meta{color:var(--muted);font-size:13px;margin:0 0 14px}
.note{font-size:13px;color:var(--muted);margin:0 0 14px}
.warn{background:var(--warn-bg);color:var(--warn-fg);border-radius:6px;padding:8px 12px;margin:0 0 10px;font-size:14px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:14px 0}
.controls input,.controls select{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:7px 10px}
.controls input{flex:1 1 200px;min-width:0}
#count{color:var(--muted);font-size:13px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}
th{position:sticky;top:0;background:var(--card);cursor:pointer;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);user-select:none}
th:hover{color:var(--accent)}
td.n,th.n{text-align:right}
tr[hidden]{display:none}
tbody tr:last-child td{border-bottom:0}
details{margin-top:22px}
summary{cursor:pointer;color:var(--muted)}
details ul{margin:8px 0 0;padding-left:20px;color:var(--muted);font-size:14px}
"""

SCRIPT = """
(function(){
  var table=document.getElementById('slate'); if(!table) return;
  var body=table.tBodies[0], rows=Array.prototype.slice.call(body.rows);
  var q=document.getElementById('q'), m=document.getElementById('mkt'), count=document.getElementById('count');
  function apply(){
    var t=q.value.trim().toLowerCase(), k=m.value, n=0;
    rows.forEach(function(r){
      var show=(!k||r.dataset.mkt===k)&&(!t||r.dataset.text.indexOf(t)>-1);
      r.hidden=!show; if(show) n++;
    });
    count.textContent=n+' of '+rows.length+' lines';
  }
  q.addEventListener('input',apply); m.addEventListener('change',apply);
  var asc={};
  Array.prototype.forEach.call(table.tHead.rows[0].cells,function(th,i){
    th.addEventListener('click',function(){
      asc[i]=!asc[i]; var s=asc[i]?1:-1;
      rows.sort(function(a,b){
        var x=a.cells[i].dataset.v, y=b.cells[i].dataset.v, nx=parseFloat(x), ny=parseFloat(y);
        return ((!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y))*s;
      });
      rows.forEach(function(r){body.appendChild(r);});
    });
  });
  apply();
})();
"""

COLUMNS = [("Tip (ET)", ""), ("Game", ""), ("Player", ""), ("Team", ""), ("Mkt", ""),
           ("Line", "n"), ("Proj", "n"), ("Diff", "n"), ("Min", "n"), ("Books", "n")]


def _cell(text, value, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f'<td{c} data-v="{escape(str(value), quote=True)}">{escape(str(text))}</td>'


def _row(r) -> str:
    label = MARKET_LABEL.get(r.market, r.market)
    game = f"{r.away} @ {r.home}"
    text = f"{r.player} {r.team} {game} {label}".lower()
    cells = [
        _cell(r.tip_et.strftime("%-I:%M %p"), r.tip_et.isoformat()),
        _cell(game, game),
        _cell(r.player, r.player),
        _cell(r.team, r.team),
        _cell(label, label),
        _cell(f"{r.line:.1f}", r.line, "n"),
        _cell(f"{r.projection:.1f}", r.projection, "n"),
        _cell(f"{r.difference:+.1f}", abs(r.difference), "n"),
        _cell(f"{r.min_pred:.0f}", r.min_pred, "n"),
        _cell(r.books, r.books, "n"),
    ]
    return (f'<tr data-mkt="{escape(r.market, quote=True)}" data-text="{escape(text, quote=True)}">'
            + "".join(cells) + "</tr>")


def _skipped(skipped: pd.DataFrame) -> str:
    if skipped.empty:
        return ""
    items = []
    for reason, group in skipped.groupby("reason"):
        names = ", ".join(sorted(group["player"].unique()))
        items.append(f"<li><strong>{escape(reason)}</strong>: {len(group)} ({escape(names)})</li>")
    return f"<details><summary>Skipped ({len(skipped)})</summary><ul>{''.join(items)}</ul></details>"


def render_page(result: SlateResult, summary: dict) -> str:
    meta, slate = result.meta, result.slate
    generated = pd.Timestamp(meta["as_of"]).tz_convert(EASTERN).strftime("%b %-d, %-I:%M %p ET")
    head = (
        f"<h1>NBA slate {escape(meta['slate_date'])}</h1>"
        f"<p class=\"meta\">History through {escape(meta['history_last_date'])} · models trained through "
        f"{escape(str(meta['model_trained_through']))} · generated {escape(generated)} · "
        f"{meta['n_projected']} of {meta['n_lines']} lines projected</p>"
    )
    warnings = "".join(f'<div class="warn">{escape(w)}</div>' for w in summary.get("warnings", []))
    body = f'<p class="note">{escape(NOTE)}</p>'
    if slate.empty:
        body += "<p>No projections for this slate.</p>"
    else:
        markets = sorted(slate["market"].unique())
        options = '<option value="">All markets</option>' + "".join(
            f'<option value="{escape(m, quote=True)}">{escape(MARKET_LABEL.get(m, m))}</option>' for m in markets
        )
        header = "".join(f'<th class="{c}">{escape(n)}</th>' if c else f"<th>{escape(n)}</th>" for n, c in COLUMNS)
        rows = "".join(_row(r) for r in slate.itertuples())
        body += (
            '<div class="controls"><input id="q" type="search" placeholder="Search player, team or game" '
            'aria-label="Search">'
            f'<select id="mkt" aria-label="Market">{options}</select><span id="count"></span></div>'
            f'<div class="wrap"><table id="slate"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
        )
    body += _skipped(result.skipped)
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>NBA slate {escape(meta['slate_date'])}</title><style>{STYLE}</style></head>"
        f"<body><main>{head}{warnings}{body}</main><script>{SCRIPT}</script></body></html>\n"
    )
