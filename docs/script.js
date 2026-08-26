async function loadData() {
  try {
    const res = await fetch("data.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`data.json fetch failed: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error(err);
    return null;
  }
}

function renderPerformanceTable(rows) {
  const tbody = document.querySelector("#performance-table tbody");
  tbody.innerHTML = "";

  if (!rows || rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="muted">No evaluation data available yet.</td></tr>`;
    return;
  }

  for (const row of rows) {
    const tr = document.createElement("tr");
    const liftClass = row.lift_pct == null ? "" : row.lift_pct >= 0 ? "lift-positive" : "lift-negative";
    const liftText = row.lift_pct == null ? "--" : `${row.lift_pct >= 0 ? "+" : ""}${row.lift_pct}%`;

    tr.innerHTML = `
      <td>${capitalize(row.prop)}</td>
      <td>${row.mae ?? "--"}</td>
      <td>${row.baseline_mae ?? "--"}</td>
      <td class="${liftClass}">${liftText}</td>
      <td>${row.rows_test ?? "--"}</td>
      <td>${row.test_date_min ?? "?"} &ndash; ${row.test_date_max ?? "?"}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderSlate(slate) {
  const container = document.getElementById("slate-content");

  if (slate === null) {
    container.innerHTML = `<p class="muted">No live slate right now &mdash; check back once the NBA season starts (~October).</p>`;
    return;
  }

  if (slate.length === 0) {
    container.innerHTML = `<p class="muted">No qualifying props for today's slate.</p>`;
    return;
  }

  const cols = ["game_date", "player_name", "prop", "chosen_side", "line", "mu_pred", "edge", "decision"];
  const headers = ["Date", "Player", "Prop", "Side", "Line", "Model", "Edge", "Decision"];

  let html = `<div class="table-scroll"><table><thead><tr>`;
  for (const h of headers) html += `<th>${h}</th>`;
  html += `</tr></thead><tbody>`;

  for (const row of slate) {
    html += "<tr>";
    for (const c of cols) {
      let val = row[c] ?? "--";
      if (c === "edge" && typeof val === "number") val = `${(val * 100).toFixed(1)}%`;
      if (c === "prop") val = capitalize(val);
      html += `<td>${val}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table></div>";
  container.innerHTML = html;
}

function capitalize(s) {
  if (!s) return s;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

(async function init() {
  const data = await loadData();

  if (!data) {
    document.getElementById("generated-at").textContent = "Data unavailable.";
    return;
  }

  renderPerformanceTable(data.model_performance);
  renderSlate(data.todays_slate);

  const generated = new Date(data.generated_at);
  document.getElementById("generated-at").textContent = `Last updated: ${generated.toLocaleString()}`;
})();
