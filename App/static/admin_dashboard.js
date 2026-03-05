// static/js/admin_dashboard.js
console.log("Admin dashboard (merged) loaded");

async function fetchJSON(url){
  const res = await fetch(url, { cache: "no-store" });
  if(!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}
function num(n){ return (n ?? 0).toLocaleString(); }

let submissionsChart, covidChartAPI, covidChartCumCSV;

// ---------- CSV utils for cumulative chart ----------
function parseCSV_simple(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines.shift().split(",").map(h => h.trim());
  const idxEntity = header.indexOf("Entity");
  const idxDay    = header.indexOf("Day");
  const idxTotal  = header.indexOf("Total confirmed cases of COVID-19");
  if (idxEntity < 0 || idxDay < 0 || idxTotal < 0) {
    throw new Error("CSV must have columns: Entity, Day, Total confirmed cases of COVID-19");
  }
  return lines.map(line => {
    const cols = line.split(",");
    return {
      entity: cols[idxEntity],
      day: cols[idxDay],
      total: cols[idxTotal] === "" ? null : Number(cols[idxTotal])
    };
  });
}
function makeSeries(rows, entity) {
  const r = rows.filter(x => x.entity === entity && x.total != null);
  r.sort((a,b) => (a.day < b.day ? -1 : a.day > b.day ? 1 : 0));
  return { dates: r.map(x => x.day), values: r.map(x => x.total) };
}
function alignTo(baseDates, series) {
  const m = new Map(series.dates.map((d,i) => [d, series.values[i]]));
  // forward-fill
  const out = [];
  let last = null;
  for (const d of baseDates) {
    const v = m.get(d);
    if (v != null && Number.isFinite(v)) last = v;
    out.push(last);
  }
  return out;
}
function fmtCompact(v){
  const n = Number(v||0);
  if (n >= 1e9) return (n/1e9).toFixed(1)+'B';
  if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return n.toLocaleString();
}

async function drawSubmissionsChart() {
  const canvas = document.getElementById("chartSubmissions");
  if (!canvas) return;

  try {
    // Fetch stats
    const stats = await fetchJSON("/admin/stats");
    const t = stats.trend_30d || { dates: [], real: [], fake: [], supports: [], refutes: [], insufficient: [] };

    // (re)create chart
    if (window.submissionsChart && window.submissionsChart.destroy) {
      window.submissionsChart.destroy();
    }

    window.submissionsChart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: t.dates,
        datasets: [
          { label: "Real", data: t.real, tension: 0.35 },
          { label: "Fake", data: t.fake, tension: 0.35 },
          { label: "Supports", data: t.supports, tension: 0.35 },
          { label: "Refutes", data: t.refutes, tension: 0.35 },
          { label: "Insufficient", data: t.insufficient, tension: 0.35 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "bottom" } },
        scales: { x: { ticks: { maxTicksLimit: 8 } } }
      }
    });
  } catch (e) {
    console.error("Error in drawing submissions chart:", e);
  }
}

// ---------- Draw cumulative chart from CSV (Malaysia only) ----------
async function drawCovidCumFromCSV() {
  const canvas = document.getElementById("chartCovidCum");
  if (!canvas) return;

  function parseCSV_simple(text) {
    const lines = text.trim().split(/\r?\n/);
    const header = lines.shift().split(",").map(h => h.trim());
    const idxEntity = header.indexOf("Entity");
    const idxDay    = header.indexOf("Day");
    const idxTotal  = header.indexOf("Total confirmed cases of COVID-19");
    if (idxEntity < 0 || idxDay < 0 || idxTotal < 0) {
      throw new Error("CSV must have columns: Entity, Day, Total confirmed cases of COVID-19");
    }
    return lines.map(line => {
      const cols = line.split(",");
      return {
        entity: cols[idxEntity],
        day: cols[idxDay],
        total: cols[idxTotal] === "" ? null : Number(cols[idxTotal])
      };
    });
  }
  function makeSeries(rows, entity) {
    const r = rows.filter(x => x.entity === entity && x.total != null);
    r.sort((a,b) => (a.day < b.day ? -1 : a.day > b.day ? 1 : 0));
    return { dates: r.map(x => x.day), values: r.map(x => x.total) };
  }
  function fmtCompact(v){
    const n = Number(v||0);
    if (n >= 1e9) return (n/1e9).toFixed(1)+'B';
    if (n >= 1e6) return (n/1e6).toFixed(1)+'M';
    if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
    return n.toLocaleString();
  }

  try {
    const res = await fetch("/static/data/covid_cumulative.csv", { cache: "no-store" });
    if (!res.ok) throw new Error("Failed to load covid_cumulative.csv");
    const txt = await res.text();
    const rows = parseCSV_simple(txt);

    // Malaysia series only
    const my = makeSeries(rows, "Malaysia");
    if (!my.dates.length) throw new Error("No Malaysia rows found in csv");

    // (re)create chart
    if (window.covidChartCumCSV && window.covidChartCumCSV.destroy) {
      window.covidChartCumCSV.destroy();
    }
    window.covidChartCumCSV = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: my.dates,
        datasets: [{
          label: "Malaysia (cumulative)",
          data: my.values,
          borderWidth: 2,
          tension: 0.25,
          pointRadius: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: { position: "bottom" },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const y = ctx.raw;
                return `${ctx.dataset.label}: ${y == null ? "n/a" : Number(y).toLocaleString()}`;
              }
            }
          },
          zoom: {
            zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "xy" },
            pan:  { enabled: true, mode: "xy" }
          }
        },
        scales: {
          x: { ticks: { maxTicksLimit: 8 }, grid: { display: false } },
          y: { beginAtZero: false, ticks: { callback: v => fmtCompact(v) } }
        }
      }
    });

    // reset button
    const resetBtn = document.getElementById("resetZoomBtn");
    if (resetBtn) resetBtn.onclick = () => window.covidChartCumCSV?.resetZoom?.();
  } catch (err) {
    console.error("Cumulative chart error:", err);
  }
}
// ---------- Your existing API-driven parts ----------
async function drawAPIWidgets() {
  // KPIs
  const stats = await fetchJSON("/admin/stats");
  const elUsers = document.getElementById("kpiUsers"); 
  const elNews  = document.getElementById("kpiSubs");
  const elFake  = document.getElementById("kpiFake");
  if (elUsers) elUsers.textContent = num(stats.totals?.users);
  if (elNews)  elNews.textContent  = num(stats.totals?.submissions);
  if (elFake)  elFake.textContent  = num(stats.totals?.fake);

  const elKpiReal   = document.getElementById("kpiReal");
  const elKpiAgree  = document.getElementById("kpiAgree");
  if (elKpiReal)  elKpiReal.textContent  = num(stats.totals?.real);
  if (elKpiAgree) elKpiAgree.textContent = Math.round((stats.totals?.agreement_rate || 0) * 100) + "%";

  // Submissions trend (last 30 days)
  const t = stats.trend_30d || { dates: [], real: [], fake: [], supports: [], refutes: [], insufficient: [] };
  const elSub = document.getElementById("submissionsChart");
  if (elSub) {
    submissionsChart?.destroy();
    submissionsChart = new Chart(elSub, {
      type: "line",
      data: {
        labels: t.dates,
        datasets: [
          { label: "Real", data: t.real, tension: 0.35 },
          { label: "Fake", data: t.fake, tension: 0.35 },
          { label: "Supports", data: t.supports, tension: 0.35 },
          { label: "Refutes", data: t.refutes, tension: 0.35 },
          { label: "Insufficient", data: t.insufficient, tension: 0.35 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "bottom" } },
        scales: { x: { ticks: { maxTicksLimit: 8 } } }
      }
    });
  }

  const elCovidAPI = document.getElementById("covidChart");
  if (elCovidAPI) {
    const covid = await fetchJSON("/admin/covid");
    const c = covid.series || { dates: [], new_cases_smoothed: [], new_deaths_smoothed: [] };
    covidChartAPI?.destroy();
    covidChartAPI = new Chart(elCovidAPI, {
      type: "line",
      data: {
        labels: c.dates,
        datasets: [
          { label: "New cases (7d avg)", data: c.new_cases_smoothed, tension: 0.35 },
          { label: "New deaths (7d avg)", data: c.new_deaths_smoothed, tension: 0.35 }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } }
    });
  }

  // Recent activity
  const act = await fetchJSON("/admin/activity");
  const body = document.getElementById("recentBody") || document.getElementById("recentTbody");
  if (body) {
    body.innerHTML = (act.rows || []).map(r => `
      <tr>
        <td>${r.created_at || "-"}</td>
        <td>${r.user_id || "-"}</td>
        <td style="color:#777">${(r.snippet || "").slice(0,120)}</td>
        <td>${r.label || "-"}</td>
        <td>${r.verification || "-"}</td>
        <td>${r.confidence != null ? Math.round(r.confidence*100)+"%" : "-"}</td>
      </tr>
    `).join("");
  }
}

async function init() {
  try {
    await drawAPIWidgets();         
    await drawSubmissionsChart();   
    await drawCovidCumFromCSV();    
  } catch (e) {
    console.error(e);
    alert("Failed to load admin data. Check console.");
  }
}

window.addEventListener("load", init);
