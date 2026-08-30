const fallbackRankings = {
  "26469": [
    { video_id: 3131, score: 0.77056956, rank: 1 },
    { video_id: 3797, score: 0.69623744, rank: 2 },
    { video_id: 5963, score: 0.68949180, rank: 3 },
    { video_id: 5137, score: 0.68120060, rank: 4 },
    { video_id: 832, score: 0.66766230, rank: 5 },
    { video_id: 584, score: 0.66081053, rank: 6 },
    { video_id: 4741, score: 0.65397880, rank: 7 },
    { video_id: 4661, score: 0.64686970, rank: 8 }
  ],
  "26540": [
    { video_id: 3984, score: 0.55650410, rank: 1 },
    { video_id: 444, score: 0.54170020, rank: 2 },
    { video_id: 5351, score: 0.51669610, rank: 3 },
    { video_id: 4278, score: 0.49210920, rank: 4 },
    { video_id: 4486, score: 0.48661196, rank: 5 },
    { video_id: 1176, score: 0.48392826, rank: 6 },
    { video_id: 277, score: 0.47918287, rank: 7 },
    { video_id: 5675, score: 0.47453910, rank: 8 }
  ],
  "16967": [
    { video_id: 6989, score: 0.124529235, rank: 1 },
    { video_id: 5351, score: 0.108776170, rank: 2 },
    { video_id: 5675, score: 0.103999450, rank: 3 },
    { video_id: 5931, score: 0.102098980, rank: 4 },
    { video_id: 7287, score: 0.096339405, rank: 5 },
    { video_id: 4927, score: 0.096236855, rank: 6 },
    { video_id: 7299, score: 0.095156240, rank: 7 },
    { video_id: 961, score: 0.094690494, rank: 8 }
  ]
};

const chartData = [
  { label: "Popularity", value: 0.817216, kind: "control" },
  { label: "Matrix factorization", value: 0.818786, kind: "control" },
  { label: "DeepFM · side features", value: 0.832888, kind: "candidate" },
  { label: "DeepFM · all exposures", value: 0.833730, kind: "candidate" },
  { label: "DCN · seed 42", value: 0.837561, kind: "candidate" },
  { label: "DCN · seed 43", value: 0.837940, kind: "winner" }
];

const colors = { control: "#a9a69e", candidate: "#25344a", winner: "#e34a32" };

const personas = {
  "26469": {
    profile: "Full-active · video author · long tenure",
    behavior: "Top score 0.771 across 160 candidates"
  },
  "26540": {
    profile: "Full-active · video author · established audience",
    behavior: "Top score 0.557 across 128 candidates"
  },
  "16967": {
    profile: "High-active · viewer account · mid tenure",
    behavior: "Top score 0.125 across 82 candidates"
  }
};

function activateTab(name, updateHash = true) {
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    const active = panel.dataset.panel === name;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });
  document.querySelectorAll("[data-tab]").forEach((button) => {
    const active = button.dataset.tab === name;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-current", active ? "page" : "false");
  });
  if (updateHash) history.replaceState(null, "", `#${name}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (name === "results") requestAnimationFrame(drawChart);
}

function drawChart() {
  const svg = document.querySelector("#results-chart");
  if (!svg) return;
  const width = Math.max(720, svg.parentElement.clientWidth);
  const height = 390;
  const margin = { top: 22, right: 84, bottom: 52, left: 162 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const min = 0.80;
  const max = 0.84;
  const x = (value) => margin.left + ((value - min) / (max - min)) * innerWidth;
  const row = innerHeight / chartData.length;
  const ticks = [0.80, 0.81, 0.82, 0.83, 0.84];
  const parts = [`<title>Successful KuaiRand-Pure NDCG at 10 results</title>`];

  ticks.forEach((tick) => {
    const px = x(tick);
    parts.push(`<line x1="${px}" y1="${margin.top}" x2="${px}" y2="${height - margin.bottom}" stroke="#dedbd2" stroke-width="1"/>`);
    parts.push(`<text x="${px}" y="${height - 19}" text-anchor="middle" fill="#77736b" font-size="11">${tick.toFixed(2)}</text>`);
  });

  chartData.forEach((item, index) => {
    const y = margin.top + index * row + row / 2;
    const barX = x(min);
    const barWidth = Math.max(2, x(item.value) - barX);
    parts.push(`<text x="${margin.left - 14}" y="${y + 4}" text-anchor="end" fill="#35342f" font-size="12" font-weight="${item.kind === "winner" ? 700 : 500}">${item.label}</text>`);
    parts.push(`<rect x="${barX}" y="${y - 9}" width="${barWidth}" height="18" fill="${colors[item.kind]}" rx="1"/>`);
    parts.push(`<circle cx="${x(item.value)}" cy="${y}" r="5" fill="${colors[item.kind]}" stroke="#fffefa" stroke-width="2"/>`);
    parts.push(`<text x="${x(item.value) + 12}" y="${y + 4}" fill="#35342f" font-size="12" font-weight="700">${item.value.toFixed(6)}</text>`);
  });

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = parts.join("");
}

function renderRanking(rows) {
  const body = document.querySelector("#ranking-body");
  body.innerHTML = rows.slice(0, 8).map((row) => `
    <tr>
      <td><strong>${String(row.rank).padStart(2, "0")}</strong></td>
      <td>${row.video_id}</td>
      <td>${Number(row.score).toFixed(6)}</td>
      <td><div class="score-track" aria-label="${(Number(row.score) * 100).toFixed(1)} percent"><span style="width:${Number(row.score) * 100}%"></span></div></td>
    </tr>`).join("");
}

function renderPersonaEvidence(userId, rows) {
  const profile = personas[userId] || {
    profile: "Anonymized KuaiRand user representation",
    behavior: `${rows.length} top-ranked candidates shown`
  };
  document.querySelector("#persona-evidence").innerHTML = `
    <div><span>Profile signal</span><strong>${profile.profile}</strong></div>
    <div><span>Ranking behavior</span><strong>${profile.behavior}</strong></div>
    <p>The model combines this user's learned representation with candidate and context features. This is ranking evidence—not an explanation of the person's real preferences.</p>`;
  document.querySelectorAll("[data-persona]").forEach((card) => card.classList.toggle("is-selected", card.dataset.persona === userId));
}

async function loadRanking() {
  const userId = document.querySelector("#user-select").value;
  const status = document.querySelector("#data-status");
  try {
    const response = await fetch(`/api/rank?user_id=${encodeURIComponent(userId)}&limit=8`);
    if (!response.ok) throw new Error("API unavailable");
    const payload = await response.json();
    renderRanking(payload.rows);
    renderPersonaEvidence(userId, payload.rows);
    status.textContent = "Live checkpoint export";
  } catch {
    const rows = fallbackRankings[userId] || fallbackRankings["26469"];
    renderRanking(rows);
    renderPersonaEvidence(userId, rows);
    status.textContent = userId === "26469" ? "Local verified sample" : "Start demo server for this user";
  }
}

async function hydrateUsers() {
  try {
    const response = await fetch("/api/sample-users");
    if (!response.ok) return;
    const payload = await response.json();
    const select = document.querySelector("#user-select");
    select.innerHTML = payload.users.map((item) => `<option value="${item.user_id}">User ${item.user_id} · ${item.candidates} candidates</option>`).join("");
  } catch {
    // Static-file mode intentionally keeps the verified embedded example.
  }
}

document.querySelectorAll("[data-tab]").forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.tab)));
document.querySelectorAll("[data-tab-link]").forEach((link) => link.addEventListener("click", (event) => { event.preventDefault(); activateTab(link.dataset.tabLink); }));
document.querySelectorAll("[data-go-to]").forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.goTo)));
document.querySelector("#load-ranking").addEventListener("click", loadRanking);
document.querySelectorAll("[data-persona]").forEach((card) => card.addEventListener("click", () => {
  document.querySelector("#user-select").value = card.dataset.persona;
  loadRanking();
}));
window.addEventListener("resize", () => { if (!document.querySelector('[data-panel="results"]').hidden) drawChart(); });

const initialTab = ["overview", "workflow", "results"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "overview";
activateTab(initialTab, false);
hydrateUsers().then(loadRanking);
