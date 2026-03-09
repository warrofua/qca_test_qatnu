const DATA_URL = './data/current_state.json';

const topologyLayouts = {
  path: {
    edges: [[0, 1], [1, 2], [2, 3], [3, 4]],
    coords: {
      0: [40, 95], 1: [95, 95], 2: [150, 95], 3: [205, 95], 4: [260, 95],
    },
  },
  cycle: {
    edges: [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [4, 0]],
    coords: {
      0: [70, 55], 1: [200, 55], 2: [220, 145], 3: [90, 145], 4: [145, 185],
    },
  },
  star: {
    edges: [[0, 1], [0, 2], [0, 3], [0, 4]],
    coords: {
      0: [145, 100], 1: [145, 28], 2: [48, 100], 3: [242, 100], 4: [145, 175],
    },
  },
};

const viewDefs = {
  scalar: { label: 'Scalar / Transfer' },
  tensor: { label: 'Tensor Falsification' },
  critical: { label: 'Critical Slowing' },
  redteam: { label: 'Redteam Stress' },
};

const state = {
  data: null,
  view: 'scalar',
  tensorN: 'N4',
  tensorLambda: null,
  tensorChi: null,
};

const els = {
  heroText: document.querySelector('#hero-text'),
  stageTitle: document.querySelector('#stage-title'),
  stageSubtitle: document.querySelector('#stage-subtitle'),
  stageGrid: document.querySelector('#stage-grid'),
  insightCopy: document.querySelector('#insight-copy'),
  sourceList: document.querySelector('#source-list'),
  establishedList: document.querySelector('#established-list'),
  indicatedList: document.querySelector('#indicated-list'),
  unsupportedList: document.querySelector('#unsupported-list'),
  viewTabs: document.querySelector('#view-tabs'),
  tensorNSelect: document.querySelector('#tensor-n-select'),
  tensorLambdaSelect: document.querySelector('#tensor-lambda-select'),
  tensorChiSelect: document.querySelector('#tensor-chi-select'),
};

init().catch((error) => {
  document.body.innerHTML = `<pre style="color:white;padding:24px;">${error.stack}</pre>`;
});

async function init() {
  const response = await fetch(DATA_URL);
  state.data = await response.json();
  populateTheory();
  buildControls();
  syncTensorControls();
  render();
}

function populateTheory() {
  const theory = state.data.theory;
  els.heroText.textContent = theory.one_line;
  renderList(els.establishedList, theory.established);
  renderList(els.indicatedList, theory.indicated);
  renderList(els.unsupportedList, theory.not_supported);

  const sourceEntries = [
    ['Theory status', state.data.sources.theory_status],
    ['Results index', state.data.sources.results],
    ['Built data', state.data.generated_at],
  ];
  els.sourceList.innerHTML = sourceEntries
    .map(([label, value]) => `<div><strong>${label}</strong><br><span>${escapeHtml(value)}</span></div>`)
    .join('');
}

function buildControls() {
  els.viewTabs.innerHTML = Object.entries(viewDefs)
    .map(([key, view]) => `<button class="view-tab${key === state.view ? ' active' : ''}" data-view="${key}">${view.label}</button>`)
    .join('');

  els.viewTabs.addEventListener('click', (event) => {
    const button = event.target.closest('[data-view]');
    if (!button) return;
    state.view = button.dataset.view;
    updateViewButtons();
    render();
  });

  const tensorRows = state.data.tensor.covariance;
  const nOptions = [...new Set(tensorRows.map((row) => scenarioN(row.scenario_id)))];
  fillSelect(els.tensorNSelect, nOptions);
  els.tensorNSelect.value = state.tensorN;
  els.tensorNSelect.addEventListener('change', () => {
    state.tensorN = els.tensorNSelect.value;
    syncTensorControls();
    render();
  });

  els.tensorLambdaSelect.addEventListener('change', () => {
    state.tensorLambda = els.tensorLambdaSelect.value;
    render();
  });

  els.tensorChiSelect.addEventListener('change', () => {
    state.tensorChi = els.tensorChiSelect.value;
    render();
  });
}

function syncTensorControls() {
  const rows = state.data.tensor.covariance.filter((row) => scenarioN(row.scenario_id) === state.tensorN);
  const lambdas = [...new Set(rows.map((row) => row.lambda))].sort((a, b) => Number(a) - Number(b));
  const chis = [...new Set(rows.map((row) => row.bond_cutoff))].sort((a, b) => Number(a) - Number(b));
  fillSelect(els.tensorLambdaSelect, lambdas);
  fillSelect(els.tensorChiSelect, chis);
  state.tensorLambda = lambdas.includes(state.tensorLambda) ? state.tensorLambda : lambdas[0];
  state.tensorChi = chis.includes(state.tensorChi) ? state.tensorChi : chis[0];
  els.tensorLambdaSelect.value = state.tensorLambda;
  els.tensorChiSelect.value = state.tensorChi;
}

function fillSelect(select, values) {
  select.innerHTML = values.map((value) => `<option value="${value}">${value}</option>`).join('');
}

function render() {
  const filtersVisible = state.view === 'tensor';
  [els.tensorNSelect, els.tensorLambdaSelect, els.tensorChiSelect].forEach((select) => {
    select.closest('label').style.display = filtersVisible ? 'grid' : 'none';
  });

  if (state.view === 'scalar') renderScalar();
  if (state.view === 'tensor') renderTensor();
  if (state.view === 'critical') renderCritical();
  if (state.view === 'redteam') renderRedteam();
}

function renderScalar() {
  const rows = state.data.scalar.holdout;
  els.stageTitle.textContent = 'Scalar / Transfer Stage';
  els.stageSubtitle.textContent = 'Clock slowdown remains the strongest supported phenomenon. The visual here is the phase structure itself: c1, revival, c2, and residual by topology.';
  els.stageGrid.innerHTML = rows.map(renderScalarCard).join('');
  els.insightCopy.innerHTML = [
    paragraph('The notable feature is not that every topology behaves the same. It is the opposite: the scalar slowdown law survives, but the landmark geometry depends strongly on graph structure.'),
    paragraph('Path can satisfy the locked holdout while star fails under the same parameter lock. That makes topology part of the phenomenon rather than an implementation nuisance.'),
  ].join('');
}

function renderTensor() {
  const slice = state.data.tensor.covariance.filter((row) =>
    scenarioN(row.scenario_id) === state.tensorN &&
    row.lambda === state.tensorLambda &&
    row.bond_cutoff === state.tensorChi
  );
  const ordered = ['path', 'cycle', 'star']
    .map((topology) => slice.find((row) => row.topology === topology))
    .filter(Boolean);

  els.stageTitle.textContent = 'Tensor Falsification Stage';
  els.stageSubtitle.textContent = `Comparing background-subtracted TT proxies across topologies at ${state.tensorN}, λ=${state.tensorLambda}, χ=${state.tensorChi}.`;
  els.stageGrid.innerHTML = ordered.map(renderTensorCard).join('');
  els.insightCopy.innerHTML = [
    paragraph('This view is intentionally comparative. A visually appealing single topology is not enough. What matters is whether any topology stays near target while controls do not.'),
    paragraph('Right now the covariance-background observable does not deliver a clean, star-only, finite-size-stable tensor win. That is why the scalar story remains the core result.'),
  ].join('');
}

function renderCritical() {
  const rows = state.data.critical.star_sensitivity;
  els.stageTitle.textContent = 'Critical Slowing Stage';
  els.stageSubtitle.textContent = 'The star peak moves under protocol and phenomenological knobs. This view makes that fragility visible instead of burying it in CSVs.';
  els.stageGrid.innerHTML = rows.map(renderCriticalCard).join('');
  els.insightCopy.innerHTML = [
    paragraph('The main lesson is not just where the peak is. It is that the peak drifts when hotspot, κ, and ΔB change.'),
    paragraph('That makes any one pretty star window scientifically weak unless it survives the sensitivity matrix.'),
  ].join('');
}

function renderRedteam() {
  const rows = state.data.redteam.harmonic_rank;
  els.stageTitle.textContent = 'Redteam Stress Stage';
  els.stageSubtitle.textContent = 'This view shows how sensitive the harmonic-background conclusion is to projector rank choices.';
  els.stageGrid.innerHTML = rows.map(renderRedteamCard).join('');
  els.insightCopy.innerHTML = [
    paragraph('The redteam pass matters because it prevents us from over-claiming a fragile effect. The current tensor verdict is weaker precisely because these stress tests were performed.'),
    paragraph('The scientific value here is discipline: the project can now show both signal and failure modes in the same visual language.'),
  ].join('');
}

function renderScalarCard(row) {
  const topology = normalizeTopology(row.graph);
  const c1 = number(row.lambda_c1);
  const revival = number(row.lambda_revival);
  const c2 = number(row.lambda_c2);
  const residual = number(row.residual_min);
  const pass = String(row.scenario_pass).toLowerCase() === 'true';
  return `
    <article class="topology-card">
      <div class="card-header">
        <div>
          <div class="card-topology">${capitalize(topology)}</div>
          <div class="card-meta">${escapeHtml(row.scenario_id)}</div>
        </div>
        <span class="card-badge${pass ? '' : ' alert'}">${pass ? 'Pass' : 'Fail'}</span>
      </div>
      ${renderTopologySVG(topology, { emphasis: pass ? 'signal' : 'warning' })}
      <div class="timeline">
        <div class="timeline-track"></div>
        ${timelineMarker('c1', c1, c2)}
        ${timelineMarker('rev', revival, c2)}
        ${timelineMarker('c2', c2, c2)}
      </div>
      <div class="metric-rail">
        ${metricRow('Residual', residual, 0.4, !pass)}
        ${metricRow('Revival', revival, 1.4, false)}
        ${metricRow('Window', c2 - c1, 1.0, false)}
      </div>
    </article>`;
}

function renderTensorCard(row) {
  const topology = normalizeTopology(row.topology);
  const rawPower = number(row.raw_power);
  const shellPower = number(row.bg_power);
  const covPower = number(row.covbg_power);
  const nearTarget = Math.abs(covPower - 2) < 0.8;
  return `
    <article class="topology-card">
      <div class="card-header">
        <div>
          <div class="card-topology">${capitalize(topology)}</div>
          <div class="card-meta">${scenarioN(row.scenario_id)} · backend ${escapeHtml(row.effective_backend)}</div>
        </div>
        <span class="card-badge${nearTarget ? '' : ' alert'}">cov ${fmt(covPower)}</span>
      </div>
      ${renderTopologySVG(topology, { emphasis: topology === 'star' ? 'signal' : 'neutral' })}
      <div class="metric-rail">
        ${signedMetricRow('Raw TT', rawPower, 2.5)}
        ${signedMetricRow('Shell BG', shellPower, 2.5)}
        ${signedMetricRow('Cov BG', covPower, 2.5)}
      </div>
      <div class="comparison-strip">
        <div class="compare-row"><strong>Target</strong><div class="metric-track"><div class="metric-fill" style="width:80%"></div></div><span>p≈2</span></div>
        <div class="compare-row"><strong>Actual</strong><div class="metric-track"><div class="metric-fill${nearTarget ? '' : ' alert'}" style="width:${Math.min(Math.abs(covPower) / 2.5, 1) * 100}%"></div></div><span>${fmt(covPower)}</span></div>
      </div>
    </article>`;
}

function renderCriticalCard(row) {
  const peak = number(row.peak_lambda);
  const tau = number(row.peak_tau_dephase_probe);
  const label = row.group.replace('_', ' ');
  return `
    <article class="topology-card">
      <div class="card-header">
        <div>
          <div class="card-topology">Star sensitivity</div>
          <div class="card-meta">${escapeHtml(label)} · χ=${escapeHtml(row.chi || '-')}</div>
        </div>
        <span class="card-badge">λ* ${fmt(peak)}</span>
      </div>
      ${renderTopologySVG('star', { emphasis: 'signal' })}
      <div class="metric-rail">
        ${metricRow('Peak λ', peak, 1.8, false)}
        ${metricRow('Peak τ', tau, 8, false)}
        ${metricRow('Hotspot', number(row.hotspot), 4, false)}
      </div>
    </article>`;
}

function renderRedteamCard(row) {
  const span = number(row.power_span);
  const pMin = number(row.power_min);
  const pMax = number(row.power_max);
  const topology = inferTopology(row.scenario_id);
  return `
    <article class="topology-card">
      <div class="card-header">
        <div>
          <div class="card-topology">${capitalize(topology)} fragility</div>
          <div class="card-meta">${escapeHtml(row.scenario_id)}</div>
        </div>
        <span class="card-badge alert">span ${fmt(span)}</span>
      </div>
      ${renderTopologySVG(topology, { emphasis: 'warning' })}
      <div class="metric-rail">
        ${signedMetricRow('Power min', pMin, 3)}
        ${signedMetricRow('Power max', pMax, 3)}
        ${metricRow('Span', span, 3, true)}
      </div>
    </article>`;
}

function renderTopologySVG(topology, { emphasis = 'neutral' } = {}) {
  const layout = topologyLayouts[topology] || topologyLayouts.path;
  const nodes = Object.entries(layout.coords).map(([id, [x, y]]) => {
    const klass = id === '0' ? 'node hot' : emphasis === 'warning' ? 'node' : 'node cool';
    return `
      <circle class="${klass}" cx="${x}" cy="${y}" r="18"></circle>
      <text class="node-label" x="${x}" y="${y}">${id}</text>`;
  }).join('');
  const edges = layout.edges.map(([a, b], index) => {
    const [x1, y1] = layout.coords[a];
    const [x2, y2] = layout.coords[b];
    const klass = index === 0 || a === 0 || b === 0 ? 'edge pulse' : 'edge';
    return `<line class="${klass}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"></line>`;
  }).join('');
  return `<svg class="topology-svg" viewBox="0 0 290 210" role="img" aria-label="${topology} topology">${edges}${nodes}</svg>`;
}

function metricRow(label, value, max, alert = false) {
  const width = Math.max(0, Math.min(1, value / max)) * 100;
  return `
    <div class="metric-row">
      <span>${escapeHtml(label)}</span>
      <div class="metric-track"><div class="metric-fill${alert ? ' alert' : ''}" style="width:${width}%"></div></div>
      <strong>${fmt(value)}</strong>
    </div>`;
}

function signedMetricRow(label, value, scale) {
  const width = Math.max(0, Math.min(1, Math.abs(value) / scale)) * 100;
  return `
    <div class="metric-row">
      <span>${escapeHtml(label)}</span>
      <div class="metric-track"><div class="metric-fill${value < 0 ? ' alert' : ''}" style="width:${width}%"></div></div>
      <strong>${fmt(value)}</strong>
    </div>`;
}

function timelineMarker(label, value, limit) {
  const left = limit > 0 ? Math.max(0, Math.min(100, (value / limit) * 100)) : 0;
  return `<div class="timeline-marker" style="left:${left}%"><span class="timeline-dot"></span><span class="timeline-label">${escapeHtml(label)}</span></div>`;
}

function paragraph(text) {
  return `<p>${escapeHtml(text)}</p>`;
}

function renderList(target, items) {
  target.innerHTML = items.map((item) => `<li>${escapeHtml(item)}</li>`).join('');
}

function updateViewButtons() {
  els.viewTabs.querySelectorAll('.view-tab').forEach((button) => {
    button.classList.toggle('active', button.dataset.view === state.view);
  });
}

function scenarioN(scenarioId) {
  return scenarioId.includes('N5') ? 'N5' : 'N4';
}

function inferTopology(scenarioId) {
  if (scenarioId.includes('star')) return 'star';
  if (scenarioId.includes('cycle')) return 'cycle';
  return 'path';
}

function normalizeTopology(raw) {
  const value = String(raw || '').toLowerCase();
  if (value.includes('star')) return 'star';
  if (value.includes('cycle')) return 'cycle';
  return 'path';
}

function number(value) {
  return Number.parseFloat(value);
}

function fmt(value) {
  if (!Number.isFinite(value)) return '-';
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 1e-3)) {
    return value.toExponential(2);
  }
  return value.toFixed(3);
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}
