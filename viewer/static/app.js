const state = {
  exp: null,
  epochs: [],
  plates: [],
  data: {},
  epochIndex: 0,
  charts: { train: null, val: null, detail: null },
  metricsOpen: true,
};

const $ = id => document.getElementById(id);

// ── Tab switching ─────────────────────────────────────────────────────────────

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    $('view-results').classList.toggle('hidden', tab !== 'results');
    $('results-controls').classList.toggle('hidden', tab !== 'results');
    $('view-infer').classList.toggle('hidden', tab !== 'infer');

    if (tab === 'infer') loadInferExperiments();
  });
});

// ── Helpers ──────────────────────────────────────────────────────────────────

function imgUrl(exp, epoch, filename) {
  return `/experiments/${exp}/results/${epoch}/${filename}`;
}

function getEpochData(epochIndex) {
  const epoch = state.epochs[epochIndex];
  return { epoch, plates: state.data[epoch] || {} };
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

async function loadExperiments() {
  const res = await fetch('/api/experiments');
  const { experiments } = await res.json();
  const sel = $('exp-select');
  experiments.forEach(name => {
    const opt = document.createElement('option');
    opt.value = opt.textContent = name;
    sel.appendChild(opt);
  });
}

// ── Experiment selection ──────────────────────────────────────────────────────

async function selectExperiment(exp) {
  if (!exp) return;

  $('loading').classList.remove('hidden');
  $('grid').innerHTML = '';
  $('epoch-bar').classList.add('hidden');

  const res = await fetch(`/api/experiments/${exp}/info`);
  if (!res.ok) {
    $('loading').textContent = 'Erreur lors du chargement.';
    return;
  }

  const { epochs, plates, data } = await res.json();
  state.exp = exp;
  state.epochs = epochs;
  state.plates = plates;
  state.data = data;
  state.epochIndex = 0;

  const slider = $('epoch-slider');
  slider.min = 0;
  slider.max = epochs.length - 1;
  slider.value = 0;

  $('loading').classList.add('hidden');
  $('epoch-bar').classList.remove('hidden');

  buildGrid();
  updateEpochLabel();
  prefetch(0);
  loadMetrics(exp);
}

// ── Grid ─────────────────────────────────────────────────────────────────────

function buildGrid() {
  const grid = $('grid');
  grid.innerHTML = '';
  const { epoch, plates } = getEpochData(state.epochIndex);

  state.plates.forEach(plateId => {
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.dataset.plate = plateId;

    const img = document.createElement('img');
    const entry = plates[plateId];
    if (entry && entry.sr) {
      img.src = imgUrl(state.exp, epoch, entry.sr);
    } else {
      img.src = '';
      img.classList.add('missing');
    }
    img.alt = `Plate ${plateId}`;
    img.loading = 'lazy';

    const label = document.createElement('span');
    label.textContent = `#${plateId}`;

    cell.appendChild(img);
    cell.appendChild(label);
    cell.addEventListener('click', () => openModal(plateId));
    grid.appendChild(cell);
  });
}

function updateGrid(epochIndex) {
  const { epoch, plates } = getEpochData(epochIndex);
  document.querySelectorAll('.cell').forEach(cell => {
    const plateId = cell.dataset.plate;
    const img = cell.querySelector('img');
    const entry = plates[plateId];
    if (entry && entry.sr) {
      img.src = imgUrl(state.exp, epoch, entry.sr);
      img.classList.remove('missing');
    } else {
      img.src = '';
      img.classList.add('missing');
    }
  });
}

// ── Epoch slider ──────────────────────────────────────────────────────────────

function updateEpochLabel() {
  const { epoch, plates } = getEpochData(state.epochIndex);
  $('epoch-label').textContent = `Epoch ${epoch}`;
  const firstPlate = Object.values(plates)[0];
  $('iter-label').textContent = firstPlate ? `iter ${firstPlate.iters}` : '';
}

$('epoch-slider').addEventListener('input', e => {
  const idx = parseInt(e.target.value, 10);
  state.epochIndex = idx;
  updateGrid(idx);
  updateEpochLabel();
  prefetch(idx);
  updateChartMarker(idx);
});

// ── Prefetch ──────────────────────────────────────────────────────────────────

function prefetch(centerIdx) {
  [-1, 1].forEach(delta => {
    const idx = centerIdx + delta;
    if (idx < 0 || idx >= state.epochs.length) return;
    const { epoch, plates } = getEpochData(idx);
    state.plates.forEach(plateId => {
      const entry = plates[plateId];
      if (entry && entry.sr) {
        new Image().src = imgUrl(state.exp, epoch, entry.sr);
      }
    });
  });
}

// ── Modal ─────────────────────────────────────────────────────────────────────

function openModal(plateId) {
  const { epoch, plates } = getEpochData(state.epochIndex);
  const entry = plates[plateId];

  $('modal-title').textContent = `Plaque #${plateId} — Epoch ${epoch}`;

  const container = $('modal-images');
  container.innerHTML = '';

  const types = [
    { key: 'hr',  label: 'HR (Ground Truth)' },
    { key: 'lr1', label: 'LR 1' },
    { key: 'lr2', label: 'LR 2' },
    { key: 'lr3', label: 'LR 3' },
    { key: 'sr',  label: 'SR (Predicted)' },
  ];

  types.forEach(({ key, label }) => {
    const wrap = document.createElement('div');
    wrap.className = 'modal-img-wrap';

    const lbl = document.createElement('div');
    lbl.className = 'modal-img-label';
    lbl.textContent = label;

    const img = document.createElement('img');
    if (entry && entry[key]) {
      img.src = imgUrl(state.exp, epoch, entry[key]);
    } else {
      img.src = '';
      img.classList.add('missing');
    }
    img.alt = label;

    wrap.appendChild(lbl);
    wrap.appendChild(img);
    container.appendChild(wrap);
  });

  $('modal').classList.remove('hidden');
}

function closeModal() {
  $('modal').classList.add('hidden');
}

$('modal-close').addEventListener('click', closeModal);
$('modal-backdrop').addEventListener('click', closeModal);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

// ── Metrics & Charts ──────────────────────────────────────────────────────────

const CHART_DEFAULTS = {
  responsive: true,
  animation: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { labels: { color: '#ccc', boxWidth: 12, font: { size: 11 } } },
    tooltip: { backgroundColor: '#222', titleColor: '#fff', bodyColor: '#ccc' },
  },
  scales: {
    x: { ticks: { color: '#888', maxTicksLimit: 10 }, grid: { color: '#222' } },
  },
};

function yScale(id, label, position = 'left') {
  return {
    [id]: {
      position,
      title: { display: true, text: label, color: '#888', font: { size: 11 } },
      ticks: { color: '#888' },
      grid: { color: position === 'left' ? '#222' : 'transparent' },
    },
  };
}

function makeMarkerPlugin(getEpoch) {
  return {
    id: 'epochMarker',
    afterDraw(chart) {
      const epoch = getEpoch();
      if (epoch == null) return;
      const meta = chart.getDatasetMeta(0);
      if (!meta || !meta.data.length) return;
      const pt = meta.data.find((_p, i) => chart.data.labels[i] === epoch);
      if (!pt) return;
      const { ctx, chartArea: { top, bottom } } = chart;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(pt.x, top);
      ctx.lineTo(pt.x, bottom);
      ctx.strokeStyle = 'rgba(74,158,255,0.6)';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.restore();
    },
  };
}

function destroyCharts() {
  ['train', 'val', 'detail'].forEach(k => {
    if (state.charts[k]) { state.charts[k].destroy(); state.charts[k] = null; }
  });
}

function currentEpoch() {
  return state.epochs[state.epochIndex] ? parseInt(state.epochs[state.epochIndex]) : null;
}

function buildCharts(metrics) {
  destroyCharts();

  const markerPlugin = makeMarkerPlugin(currentEpoch);

  // ── Train loss ──
  if (metrics.train.length) {
    const labels = metrics.train.map(r => r.epoch);
    state.charts.train = new Chart($('chart-train'), {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'avg_train_loss',
          data: metrics.train.map(r => r.avg_train_loss),
          borderColor: '#f97316', backgroundColor: 'transparent',
          pointRadius: 2, borderWidth: 1.5, yAxisID: 'y',
        }],
      },
      options: {
        ...CHART_DEFAULTS,
        scales: { x: CHART_DEFAULTS.scales.x, ...yScale('y', 'Loss') },
      },
      plugins: [markerPlugin],
    });
  }

  // ── Val PSNR + loss ──
  if (metrics.val.length) {
    const labels = metrics.val.map(r => r.epoch);
    state.charts.val = new Chart($('chart-val'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'PSNR',
            data: metrics.val.map(r => r.psnr),
            borderColor: '#4a9eff', backgroundColor: 'transparent',
            pointRadius: 3, borderWidth: 1.5, yAxisID: 'yPsnr',
          },
          {
            label: 'val_loss',
            data: metrics.val.map(r => r.loss),
            borderColor: '#a78bfa', backgroundColor: 'transparent',
            pointRadius: 3, borderWidth: 1.5, yAxisID: 'yLoss',
          },
        ],
      },
      options: {
        ...CHART_DEFAULTS,
        scales: {
          x: CHART_DEFAULTS.scales.x,
          ...yScale('yPsnr', 'PSNR (dB)', 'left'),
          ...yScale('yLoss', 'Loss', 'right'),
        },
      },
      plugins: [markerPlugin],
    });
  }

  // ── Detailed train losses ──
  if (metrics.train_detail.length) {
    const labels = metrics.train_detail.map(r => r.epoch);
    state.charts.detail = new Chart($('chart-detail'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: 'l_pix',       data: metrics.train_detail.map(r => r.l_pix),       borderColor: '#34d399', backgroundColor: 'transparent', pointRadius: 1, borderWidth: 1.5, yAxisID: 'y' },
          { label: 'l_diffusion', data: metrics.train_detail.map(r => r.l_diffusion), borderColor: '#fb7185', backgroundColor: 'transparent', pointRadius: 1, borderWidth: 1.5, yAxisID: 'y' },
          { label: 'l_mta',       data: metrics.train_detail.map(r => r.l_mta),       borderColor: '#fbbf24', backgroundColor: 'transparent', pointRadius: 1, borderWidth: 1.5, yAxisID: 'y' },
        ],
      },
      options: {
        ...CHART_DEFAULTS,
        scales: { x: CHART_DEFAULTS.scales.x, ...yScale('y', 'Loss') },
      },
      plugins: [markerPlugin],
    });
  }
}

function updateChartMarker() {
  ['train', 'val', 'detail'].forEach(k => {
    if (state.charts[k]) state.charts[k].update();
  });
}

async function loadMetrics(exp) {
  const res = await fetch(`/api/experiments/${exp}/metrics`);
  if (!res.ok) return;
  const metrics = await res.json();
  $('metrics-section').classList.remove('hidden');
  buildCharts(metrics);
}

// ── Metrics toggle ────────────────────────────────────────────────────────────

$('metrics-toggle').addEventListener('click', () => {
  state.metricsOpen = !state.metricsOpen;
  $('metrics-body').classList.toggle('hidden', !state.metricsOpen);
  $('metrics-chevron').textContent = state.metricsOpen ? '▼' : '▶';
});

// ── Init (results tab) ────────────────────────────────────────────────────────

$('exp-select').addEventListener('change', e => selectExperiment(e.target.value));

loadExperiments();


// ═══════════════════════════════════════════════════════════════════════════════
// INFERENCE TAB
// ═══════════════════════════════════════════════════════════════════════════════

const inferState = {
  files: [],   // File objects chosen by user (up to 3)
  running: false,
};

// ── Experiment list ───────────────────────────────────────────────────────────

async function loadInferExperiments() {
  const sel = $('infer-exp-select');
  const current = sel.value;
  // Don't reload if already populated
  if (sel.options.length > 1) return;

  const res = await fetch('/api/infer/experiments');
  const { experiments } = await res.json();
  experiments.forEach(name => {
    const opt = document.createElement('option');
    opt.value = opt.textContent = name;
    sel.appendChild(opt);
  });

  if (current && [...sel.options].some(o => o.value === current)) {
    sel.value = current;
  }
}

$('infer-exp-select').addEventListener('change', async e => {
  const exp = e.target.value;
  $('infer-ckpt-label').textContent = '';
  if (!exp) { updateRunBtn(); return; }

  const res = await fetch(`/api/infer/status/${exp}`);
  const info = await res.json();
  const label = $('infer-ckpt-label');
  if (info.loaded) {
    label.textContent = 'Modèle chargé';
    label.className = 'infer-ckpt-label loaded';
  } else if (info.has_checkpoint) {
    label.textContent = info.checkpoint;
    label.className = 'infer-ckpt-label';
  } else {
    label.textContent = 'Aucun checkpoint trouvé';
    label.className = 'infer-ckpt-label error';
  }
  updateRunBtn();
});

// ── File picking ──────────────────────────────────────────────────────────────

$('browse-btn').addEventListener('click', () => $('file-input').click());

$('file-input').addEventListener('change', e => {
  handleFiles([...e.target.files]);
  e.target.value = '';  // reset so same file can be re-added
});

const dropZone = $('drop-zone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles([...e.dataTransfer.files]);
});

function handleFiles(newFiles) {
  const imageFiles = newFiles.filter(f => f.type.startsWith('image/'));
  const combined = [...inferState.files, ...imageFiles];

  $('file-error').classList.add('hidden');

  if (combined.length > 3) {
    $('file-error').textContent = `Trop d'images : ${combined.length} sélectionnées, maximum 3.`;
    $('file-error').classList.remove('hidden');
    // Keep only up to 3
    inferState.files = combined.slice(0, 3);
  } else {
    inferState.files = combined;
  }

  renderThumbs();
  updateRunBtn();
}

function renderThumbs() {
  const thumbsEl = $('thumbs');
  thumbsEl.innerHTML = '';
  inferState.files.forEach((file, idx) => {
    const wrap = document.createElement('div');
    wrap.className = 'thumb-wrap';

    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.alt = file.name;

    const remove = document.createElement('button');
    remove.className = 'thumb-remove';
    remove.textContent = '✕';
    remove.title = 'Supprimer';
    remove.addEventListener('click', () => {
      inferState.files.splice(idx, 1);
      $('file-error').classList.add('hidden');
      renderThumbs();
      updateRunBtn();
    });

    const label = document.createElement('span');
    label.className = 'thumb-label';
    label.textContent = `LR ${idx + 1}`;

    wrap.appendChild(img);
    wrap.appendChild(remove);
    wrap.appendChild(label);
    thumbsEl.appendChild(wrap);
  });

  const row = $('thumb-row');
  if (inferState.files.length > 0) {
    row.classList.remove('hidden');
    $('frame-count').textContent = `${inferState.files.length}/3 image${inferState.files.length > 1 ? 's' : ''}`;
    if (inferState.files.length < 3) {
      $('frame-count').title = 'Les images manquantes seront complétées par duplication';
    }
  } else {
    row.classList.add('hidden');
  }
}

function updateRunBtn() {
  const exp = $('infer-exp-select').value;
  const hasFiles = inferState.files.length > 0;
  $('run-btn').disabled = !exp || !hasFiles || inferState.running;
}

// ── Run inference ─────────────────────────────────────────────────────────────

$('run-btn').addEventListener('click', runInference);

async function runInference() {
  const exp = $('infer-exp-select').value;
  if (!exp || inferState.files.length === 0) return;

  inferState.running = true;
  updateRunBtn();

  const statusEl = $('infer-status');
  statusEl.textContent = 'Chargement du modèle et inférence en cours…';
  statusEl.className = 'infer-status';

  $('infer-results').classList.add('hidden');

  const form = new FormData();
  inferState.files.forEach(f => form.append('files', f));

  try {
    const res = await fetch(`/api/infer/${exp}`, { method: 'POST', body: form });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    const data = await res.json();
    showInferResults(data);
    statusEl.textContent = 'Inférence terminée.';
    statusEl.className = 'infer-status ok';

    // Update the ckpt label to show "Modèle chargé"
    const lbl = $('infer-ckpt-label');
    lbl.textContent = 'Modèle chargé';
    lbl.className = 'infer-ckpt-label loaded';

  } catch (err) {
    statusEl.textContent = `Erreur : ${err.message}`;
    statusEl.className = 'infer-status error';
  } finally {
    inferState.running = false;
    updateRunBtn();
  }
}

function showInferResults(data) {
  const row = $('infer-results-row');
  row.innerHTML = '';

  const items = [
    ...data.lr_inputs.map((b64, i) => ({
      label: `LR ${i + 1}${i >= data.n_frames ? ' (dupliquée)' : ''}`,
      b64,
    })),
    { label: 'MTA (condition)', b64: data.condition },
    { label: 'SR — Résultat', b64: data.sr, highlight: true },
  ];

  items.forEach(({ label, b64, highlight }) => {
    const wrap = document.createElement('div');
    wrap.className = 'infer-img-wrap' + (highlight ? ' highlight' : '');

    const lbl = document.createElement('div');
    lbl.className = 'infer-img-label';
    lbl.textContent = label;

    const img = document.createElement('img');
    img.src = `data:image/png;base64,${b64}`;
    img.alt = label;

    wrap.appendChild(lbl);
    wrap.appendChild(img);
    row.appendChild(wrap);
  });

  $('infer-results').classList.remove('hidden');
}
