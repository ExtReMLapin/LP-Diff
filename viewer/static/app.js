const state = {
  exp: null,
  epochs: [],
  plates: [],
  data: {},
  epochIndex: 0,
};

const $ = id => document.getElementById(id);

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
  // find a sample iters value
  const firstPlate = Object.values(plates)[0];
  $('iter-label').textContent = firstPlate ? `iter ${firstPlate.iters}` : '';
}

$('epoch-slider').addEventListener('input', e => {
  const idx = parseInt(e.target.value, 10);
  state.epochIndex = idx;
  updateGrid(idx);
  updateEpochLabel();
  prefetch(idx);
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

// ── Init ──────────────────────────────────────────────────────────────────────

$('exp-select').addEventListener('change', e => selectExperiment(e.target.value));

loadExperiments();
