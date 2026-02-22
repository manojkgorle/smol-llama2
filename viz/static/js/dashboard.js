/**
 * dashboard.js — Training Dashboard tab for LLaMA-2 interpretability.
 *
 * Shows real-time training metrics: loss curves, learning rate schedule,
 * gradient norms, per-layer residual stream norms, and SwiGLU gate sparsity.
 *
 * Relies on shared utilities from app.js:
 *   apiFetch, darkLayout, showInlineLoading, showError, hideLoading,
 *   window.LAYER_COLORS, window.N_LAYERS, window.PLOTLY_CONFIG
 */

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
var _dashboardState = {
    steps: [],
    validations: [],
    initialized: false,
};

// ---------------------------------------------------------------------------
// initDashboard — called once on first tab visit
// ---------------------------------------------------------------------------
function initDashboard() {
    if (_dashboardState.initialized) return;
    _dashboardState.initialized = true;

    _createStatCards();
    _createChartContainers();
    _fetchAllMetrics();
}

// ---------------------------------------------------------------------------
// Stat cards
// ---------------------------------------------------------------------------
function _createStatCards() {
    var container = document.getElementById('dashboard-stats');
    if (!container) return;

    var cards = [
        { id: 'stat-loss',     label: 'Loss',          value: '--',  sub: 'Train loss',
          help: 'Cross-entropy loss measures how far the model\'s predictions are from the true next tokens. <strong>Lower is better</strong>. A loss near 0 means the model is very confident in the correct answer.' },
        { id: 'stat-ppl',      label: 'Perplexity',    value: '--',  sub: 'exp(loss)',
          help: '<strong>Perplexity = e^loss</strong>. Intuitively, it\'s the number of equally likely tokens the model is choosing between. A perplexity of 10 means the model is as uncertain as choosing between 10 options.' },
        { id: 'stat-lr',       label: 'Learning Rate',  value: '--',  sub: 'Current LR',
          help: 'Controls how large each gradient update step is. Typically follows a <strong>warmup + cosine decay</strong> schedule. Too high = unstable training. Too low = slow learning.' },
        { id: 'stat-step',     label: 'Step',           value: '--',  sub: 'Training step' },
        { id: 'stat-grad',     label: 'Grad Norm',      value: '--',  sub: 'Global gradient norm',
          help: '<strong>Gradient norm</strong> measures the magnitude of the update signal. Spikes may indicate the model encountering unusual data. Gradient clipping caps this value to prevent training instability.' },
        { id: 'stat-val-loss', label: 'Val Loss',       value: '--',  sub: 'Latest validation' },
    ];

    cards.forEach(function (c) {
        var card = document.createElement('div');
        card.className = 'stat-card';
        card.id = c.id;

        var labelDiv = document.createElement('div');
        labelDiv.className = 'stat-label';
        labelDiv.textContent = c.label;
        if (c.help) {
            labelDiv.appendChild(createHelpIcon(c.label, c.help));
        }
        card.appendChild(labelDiv);

        card.innerHTML +=
            '<div class="stat-value">' + escapeHtml(c.value) + '</div>' +
            '<div class="stat-sub">' + escapeHtml(c.sub) + '</div>';
        container.appendChild(card);
    });
}

// ---------------------------------------------------------------------------
// Chart containers
// ---------------------------------------------------------------------------
function _createChartContainers() {
    var grid = document.getElementById('dashboard-charts');
    if (!grid) return;
    grid.classList.add('two-col');

    var charts = [
        { id: 'chart-loss',        title: 'Loss & Perplexity',           fullWidth: true },
        { id: 'chart-lr',          title: 'Learning Rate Schedule',      fullWidth: false },
        { id: 'chart-grad',        title: 'Gradient Norm',               fullWidth: false },
        { id: 'chart-residual',    title: 'Per-Layer Residual Stream Norms', fullWidth: true },
        { id: 'chart-gate',        title: 'Per-Layer SwiGLU Gate Sparsity',  fullWidth: true },
    ];

    charts.forEach(function (c) {
        var card = document.createElement('div');
        card.className = 'chart-card' + (c.fullWidth ? ' full-width' : '');
        card.innerHTML =
            '<div class="chart-title">' + escapeHtml(c.title) + '</div>' +
            '<div class="chart-container" id="' + c.id + '"></div>';
        grid.appendChild(card);
    });
}

// ---------------------------------------------------------------------------
// Fetch all historical metrics
// ---------------------------------------------------------------------------
function _fetchAllMetrics() {
    showInlineLoading('dashboard-charts');

    apiFetch('/api/metrics/all')
        .then(function (data) {
            _dashboardState.steps = data.steps || [];
            _dashboardState.validations = data.validations || [];
            _renderAll();
        })
        .catch(function (err) {
            showError('dashboard-charts', 'Failed to load metrics: ' + err.message);
        });
}

// ---------------------------------------------------------------------------
// WebSocket handlers
// ---------------------------------------------------------------------------
function handleStepUpdate(data) {
    _dashboardState.steps.push(data);
    _updateStatCards(data);
    _renderAll();
}

function handleValUpdate(data) {
    _dashboardState.validations.push(data);
    _updateValStatCard(data);
    _renderAll();
}

// ---------------------------------------------------------------------------
// Update stat cards
// ---------------------------------------------------------------------------
function _updateStatCards(latest) {
    _setStatValue('stat-loss', _fmtNum(latest.loss, 4));
    _setStatValue('stat-ppl', _fmtNum(latest.perplexity, 2));
    _setStatValue('stat-lr', _fmtSci(latest.learning_rate));
    _setStatValue('stat-step', String(latest.step));
    _setStatValue('stat-grad', _fmtNum(latest.grad_norm, 4));
}

function _updateValStatCard(latest) {
    _setStatValue('stat-val-loss', _fmtNum(latest.val_loss, 4));
}

function _setStatValue(cardId, value) {
    var card = document.getElementById(cardId);
    if (!card) return;
    var el = card.querySelector('.stat-value');
    if (el) el.textContent = value;
}

// ---------------------------------------------------------------------------
// Render all charts
// ---------------------------------------------------------------------------
function _renderAll() {
    var steps = _dashboardState.steps;
    var vals = _dashboardState.validations;

    // Re-create chart containers if they were replaced by inline loading
    if (!document.getElementById('chart-loss')) {
        var grid = document.getElementById('dashboard-charts');
        if (grid) grid.innerHTML = '';
        _createChartContainers();
    }

    if (steps.length === 0) {
        _showPlaceholder('chart-loss', 'No training metrics yet. Start training to see live charts.');
        _showPlaceholder('chart-lr', 'Waiting for data...');
        _showPlaceholder('chart-grad', 'Waiting for data...');
        _showPlaceholder('chart-residual', 'Waiting for data...');
        _showPlaceholder('chart-gate', 'Waiting for data...');
        return;
    }

    // Update stat cards with latest step
    var latest = steps[steps.length - 1];
    _updateStatCards(latest);
    if (vals.length > 0) {
        _updateValStatCard(vals[vals.length - 1]);
    }

    _renderLossChart(steps, vals);
    _renderLRChart(steps);
    _renderGradChart(steps);
    _renderResidualChart(steps);
    _renderGateSparsityChart(steps);
}

// ---------------------------------------------------------------------------
// Chart 1: Loss curve + perplexity (dual y-axis)
// ---------------------------------------------------------------------------
function _renderLossChart(steps, vals) {
    var el = document.getElementById('chart-loss');
    if (!el) return;

    var xSteps = steps.map(function (s) { return s.step; });
    var trainLoss = steps.map(function (s) { return s.loss; });
    var ppl = steps.map(function (s) { return s.perplexity; });

    var traces = [
        {
            x: xSteps,
            y: trainLoss,
            name: 'Train Loss',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#f87171', width: 2 },
            yaxis: 'y',
        },
        {
            x: ppl.map(function (_, i) { return xSteps[i]; }),
            y: ppl,
            name: 'Perplexity',
            type: 'scatter',
            mode: 'lines',
            line: { color: '#fbbf24', width: 2, dash: 'dot' },
            yaxis: 'y2',
        },
    ];

    if (vals.length > 0) {
        traces.push({
            x: vals.map(function (v) { return v.step; }),
            y: vals.map(function (v) { return v.val_loss; }),
            name: 'Val Loss',
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#34d399', width: 2 },
            marker: { size: 6, color: '#34d399' },
            yaxis: 'y',
        });
    }

    var layout = darkLayout({
        xaxis: { title: 'Step', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Loss', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)' },
        yaxis2: {
            title: 'Perplexity',
            overlaying: 'y',
            side: 'right',
            gridcolor: 'rgba(255,255,255,0.03)',
            showgrid: false,
        },
        legend: {
            x: 0.01, y: 0.99,
            bgcolor: 'rgba(9,9,11,0.9)',
            bordercolor: 'rgba(255,255,255,0.06)',
            borderwidth: 1,
            font: { size: 11 },
        },
    });

    Plotly.react(el, traces, layout, window.PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Chart 2: Learning rate schedule
// ---------------------------------------------------------------------------
function _renderLRChart(steps) {
    var el = document.getElementById('chart-lr');
    if (!el) return;

    var traces = [{
        x: steps.map(function (s) { return s.step; }),
        y: steps.map(function (s) { return s.learning_rate; }),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#a78bfa', width: 2 },
        name: 'Learning Rate',
    }];

    var layout = darkLayout({
        xaxis: { title: 'Step', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)' },
        yaxis: {
            title: 'Learning Rate',
            gridcolor: 'rgba(255,255,255,0.06)',
            zerolinecolor: 'rgba(255,255,255,0.06)',
            tickformat: '.1e',
        },
        showlegend: false,
    });

    Plotly.react(el, traces, layout, window.PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Chart 3: Gradient norm
// ---------------------------------------------------------------------------
function _renderGradChart(steps) {
    var el = document.getElementById('chart-grad');
    if (!el) return;

    var traces = [{
        x: steps.map(function (s) { return s.step; }),
        y: steps.map(function (s) { return s.grad_norm; }),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#fb923c', width: 2 },
        name: 'Grad Norm',
    }];

    var layout = darkLayout({
        xaxis: { title: 'Step', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Gradient Norm', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)' },
        showlegend: false,
    });

    Plotly.react(el, traces, layout, window.PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Chart 4: Per-layer residual stream norms (8 traces)
// ---------------------------------------------------------------------------
function _renderResidualChart(steps) {
    var el = document.getElementById('chart-residual');
    if (!el) return;

    // Filter steps that have residual_norms data
    var filtered = steps.filter(function (s) {
        return s.residual_norms && s.residual_norms.length > 0;
    });

    if (filtered.length === 0) {
        _showPlaceholder('chart-residual', 'No per-layer residual norm data available.');
        return;
    }

    var nLayers = Math.min(filtered[0].residual_norms.length, window.N_LAYERS);
    var traces = [];

    for (var layer = 0; layer < nLayers; layer++) {
        (function (l) {
            traces.push({
                x: filtered.map(function (s) { return s.step; }),
                y: filtered.map(function (s) { return s.residual_norms[l]; }),
                name: 'Layer ' + l,
                type: 'scatter',
                mode: 'lines',
                line: { color: window.LAYER_COLORS[l % window.LAYER_COLORS.length], width: 1.5 },
            });
        })(layer);
    }

    var layout = darkLayout({
        xaxis: { title: 'Step', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Residual Norm', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)' },
        legend: {
            x: 1.02, y: 1,
            bgcolor: 'rgba(9,9,11,0.9)',
            bordercolor: 'rgba(255,255,255,0.06)',
            borderwidth: 1,
            font: { size: 10 },
        },
    });

    Plotly.react(el, traces, layout, window.PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Chart 5: Per-layer SwiGLU gate sparsity (8 traces)
// ---------------------------------------------------------------------------
function _renderGateSparsityChart(steps) {
    var el = document.getElementById('chart-gate');
    if (!el) return;

    // Filter steps that have ffn_gate_sparsity data
    var filtered = steps.filter(function (s) {
        return s.ffn_gate_sparsity && s.ffn_gate_sparsity.length > 0;
    });

    if (filtered.length === 0) {
        _showPlaceholder('chart-gate', 'No SwiGLU gate sparsity data available.');
        return;
    }

    var nLayers = Math.min(filtered[0].ffn_gate_sparsity.length, window.N_LAYERS);
    var traces = [];

    for (var layer = 0; layer < nLayers; layer++) {
        (function (l) {
            traces.push({
                x: filtered.map(function (s) { return s.step; }),
                y: filtered.map(function (s) { return s.ffn_gate_sparsity[l]; }),
                name: 'Layer ' + l,
                type: 'scatter',
                mode: 'lines',
                line: { color: window.LAYER_COLORS[l % window.LAYER_COLORS.length], width: 1.5 },
            });
        })(layer);
    }

    var layout = darkLayout({
        xaxis: { title: 'Step', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)' },
        yaxis: {
            title: 'Gate Sparsity (fraction near-zero)',
            gridcolor: 'rgba(255,255,255,0.06)',
            zerolinecolor: 'rgba(255,255,255,0.06)',
            range: [0, 1],
        },
        legend: {
            x: 1.02, y: 1,
            bgcolor: 'rgba(9,9,11,0.9)',
            bordercolor: 'rgba(255,255,255,0.06)',
            borderwidth: 1,
            font: { size: 10 },
        },
    });

    Plotly.react(el, traces, layout, window.PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function _showPlaceholder(containerId, message) {
    var el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '<div class="placeholder-message">' + escapeHtml(message) + '</div>';
}

function _fmtNum(val, decimals) {
    if (val === undefined || val === null) return '--';
    return Number(val).toFixed(decimals);
}

function _fmtSci(val) {
    if (val === undefined || val === null) return '--';
    return Number(val).toExponential(2);
}
