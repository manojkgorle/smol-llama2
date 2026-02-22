/**
 * attribution.js — Logit Attribution tab (logit lens + DLA).
 *
 * Sections:
 *   1. Token Strip — color-coded by final_prob; click to select for DLA
 *   2. Next Token Prediction — top-10 pills
 *   3. Logit Lens Table — per-depth top prediction, prob, target logit
 *   4. Layer Contributions — bar chart of marginal contribution per layer
 *   5. DLA Heatmap — 8x6 head contributions + FFN bar chart
 */

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
var _attrData = null;       // cached /api/attribution response
var _attrPrompt = '';       // current prompt text
var _selectedPos = null;    // currently selected token position for DLA

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function initAttribution() {
    var controls = document.getElementById('attribution-controls');
    if (!controls) return;

    createPromptInput('attribution-controls', function (prompt) {
        _attrPrompt = prompt;
        analyzeAttribution(prompt);
    }, { placeholder: 'Enter a prompt for logit attribution...', buttonLabel: 'Analyze' });

    // Build detail area containers
    var detail = document.getElementById('attribution-detail');
    if (!detail) return;

    detail.innerHTML =
        '<div class="chart-card full-width" id="attr-predictions-card">' +
            '<div class="chart-title">Next Token Prediction</div>' +
            '<div id="attr-predictions"></div>' +
        '</div>' +

        '<div class="chart-card full-width" id="attr-logit-lens-card">' +
            '<div class="chart-title">Logit Lens</div>' +
            '<div id="attr-logit-lens"></div>' +
        '</div>' +

        '<div class="chart-card full-width" id="attr-contrib-card">' +
            '<div class="chart-title">Layer Contributions</div>' +
            '<div class="chart-container" id="attr-contrib-chart"></div>' +
        '</div>' +

        '<div class="chart-card full-width" id="attr-dla-card">' +
            '<div class="chart-title">Direct Logit Attribution — click a token above</div>' +
            '<div class="chart-container" id="attr-dla-heatmap"></div>' +
            '<div class="chart-container" id="attr-dla-ffn"></div>' +
        '</div>';
}

// ---------------------------------------------------------------------------
// Fetch
// ---------------------------------------------------------------------------
async function analyzeAttribution(prompt) {
    showLoading('attribution-detail');
    try {
        var data = await apiFetch('/api/attribution', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, top_k: 10 }),
        });
        _attrData = data;
        _selectedPos = null;
        _renderTokenStrip(data);
        _renderPredictions(data.next_token_predictions);

        // Auto-select last valid position
        if (data.positions && data.positions.length > 0) {
            var lastPos = data.positions[data.positions.length - 1];
            _selectToken(lastPos.position);
        }
    } catch (err) {
        showError('attribution-detail', err.message);
    } finally {
        hideLoading('attribution-detail');
    }
}

// ---------------------------------------------------------------------------
// Token strip
// ---------------------------------------------------------------------------
function _renderTokenStrip(data) {
    var strip = document.getElementById('attribution-token-strip');
    if (!strip) return;
    strip.innerHTML = '';

    // positions array covers indices 0..(T-2), each with final_prob for that position
    var posMap = {};
    if (data.positions) {
        data.positions.forEach(function (p) { posMap[p.position] = p; });
    }

    data.tokens.forEach(function (tok, idx) {
        var span = document.createElement('span');
        span.className = 'token-span';
        span.textContent = tok;
        span.dataset.pos = idx;

        // Color by final_prob (green = high, red = low)
        var pInfo = posMap[idx];
        if (pInfo) {
            var prob = pInfo.final_prob;
            var r = Math.round(255 * (1 - prob));
            var g = Math.round(200 * prob);
            var b = 60;
            span.style.background = 'rgba(' + r + ',' + g + ',' + b + ',0.55)';
            span.title = 'pos ' + idx + '  P(target)=' + prob.toFixed(4);

            span.addEventListener('click', function () {
                _selectToken(idx);
            });
        } else {
            // Last token has no target — dimmed
            span.style.background = 'rgba(255,255,255,0.04)';
            span.style.cursor = 'default';
            span.title = 'pos ' + idx + ' (no target)';
        }

        strip.appendChild(span);
    });
}

function _selectToken(pos) {
    _selectedPos = pos;

    // Highlight in strip
    var strip = document.getElementById('attribution-token-strip');
    if (strip) {
        strip.querySelectorAll('.token-span').forEach(function (el) {
            el.classList.toggle('selected', parseInt(el.dataset.pos, 10) === pos);
        });
    }

    // Update logit lens + layer contributions for this position
    var posData = null;
    if (_attrData && _attrData.positions) {
        _attrData.positions.forEach(function (p) {
            if (p.position === pos) posData = p;
        });
    }
    if (posData) {
        _renderLogitLens(posData);
        _renderLayerContributions(posData);
    }

    // Fetch DLA for this position
    _fetchDLA(pos);
}

// ---------------------------------------------------------------------------
// Next Token Prediction
// ---------------------------------------------------------------------------
function _renderPredictions(predictions) {
    var container = document.getElementById('attr-predictions');
    if (!container || !predictions || predictions.length === 0) return;

    var html = '<div class="prediction-display">' +
        '<div class="prediction-title">Next Token Predictions</div>';

    if (predictions.length > 0) {
        var top = predictions[0];
        html += '<div class="prediction-main">' +
            '<span class="prediction-label">Top:</span>' +
            '<span class="prediction-token-main">' + escapeHtml(top.token) + '</span>' +
            '<span class="prediction-prob-main">' + (top.prob * 100).toFixed(1) + '%</span>' +
        '</div>';
    }

    if (predictions.length > 1) {
        html += '<div class="prediction-runners">';
        for (var i = 1; i < predictions.length; i++) {
            var p = predictions[i];
            html += '<span class="prediction-pill">' +
                '<span class="prediction-pill-token">' + escapeHtml(p.token) + '</span>' +
                '<span class="prediction-pill-prob">' + (p.prob * 100).toFixed(1) + '%</span>' +
            '</span>';
        }
        html += '</div>';
    }

    html += '</div>';
    container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Logit Lens Table
// ---------------------------------------------------------------------------
function _renderLogitLens(posData) {
    var container = document.getElementById('attr-logit-lens');
    if (!container) return;

    var SHOW_K = 5;

    var depths = ['embedding'];
    for (var i = 0; i < window.N_LAYERS; i++) depths.push(String(i));

    var depthLabels = ['Embedding'];
    for (var j = 0; j < window.N_LAYERS; j++) depthLabels.push('Layer ' + j);

    var target = posData.target;

    var html = '<table class="logit-lens-table">' +
        '<thead><tr>' +
            '<th>Depth</th><th>Top-' + SHOW_K + ' Predictions</th><th>Target Logit</th>' +
        '</tr></thead><tbody>';

    var prevLogit = null;

    for (var d = 0; d < depths.length; d++) {
        var key = depths[d];
        var preds = posData.cumulative_predictions[key];
        var contrib = posData.layer_contributions[key];

        // Cumulative target logit (sum of contributions up to this depth)
        if (prevLogit === null) {
            prevLogit = contrib;
        } else {
            prevLogit = prevLogit + contrib;
        }

        // Build prediction pills for top-K
        var pillsHtml = '';
        var k = preds ? Math.min(preds.length, SHOW_K) : 0;
        for (var p = 0; p < k; p++) {
            var pred = preds[p];
            var isTarget = pred.token === target;
            var isTop = p === 0;
            var cls = 'lens-pill';
            if (isTarget) cls += ' lens-pill-target';
            if (isTop) cls += ' lens-pill-top';
            pillsHtml +=
                '<span class="' + cls + '">' +
                    '<span class="lens-pill-token">' + escapeHtml(pred.token) + '</span>' +
                    '<span class="lens-pill-prob">' + (pred.prob * 100).toFixed(1) + '%</span>' +
                '</span>';
        }
        if (k === 0) pillsHtml = '<span class="lens-pill"><span class="lens-pill-token">?</span></span>';

        html += '<tr>' +
            '<td>' + escapeHtml(depthLabels[d]) + '</td>' +
            '<td class="lens-preds-cell"><div class="lens-preds-wrap">' + pillsHtml + '</div></td>' +
            '<td>' + prevLogit.toFixed(3) + '</td>' +
        '</tr>';
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Layer Contributions bar chart
// ---------------------------------------------------------------------------
function _renderLayerContributions(posData) {
    var lc = posData.layer_contributions;
    if (!lc) return;

    var labels = ['Emb'];
    var values = [lc['embedding'] || 0];
    var colors = ['#71717a'];

    for (var i = 0; i < window.N_LAYERS; i++) {
        labels.push('L' + i);
        values.push(lc[String(i)] || 0);
        colors.push(window.LAYER_COLORS[i]);
    }

    Plotly.react('attr-contrib-chart', [{
        type: 'bar',
        x: labels,
        y: values,
        marker: { color: colors },
        hovertemplate: '%{x}<br>Contribution: %{y:.4f}<extra></extra>',
    }], darkLayout({
        title: {
            text: 'Layer Contributions to target "' + escapeHtml(posData.target) + '" at pos ' + posData.position,
            font: { size: 14 },
        },
        xaxis: { title: 'Layer', gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Marginal Logit Contribution', gridcolor: 'rgba(255,255,255,0.06)' },
        margin: { t: 44, b: 48 },
    }), window.PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// DLA (Direct Logit Attribution)
// ---------------------------------------------------------------------------
async function _fetchDLA(pos) {
    if (!_attrPrompt) return;

    var cardTitle = document.querySelector('#attr-dla-card .chart-title');
    if (cardTitle) {
        cardTitle.textContent = 'Direct Logit Attribution — loading pos ' + pos + '...';
    }

    showInlineLoading('attr-dla-heatmap');

    try {
        var data = await apiFetch('/api/dla', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: _attrPrompt, position: pos }),
        });
        _renderDLA(data);
    } catch (err) {
        showError('attr-dla-heatmap', err.message);
    }
}

function _renderDLA(data) {
    var cardTitle = document.querySelector('#attr-dla-card .chart-title');
    if (cardTitle) {
        cardTitle.textContent = 'Direct Logit Attribution — pos ' + data.position +
            ' → "' + data.target + '" (P=' + data.final_prob.toFixed(4) + ')';
    }

    // Head contributions heatmap: 8 layers x 6 heads
    var zValues = [];
    var yLabels = [];
    var xLabels = [];

    for (var h = 0; h < window.N_HEADS; h++) {
        xLabels.push('H' + h);
    }
    for (var l = 0; l < window.N_LAYERS; l++) {
        yLabels.push('L' + l);
        zValues.push(data.head_contributions[l] || []);
    }

    Plotly.react('attr-dla-heatmap', [{
        type: 'heatmap',
        z: zValues,
        x: xLabels,
        y: yLabels,
        colorscale: [
            [0, '#f87171'],
            [0.5, '#0a0a0f'],
            [1, '#34d399'],
        ],
        zmid: 0,
        colorbar: { title: { text: 'Logit', side: 'right' }, tickfont: { size: 11 } },
        hovertemplate: '%{y} %{x}<br>Contribution: %{z:.4f}<extra></extra>',
    }], darkLayout({
        title: { text: 'Per-Head Logit Attribution (layers x heads)', font: { size: 14 } },
        xaxis: { title: 'Head', side: 'bottom', gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Layer', autorange: 'reversed', gridcolor: 'rgba(255,255,255,0.06)' },
        margin: { t: 44, b: 48, l: 56 },
    }), window.PLOTLY_CONFIG);

    // FFN contributions bar chart
    var ffnLabels = [];
    var ffnValues = [];
    var ffnColors = [];
    for (var fi = 0; fi < window.N_LAYERS; fi++) {
        ffnLabels.push('L' + fi + ' FFN');
        ffnValues.push(data.ffn_contributions[fi] || 0);
        ffnColors.push(window.LAYER_COLORS[fi]);
    }

    Plotly.react('attr-dla-ffn', [{
        type: 'bar',
        x: ffnLabels,
        y: ffnValues,
        marker: { color: ffnColors },
        hovertemplate: '%{x}<br>Contribution: %{y:.4f}<extra></extra>',
    }], darkLayout({
        title: { text: 'Per-FFN Logit Attribution', font: { size: 14 } },
        xaxis: { title: 'Layer', gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Logit Contribution', gridcolor: 'rgba(255,255,255,0.06)' },
        margin: { t: 44, b: 48 },
    }), window.PLOTLY_CONFIG);
}
