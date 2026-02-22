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
var _attrViewMode = 'logit-lens';  // 'logit-lens' or 'waterfall'

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function initAttribution() {
    var controls = document.getElementById('attribution-controls');
    if (!controls) return;

    createPromptInput('attribution-controls', function (prompt) {
        _attrPrompt = prompt;
        if (_attrViewMode === 'waterfall') {
            _analyzeWaterfall(prompt);
        } else {
            analyzeAttribution(prompt);
        }
    }, { placeholder: 'Enter a prompt for logit attribution...', buttonLabel: 'Analyze' });

    // View switcher
    var viewSwitcher = document.createElement('div');
    viewSwitcher.className = 'view-switcher';
    viewSwitcher.id = 'attr-view-switcher';

    var btnLens = document.createElement('button');
    btnLens.className = 'active';
    btnLens.textContent = 'Logit Lens';
    btnLens.dataset.view = 'logit-lens';

    var btnWaterfall = document.createElement('button');
    btnWaterfall.textContent = 'Waterfall';
    btnWaterfall.dataset.view = 'waterfall';

    viewSwitcher.appendChild(btnLens);
    viewSwitcher.appendChild(btnWaterfall);
    controls.appendChild(viewSwitcher);

    viewSwitcher.querySelectorAll('button').forEach(function (btn) {
        btn.addEventListener('click', function () {
            viewSwitcher.querySelectorAll('button').forEach(function (b) { b.classList.remove('active'); });
            btn.classList.add('active');
            _attrViewMode = btn.dataset.view;
            if (_attrPrompt) {
                if (_attrViewMode === 'waterfall') {
                    _analyzeWaterfall(_attrPrompt);
                } else {
                    analyzeAttribution(_attrPrompt);
                }
            }
        });
    });

    // Build detail area containers
    var detail = document.getElementById('attribution-detail');
    if (!detail) return;

    _buildAttributionContainers(detail);
}

// ---------------------------------------------------------------------------
// Fetch
// ---------------------------------------------------------------------------
async function analyzeAttribution(prompt) {
    // Ensure containers exist (may have been replaced by waterfall)
    var detail = document.getElementById('attribution-detail');
    if (detail && !document.getElementById('attr-predictions-card')) {
        _buildAttributionContainers(detail);
    }
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

function _buildAttributionContainers(detail) {
    detail.innerHTML = '';

    // Predictions card with help icon
    var predsCard = document.createElement('div');
    predsCard.className = 'chart-card full-width';
    predsCard.id = 'attr-predictions-card';
    var predsTitle = document.createElement('div');
    predsTitle.className = 'chart-title';
    predsTitle.textContent = 'Next Token Prediction';
    predsCard.appendChild(predsTitle);
    predsCard.innerHTML += '<div id="attr-predictions"></div>';
    detail.appendChild(predsCard);

    // Logit lens card with help icon
    var lensCard = document.createElement('div');
    lensCard.className = 'chart-card full-width';
    lensCard.id = 'attr-logit-lens-card';
    var lensTitle = document.createElement('div');
    lensTitle.className = 'chart-title';
    lensTitle.textContent = 'Logit Lens';
    lensTitle.appendChild(createHelpIcon('Logit Lens',
        'We peek at the model\'s prediction at <strong>every layer</strong> by applying the final output projection to intermediate representations. ' +
        'This reveals <strong>when and where</strong> the model "decides" on its answer. ' +
        'Highlighted pills show the target token appearing in top predictions.'
    ));
    lensCard.appendChild(lensTitle);
    lensCard.innerHTML += '<div id="attr-logit-lens"></div>';
    detail.appendChild(lensCard);

    // Layer contributions card with help icon
    var contribCard = document.createElement('div');
    contribCard.className = 'chart-card full-width';
    contribCard.id = 'attr-contrib-card';
    var contribTitle = document.createElement('div');
    contribTitle.className = 'chart-title';
    contribTitle.textContent = 'Layer Contributions';
    contribTitle.appendChild(createHelpIcon('Layer Contributions',
        'Shows the <strong>marginal contribution</strong> of each layer to the target token\'s logit. ' +
        'Positive bars push the model toward the correct answer. ' +
        'Negative bars push away from it. The embedding provides the initial signal.'
    ));
    contribCard.appendChild(contribTitle);
    contribCard.innerHTML += '<div class="chart-container" id="attr-contrib-chart"></div>';
    detail.appendChild(contribCard);

    // DLA card with help icon
    var dlaCard = document.createElement('div');
    dlaCard.className = 'chart-card full-width';
    dlaCard.id = 'attr-dla-card';
    var dlaTitle = document.createElement('div');
    dlaTitle.className = 'chart-title';
    dlaTitle.textContent = 'Direct Logit Attribution — click a token above';
    dlaTitle.appendChild(createHelpIcon('DLA',
        '<strong>Direct Logit Attribution</strong> decomposes the final logit into contributions from each attention head and FFN layer. ' +
        'It measures how much each component <strong>directly</strong> pushes toward or away from the target token, ' +
        'by projecting component outputs onto the unembedding direction.'
    ));
    dlaCard.appendChild(dlaTitle);
    dlaCard.innerHTML += '<div class="chart-container" id="attr-dla-heatmap"></div><div class="chart-container" id="attr-dla-ffn"></div>';
    detail.appendChild(dlaCard);
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

// ---------------------------------------------------------------------------
// Token Probability Waterfall
// ---------------------------------------------------------------------------
async function _analyzeWaterfall(prompt) {
    var detail = document.getElementById('attribution-detail');
    if (!detail) return;
    detail.innerHTML = '';

    var strip = document.getElementById('attribution-token-strip');
    if (strip) strip.innerHTML = '';

    // Waterfall card
    var card = document.createElement('div');
    card.className = 'chart-card full-width';
    var title = document.createElement('div');
    title.className = 'chart-title';
    title.textContent = 'Token Probability Waterfall';
    title.appendChild(createHelpIcon('Probability Waterfall',
        'Shows how confident the model is in the <strong>correct next token</strong> at each layer. ' +
        '<strong>Bright green</strong> = high confidence. <strong>Dark</strong> = low confidence. ' +
        'Sudden jumps reveal which layers are <strong>critical</strong> for making the prediction. ' +
        'Click any cell to see top-5 predictions at that depth/position.'
    ));
    card.appendChild(title);

    var chartDiv = document.createElement('div');
    chartDiv.id = 'attr-waterfall-chart';
    chartDiv.style.minHeight = '400px';
    card.appendChild(chartDiv);

    var detailDiv = document.createElement('div');
    detailDiv.id = 'attr-waterfall-detail';
    card.appendChild(detailDiv);

    detail.appendChild(card);
    showInlineLoading('attr-waterfall-chart');

    try {
        var data = await apiFetch('/api/waterfall', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt }),
        });

        var tokens = data.tokens;
        var depthLabels = data.depth_labels;
        var probMatrix = data.probability_matrix;

        // Build position labels (token[pos] -> target)
        var yLabels = [];
        for (var i = 0; i < probMatrix.length; i++) {
            yLabels.push(tokens[i] + ' → ' + tokens[i + 1]);
        }

        // Transpose for plotly: z[y][x]
        Plotly.newPlot('attr-waterfall-chart', [{
            z: probMatrix,
            x: depthLabels.map(function (d) { return d === 'embedding' ? 'Emb' : 'L' + d; }),
            y: yLabels,
            type: 'heatmap',
            colorscale: [
                [0, '#09090b'],
                [0.25, '#064e3b'],
                [0.5, '#059669'],
                [0.75, '#34d399'],
                [1, '#6ee7b7'],
            ],
            colorbar: {
                title: 'P(target)',
                tickformat: '.0%',
                tickfont: { color: '#e0e0e0' },
                titlefont: { color: '#e0e0e0' },
            },
            hovertemplate: 'Pos: %{y}<br>Depth: %{x}<br>P(target): %{z:.4f}<extra></extra>',
        }], darkLayout({
            title: { text: 'P(correct next token) by Layer Depth', font: { size: 14 } },
            xaxis: { title: 'Depth', side: 'bottom', gridcolor: 'rgba(255,255,255,0.06)' },
            yaxis: { title: 'Position', autorange: 'reversed', gridcolor: 'rgba(255,255,255,0.06)' },
            height: Math.max(400, probMatrix.length * 28 + 120),
            margin: { t: 44, b: 60, l: 120 },
        }), window.PLOTLY_CONFIG);

        // Click handler to show predictions at that cell
        var chartEl = document.getElementById('attr-waterfall-chart');
        chartEl.on('plotly_click', function (clickData) {
            if (clickData.points && clickData.points.length > 0) {
                var pt = clickData.points[0];
                var posIdx = pt.y;
                var depthIdx = pt.x;

                // Find indices
                var yIdx = yLabels.indexOf(posIdx);
                var xIdx = depthLabels.map(function (d) { return d === 'embedding' ? 'Emb' : 'L' + d; }).indexOf(depthIdx);

                if (yIdx >= 0 && xIdx >= 0 && data.predictions_by_depth[yIdx] && data.predictions_by_depth[yIdx][xIdx]) {
                    var preds = data.predictions_by_depth[yIdx][xIdx];
                    var html = '<div style="margin-top:12px; padding:12px 16px; background:var(--bg-card); border:1px solid var(--border); border-radius:8px;">' +
                        '<div style="font-size:11px; color:var(--text-muted); font-family:var(--font-mono); margin-bottom:8px;">Top-5 at ' + posIdx + ' / ' + depthIdx + '</div>';
                    html += '<div class="prediction-runners">';
                    for (var pi = 0; pi < preds.length; pi++) {
                        html += '<span class="prediction-pill">' +
                            '<span class="prediction-pill-token">' + escapeHtml(preds[pi].token) + '</span>' +
                            '<span class="prediction-pill-prob">' + (preds[pi].prob * 100).toFixed(1) + '%</span></span>';
                    }
                    html += '</div></div>';
                    detailDiv.innerHTML = html;
                }
            }
        });
    } catch (err) {
        showError('attr-waterfall-chart', err.message);
    }
}
