/**
 * attention.js — Attention Inspector tab for LLaMA-2 interpretability.
 *
 * Handles Grouped-Query Attention (GQA): 6 query heads sharing 2 KV heads.
 * Views: All-Heads (mini heatmaps), Single Head, Entropy Heatmap, Head Ablation.
 *
 * Relies on shared utilities from app.js:
 *   createPromptInput, createDropdown, apiFetch, darkLayout, showLoading,
 *   hideLoading, showError, escapeHtml, getGQAGroup, getGQAColor,
 *   window.N_LAYERS, window.N_HEADS, window.N_KV_HEADS, window.N_KV_GROUPS,
 *   window.GQA_COLORS, window.LAYER_COLORS, window.PLOTLY_CONFIG
 *
 * Uses D3.js v7 for heatmaps and Plotly for entropy/ablation views.
 */

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
var _attnState = {
    initialized: false,
    data: null,           // response from /api/attention
    ablationData: null,   // response from /api/ablation
    selectedLayer: 0,
    selectedHead: 0,
    currentView: 'all-heads',
    tooltip: null,
};

// ---------------------------------------------------------------------------
// initAttention — called once on first tab visit
// ---------------------------------------------------------------------------
function initAttention() {
    if (_attnState.initialized) return;
    _attnState.initialized = true;

    _buildControls();
    _buildViewArea();
    _createTooltip();
}

// ---------------------------------------------------------------------------
// Build controls bar
// ---------------------------------------------------------------------------
function _buildControls() {
    var controlsBar = document.getElementById('attention-controls');
    if (!controlsBar) return;

    // Prompt input
    createPromptInput('attention-controls', function (prompt) {
        analyzeAttention(prompt);
    }, { placeholder: 'Enter a prompt to inspect attention...', buttonLabel: 'Inspect' });

    // Layer dropdown (0 to N_LAYERS-1)
    var layerOptions = [];
    for (var i = 0; i < window.N_LAYERS; i++) {
        layerOptions.push({ value: String(i), label: 'Layer ' + i });
    }
    var layerSel = createDropdown('attention-controls', 'Layer:', 'attn-layer-select', layerOptions);
    if (layerSel) {
        layerSel.addEventListener('change', function () {
            _attnState.selectedLayer = parseInt(layerSel.value, 10);
            _renderCurrentView();
        });
    }

    // View dropdown
    var viewOptions = [
        { value: 'all-heads', label: 'All Heads' },
        { value: 'model-view', label: 'Model View' },
        { value: 'entropy',   label: 'Entropy' },
        { value: 'ablation',  label: 'Ablation' },
        { value: 'flow',      label: 'Flow / Rollout' },
    ];
    var viewSel = createDropdown('attention-controls', 'View:', 'attn-view-select', viewOptions);
    if (viewSel) {
        viewSel.addEventListener('change', function () {
            _attnState.currentView = viewSel.value;
            _renderCurrentView();
        });
    }
}

// ---------------------------------------------------------------------------
// Build view area (two regions: main + detail)
// ---------------------------------------------------------------------------
function _buildViewArea() {
    var viewArea = document.getElementById('attention-view');
    if (!viewArea) return;

    // Helper text shown before first analysis
    var guide = document.createElement('div');
    guide.id = 'attn-guide';
    guide.className = 'tab-guide';
    guide.innerHTML =
        '<div class="guide-title">Attention Inspector</div>' +
        '<p>Visualize how each attention head distributes focus across tokens. ' +
        'This model uses <strong>Grouped-Query Attention</strong> (GQA) — 6 query heads share 2 KV heads.</p>' +
        '<div class="guide-features">' +
            '<div class="guide-item"><span class="guide-tag">all heads</span>Mini heatmaps for every head at a chosen layer. Click to expand.</div>' +
            '<div class="guide-item"><span class="guide-tag">model view</span>All heads at all layers in a single grid. Read bottom-to-top.</div>' +
            '<div class="guide-item"><span class="guide-tag">entropy</span>Which heads focus sharply vs. spread attention uniformly.</div>' +
            '<div class="guide-item"><span class="guide-tag">ablation</span>Zero out each head and measure how loss changes. Shows which heads matter most.</div>' +
            '<div class="guide-item"><span class="guide-tag">flow</span>Attention rollout showing effective attention after all layers and residual connections.</div>' +
        '</div>' +
        '<p class="guide-hint">Enter a prompt above and click <strong>Inspect</strong> to begin.</p>';
    viewArea.appendChild(guide);

    var mainArea = document.createElement('div');
    mainArea.id = 'attn-main-area';
    viewArea.appendChild(mainArea);

    var detailArea = document.createElement('div');
    detailArea.id = 'attn-detail-area';
    detailArea.style.marginTop = '20px';
    viewArea.appendChild(detailArea);
}

// ---------------------------------------------------------------------------
// Global tooltip for D3 heatmaps
// ---------------------------------------------------------------------------
function _createTooltip() {
    if (_attnState.tooltip) return;
    var tip = document.createElement('div');
    tip.className = 'heatmap-tooltip';
    tip.style.display = 'none';
    document.body.appendChild(tip);
    _attnState.tooltip = tip;
}

function _showTooltip(html, event) {
    var tip = _attnState.tooltip;
    if (!tip) return;
    tip.innerHTML = html;
    tip.style.display = 'block';
    tip.style.left = (event.clientX + 14) + 'px';
    tip.style.top = (event.clientY - 10) + 'px';
}

function _hideTooltip() {
    var tip = _attnState.tooltip;
    if (tip) tip.style.display = 'none';
}

// ---------------------------------------------------------------------------
// analyzeAttention — fetch attention + ablation data
// ---------------------------------------------------------------------------
function analyzeAttention(prompt) {
    // Hide the guide on first analysis
    var guide = document.getElementById('attn-guide');
    if (guide) guide.style.display = 'none';

    var mainArea = document.getElementById('attn-main-area');
    if (!mainArea) return;
    mainArea.innerHTML = '';
    var detailArea = document.getElementById('attn-detail-area');
    if (detailArea) detailArea.innerHTML = '';

    showLoading('attention-view');

    var attnPromise = apiFetch('/api/attention', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt }),
    });

    var ablationPromise = apiFetch('/api/ablation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt }),
    });

    Promise.all([attnPromise, ablationPromise])
        .then(function (results) {
            hideLoading('attention-view');
            _attnState.data = results[0];
            _attnState.ablationData = results[1];
            _renderCurrentView();
        })
        .catch(function (err) {
            hideLoading('attention-view');
            showError('attn-main-area', 'Analysis failed: ' + err.message);
        });
}

// ---------------------------------------------------------------------------
// View router
// ---------------------------------------------------------------------------
function _renderCurrentView() {
    if (!_attnState.data) return;

    var mainArea = document.getElementById('attn-main-area');
    var detailArea = document.getElementById('attn-detail-area');
    if (!mainArea) return;

    mainArea.innerHTML = '';
    if (detailArea) detailArea.innerHTML = '';

    switch (_attnState.currentView) {
        case 'all-heads':
            _renderAllHeadsView(mainArea, detailArea);
            break;
        case 'model-view':
            _renderModelView(mainArea);
            break;
        case 'entropy':
            _renderEntropyView(mainArea);
            break;
        case 'ablation':
            _renderAblationView(mainArea);
            break;
        case 'flow':
            _renderFlowView(mainArea);
            break;
        default:
            _renderAllHeadsView(mainArea, detailArea);
    }
}

// ---------------------------------------------------------------------------
// VIEW 1: All Heads — 6 mini D3 heatmaps in 3-column grid
// ---------------------------------------------------------------------------
function _renderAllHeadsView(mainArea, detailArea) {
    var data = _attnState.data;
    var layer = _attnState.selectedLayer;
    var layerKey = String(layer);

    if (!data.layers || !data.layers[layerKey]) {
        mainArea.innerHTML = '<div class="placeholder-message">No attention data for layer ' + layer + '</div>';
        return;
    }

    var tokens = data.tokens;
    var nHeads = data.n_heads;
    var layerHeads = data.layers[layerKey];

    // Grid container
    var grid = document.createElement('div');
    grid.className = 'mini-heatmap-grid';
    mainArea.appendChild(grid);

    for (var h = 0; h < nHeads; h++) {
        (function (headIdx) {
            var headKey = String(headIdx);
            var weights = layerHeads[headKey];
            if (!weights) return;

            var group = getGQAGroup(headIdx);
            var kvHead = group;

            var miniCard = document.createElement('div');
            miniCard.className = 'mini-heatmap gqa-group-' + group;
            miniCard.title = 'Head ' + headIdx + ' (KV head ' + kvHead + ')';

            var label = document.createElement('div');
            label.className = 'mini-label';
            label.textContent = 'Head ' + headIdx + ' (KV ' + kvHead + ')';
            label.style.color = getGQAColor(headIdx);
            miniCard.appendChild(label);

            var svgContainer = document.createElement('div');
            svgContainer.id = 'mini-heatmap-' + headIdx;
            miniCard.appendChild(svgContainer);

            grid.appendChild(miniCard);

            // Draw mini heatmap
            _drawD3Heatmap(svgContainer, weights, tokens, {
                cellSize: Math.max(8, Math.min(20, Math.floor(180 / tokens.length))),
                showLabels: tokens.length <= 12,
                mini: true,
            });

            // Click to expand
            miniCard.addEventListener('click', function () {
                _attnState.selectedHead = headIdx;
                _renderSingleHeadView(detailArea, layer, headIdx);
            });
        })(h);
    }

    // GQA legend
    var legend = document.createElement('div');
    legend.style.cssText = 'margin-top:12px; display:flex; gap:20px; font-size:12px; color:var(--text-muted); align-items:center;';
    legend.innerHTML =
        '<span style="display:flex;align-items:center;gap:6px;">' +
        '<span style="width:14px;height:14px;border-radius:3px;background:var(--gqa-group0);display:inline-block;"></span>' +
        'KV Head 0 (Heads 0-2)</span>' +
        '<span style="display:flex;align-items:center;gap:6px;">' +
        '<span style="width:14px;height:14px;border-radius:3px;background:var(--gqa-group1);display:inline-block;"></span>' +
        'KV Head 1 (Heads 3-5)</span>';
    legend.appendChild(createHelpIcon('GQA Grouping',
        '<strong>Grouped-Query Attention</strong> shares Key-Value heads across multiple Query heads to reduce memory. ' +
        'Heads 0-2 share KV Head 0, Heads 3-5 share KV Head 1. ' +
        'Heads within a group see the same keys/values but can learn different query patterns.'
    ));
    mainArea.appendChild(legend);
}

// ---------------------------------------------------------------------------
// VIEW 2: Single Head — full-size D3 heatmap
// ---------------------------------------------------------------------------
function _renderSingleHeadView(container, layer, head) {
    if (!container) return;
    container.innerHTML = '';

    var data = _attnState.data;
    var layerKey = String(layer);
    var headKey = String(head);

    if (!data.layers[layerKey] || !data.layers[layerKey][headKey]) return;

    var tokens = data.tokens;
    var weights = data.layers[layerKey][headKey];
    var group = getGQAGroup(head);

    var card = document.createElement('div');
    card.className = 'heatmap-container';

    var title = document.createElement('div');
    title.className = 'chart-title';
    title.style.color = getGQAColor(head);
    title.textContent = 'Layer ' + layer + ' / Head ' + head + ' (KV Head ' + group + ')';
    card.appendChild(title);

    var svgContainer = document.createElement('div');
    svgContainer.id = 'single-head-heatmap';
    card.appendChild(svgContainer);

    container.appendChild(card);

    _drawD3Heatmap(svgContainer, weights, tokens, {
        cellSize: Math.max(16, Math.min(40, Math.floor(500 / tokens.length))),
        showLabels: true,
        mini: false,
    });
}

// ---------------------------------------------------------------------------
// VIEW 3: Entropy Heatmap — Plotly 8x6 (layers x heads)
// ---------------------------------------------------------------------------
function _renderEntropyView(container) {
    var data = _attnState.data;
    if (!data.entropy_summary) {
        container.innerHTML = '<div class="placeholder-message">No entropy data available.</div>';
        return;
    }

    var card = document.createElement('div');
    card.className = 'chart-card full-width';
    var entropyTitle = document.createElement('div');
    entropyTitle.className = 'chart-title';
    entropyTitle.textContent = 'Mean Attention Entropy per Head';
    entropyTitle.appendChild(createHelpIcon('Attention Entropy',
        'Entropy measures how <strong>spread out</strong> vs <strong>focused</strong> attention is. ' +
        '<strong>Low entropy</strong> = head focuses on 1-2 tokens. ' +
        '<strong>High entropy</strong> = attention spread broadly across many tokens. ' +
        'Focused heads often perform specific functions like "attend to previous token" or "attend to first token".'
    ));
    card.appendChild(entropyTitle);
    var chartDiv = document.createElement('div');
    chartDiv.id = 'attn-entropy-chart';
    chartDiv.style.minHeight = '400px';
    card.appendChild(chartDiv);
    container.appendChild(card);

    var nLayers = window.N_LAYERS;
    var nHeads = data.n_heads;

    // Build z matrix: rows = layers (bottom to top), cols = heads
    var z = [];
    var yLabels = [];
    var xLabels = [];

    for (var h = 0; h < nHeads; h++) {
        xLabels.push('Head ' + h);
    }

    for (var l = 0; l < nLayers; l++) {
        yLabels.push('Layer ' + l);
        var row = [];
        var layerEntropy = data.entropy_summary[String(l)];
        for (var h2 = 0; h2 < nHeads; h2++) {
            row.push(layerEntropy ? (layerEntropy[String(h2)] || 0) : 0);
        }
        z.push(row);
    }

    var trace = {
        z: z,
        x: xLabels,
        y: yLabels,
        type: 'heatmap',
        colorscale: 'Viridis',
        colorbar: {
            title: 'Entropy (nats)',
            titleside: 'right',
            tickfont: { color: '#e0e0e0' },
            titlefont: { color: '#e0e0e0' },
        },
        hovertemplate: '%{y}, %{x}<br>Entropy: %{z:.3f}<extra></extra>',
    };

    var layout = darkLayout({
        xaxis: { title: 'Attention Head', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)', dtick: 1 },
        yaxis: { title: 'Layer', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)', dtick: 1, autorange: true },
        margin: { t: 36, r: 100, b: 60, l: 80 },
    });

    Plotly.react(chartDiv, [trace], layout, window.PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// VIEW 4: Head Ablation — Plotly 8x6 diverging heatmap
// ---------------------------------------------------------------------------
function _renderAblationView(container) {
    var ablation = _attnState.ablationData;
    if (!ablation || !ablation.importance) {
        container.innerHTML = '<div class="placeholder-message">No ablation data available. Submit a prompt first.</div>';
        return;
    }

    var card = document.createElement('div');
    card.className = 'chart-card full-width';
    var ablTitle = document.createElement('div');
    ablTitle.className = 'chart-title';
    ablTitle.textContent = 'Head Ablation: Delta-Loss (blue = helps, red = hurts)';
    ablTitle.appendChild(createHelpIcon('Head Ablation',
        'Each head is <strong>zeroed out</strong> one at a time and the loss change is measured. ' +
        '<strong>Positive delta</strong> (red) = removing the head hurts performance (important head). ' +
        '<strong>Negative delta</strong> (blue) = removing the head helps performance (possibly harmful head). ' +
        'This reveals which heads matter most for the model\'s prediction.'
    ));
    card.appendChild(ablTitle);
    var chartDiv = document.createElement('div');
    chartDiv.id = 'attn-ablation-chart';
    chartDiv.style.minHeight = '400px';
    card.appendChild(chartDiv);
    container.appendChild(card);

    var importance = ablation.importance; // 8 x 6 array
    var nLayers = importance.length;
    var nHeads = importance[0].length;

    var yLabels = [];
    var xLabels = [];

    for (var h = 0; h < nHeads; h++) {
        xLabels.push('Head ' + h);
    }
    for (var l = 0; l < nLayers; l++) {
        yLabels.push('Layer ' + l);
    }

    // Find max absolute value for symmetric color scale
    var maxAbs = 0;
    for (var i = 0; i < nLayers; i++) {
        for (var j = 0; j < nHeads; j++) {
            var absVal = Math.abs(importance[i][j]);
            if (absVal > maxAbs) maxAbs = absVal;
        }
    }
    if (maxAbs === 0) maxAbs = 1;

    var trace = {
        z: importance,
        x: xLabels,
        y: yLabels,
        type: 'heatmap',
        colorscale: [
            [0,   '#2166ac'],
            [0.25,'#67a9cf'],
            [0.5, '#f7f7f7'],
            [0.75,'#ef8a62'],
            [1,   '#b2182b'],
        ],
        zmin: -maxAbs,
        zmax: maxAbs,
        colorbar: {
            title: 'Delta Loss',
            titleside: 'right',
            tickfont: { color: '#e0e0e0' },
            titlefont: { color: '#e0e0e0' },
        },
        hovertemplate: '%{y}, %{x}<br>Delta Loss: %{z:.4f}<extra></extra>',
    };

    var layout = darkLayout({
        xaxis: { title: 'Attention Head', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)', dtick: 1 },
        yaxis: { title: 'Layer', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.06)', dtick: 1, autorange: true },
        margin: { t: 36, r: 100, b: 60, l: 80 },
    });

    Plotly.react(chartDiv, [trace], layout, window.PLOTLY_CONFIG);

    // Show max importance badge
    if (ablation.max_importance) {
        var mi = ablation.max_importance;
        var badge = document.createElement('div');
        badge.style.cssText = 'margin-top:12px; padding:10px 16px; background:var(--bg-card); border:1px solid var(--border); border-radius:8px; font-size:13px; color:var(--text-muted);';
        badge.innerHTML =
            'Most impactful head: <strong style="color:var(--highlight);">Layer ' + mi.layer +
            ', Head ' + mi.head + '</strong> (delta loss: <span style="font-family:var(--font-mono);color:' +
            (mi.delta > 0 ? '#b2182b' : '#2166ac') + ';">' + mi.delta.toFixed(4) + '</span>)' +
            ' | Baseline loss: <span style="font-family:var(--font-mono);color:var(--warning);">' +
            ablation.baseline_loss.toFixed(4) + '</span>';
        container.appendChild(badge);
    }
}

// ---------------------------------------------------------------------------
// D3 Heatmap renderer
// ---------------------------------------------------------------------------
function _drawD3Heatmap(container, weights, tokens, options) {
    var cellSize = options.cellSize || 20;
    var showLabels = options.showLabels !== false;
    var isMini = options.mini || false;

    var T = tokens.length;
    var labelMargin = showLabels ? Math.min(80, Math.max(40, _maxTokenLabelWidth(tokens))) : 0;
    var svgW = T * cellSize + labelMargin + 4;
    var svgH = T * cellSize + labelMargin + 4;

    // Clear container
    d3.select(container).selectAll('*').remove();

    var svg = d3.select(container)
        .append('svg')
        .attr('width', svgW)
        .attr('height', svgH);

    var g = svg.append('g')
        .attr('transform', 'translate(' + labelMargin + ',' + labelMargin + ')');

    var colorScale = d3.scaleSequential(d3.interpolateInferno)
        .domain([0, 1]);

    // Draw cells
    for (var qIdx = 0; qIdx < T; qIdx++) {
        for (var kIdx = 0; kIdx < T; kIdx++) {
            // Causal mask: upper triangle is transparent (query position < key position)
            if (kIdx > qIdx) continue;

            var weight = weights[qIdx][kIdx];

            (function (qi, ki, w) {
                var rect = g.append('rect')
                    .attr('x', ki * cellSize)
                    .attr('y', qi * cellSize)
                    .attr('width', cellSize)
                    .attr('height', cellSize)
                    .attr('fill', colorScale(w))
                    .attr('stroke', 'none');

                if (!isMini) {
                    rect
                        .on('mousemove', function (event) {
                            var html = 'query[' + escapeHtml(tokens[qi]) + '] &rarr; key[' +
                                escapeHtml(tokens[ki]) + ']: <strong>' + w.toFixed(4) + '</strong>';
                            _showTooltip(html, event);
                        })
                        .on('mouseout', function () {
                            _hideTooltip();
                        });
                }
            })(qIdx, kIdx, weight);
        }
    }

    // Token labels on axes
    if (showLabels) {
        // X-axis labels (keys) — along the top, rotated
        for (var ki2 = 0; ki2 < T; ki2++) {
            g.append('text')
                .attr('x', ki2 * cellSize + cellSize / 2)
                .attr('y', -4)
                .attr('text-anchor', 'start')
                .attr('transform', 'rotate(-45,' + (ki2 * cellSize + cellSize / 2) + ',-4)')
                .style('font-size', isMini ? '7px' : '10px')
                .style('font-family', 'var(--font-mono)')
                .style('fill', '#71717a')
                .text(_truncateToken(tokens[ki2], isMini ? 4 : 8));
        }

        // Y-axis labels (queries) — along the left
        for (var qi2 = 0; qi2 < T; qi2++) {
            g.append('text')
                .attr('x', -4)
                .attr('y', qi2 * cellSize + cellSize / 2 + 3)
                .attr('text-anchor', 'end')
                .style('font-size', isMini ? '7px' : '10px')
                .style('font-family', 'var(--font-mono)')
                .style('fill', '#71717a')
                .text(_truncateToken(tokens[qi2], isMini ? 4 : 8));
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function _truncateToken(token, maxLen) {
    if (token.length <= maxLen) return token;
    return token.substring(0, maxLen - 1) + '\u2026';
}

function _maxTokenLabelWidth(tokens) {
    // Estimate pixel width of longest token label
    var maxLen = 0;
    for (var i = 0; i < tokens.length; i++) {
        if (tokens[i].length > maxLen) maxLen = tokens[i].length;
    }
    return Math.min(maxLen * 7 + 10, 80);
}

// ---------------------------------------------------------------------------
// VIEW: Model View — all layers x all heads in a grid
// ---------------------------------------------------------------------------
function _renderModelView(container) {
    var data = _attnState.data;
    if (!data || !data.layers) {
        container.innerHTML = '<div class="placeholder-message">No attention data. Submit a prompt first.</div>';
        return;
    }

    var card = document.createElement('div');
    card.className = 'chart-card full-width';

    var title = document.createElement('div');
    title.className = 'chart-title';
    title.textContent = 'Model-Wide Attention View (all layers x all heads)';
    title.appendChild(createHelpIcon('Model View',
        'Shows attention patterns for <strong>every head at every layer</strong> simultaneously. ' +
        'Read bottom-to-top to see how patterns evolve through the network. ' +
        'Colors indicate GQA group membership. Click any mini heatmap to expand it.'
    ));
    card.appendChild(title);

    var tokens = data.tokens;
    var nLayers = window.N_LAYERS;
    var nHeads = data.n_heads;

    // Grid: rows = layers (bottom to top), cols = heads
    var grid = document.createElement('div');
    grid.style.cssText = 'display:grid; grid-template-columns: 60px repeat(' + nHeads + ', 1fr); gap:4px; align-items:center;';

    // Header row
    var emptyCorner = document.createElement('div');
    grid.appendChild(emptyCorner);
    for (var h = 0; h < nHeads; h++) {
        var hLabel = document.createElement('div');
        hLabel.style.cssText = 'text-align:center; font-size:10px; color:' + getGQAColor(h) + '; font-family:var(--font-mono); font-weight:600;';
        hLabel.textContent = 'H' + h;
        grid.appendChild(hLabel);
    }

    // Layer rows (bottom = layer 0 at bottom, layer 7 at top)
    for (var l = nLayers - 1; l >= 0; l--) {
        var layerKey = String(l);
        if (!data.layers[layerKey]) continue;

        // Row label
        var rowLabel = document.createElement('div');
        rowLabel.className = 'model-view-row-label';
        rowLabel.textContent = 'Layer ' + l;
        rowLabel.style.color = window.LAYER_COLORS[l];
        grid.appendChild(rowLabel);

        for (var hi = 0; hi < nHeads; hi++) {
            (function (layerIdx, headIdx) {
                var headKey = String(headIdx);
                var weights = data.layers[layerKey][headKey];
                if (!weights) {
                    grid.appendChild(document.createElement('div'));
                    return;
                }

                var group = getGQAGroup(headIdx);
                var cell = document.createElement('div');
                cell.className = 'model-view-cell gqa-group-' + group;

                var svgContainer = document.createElement('div');
                svgContainer.id = 'mv-' + layerIdx + '-' + headIdx;
                cell.appendChild(svgContainer);
                grid.appendChild(cell);

                var cellSize = Math.max(4, Math.min(10, Math.floor(80 / tokens.length)));
                _drawD3Heatmap(svgContainer, weights, tokens, {
                    cellSize: cellSize,
                    showLabels: false,
                    mini: true,
                });

                cell.addEventListener('click', function () {
                    _attnState.selectedLayer = layerIdx;
                    _attnState.selectedHead = headIdx;
                    var detailArea = document.getElementById('attn-detail-area');
                    _renderSingleHeadView(detailArea, layerIdx, headIdx);
                });
            })(l, hi);
        }
    }

    card.appendChild(grid);
    container.appendChild(card);
}

// ---------------------------------------------------------------------------
// VIEW: Attention Flow / Rollout
// ---------------------------------------------------------------------------
function _renderFlowView(container) {
    var data = _attnState.data;
    if (!data) {
        container.innerHTML = '<div class="placeholder-message">No attention data. Submit a prompt first.</div>';
        return;
    }

    var card = document.createElement('div');
    card.className = 'chart-card full-width';

    var title = document.createElement('div');
    title.className = 'chart-title';
    title.textContent = 'Attention Flow / Rollout';
    title.appendChild(createHelpIcon('Attention Flow',
        'Attention flow shows <strong>effective attention</strong> after accounting for all layers and residual connections. ' +
        'Information can flow indirectly through intermediate tokens. ' +
        'The rollout matrix is computed by multiplying attention matrices across layers with residual connection weighting.'
    ));
    card.appendChild(title);

    var chartDiv = document.createElement('div');
    chartDiv.id = 'attn-flow-chart';
    chartDiv.style.minHeight = '400px';
    card.appendChild(chartDiv);

    var statusDiv = document.createElement('div');
    statusDiv.id = 'attn-flow-status';
    card.appendChild(statusDiv);
    container.appendChild(card);

    showInlineLoading('attn-flow-chart');

    // Get the prompt from the input
    var promptInput = document.getElementById('attention-controls-prompt-input');
    var prompt = promptInput ? promptInput.value.trim() : '';
    if (!prompt) {
        showError('attn-flow-chart', 'No prompt available. Enter a prompt and click Inspect first.');
        return;
    }

    apiFetch('/api/attention_flow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt }),
    })
        .then(function (flowData) {
            var tokens = flowData.tokens;
            var rollout = flowData.rollout;

            Plotly.newPlot('attn-flow-chart', [{
                z: rollout,
                x: tokens,
                y: tokens,
                type: 'heatmap',
                colorscale: 'Viridis',
                colorbar: {
                    title: 'Flow Weight',
                    tickfont: { color: '#e0e0e0' },
                    titlefont: { color: '#e0e0e0' },
                },
                hovertemplate: 'from: %{x}<br>to: %{y}<br>weight: %{z:.4f}<extra></extra>',
            }], darkLayout({
                title: { text: 'Attention Rollout Matrix', font: { size: 14 } },
                xaxis: { title: 'Source Token (key)', tickangle: -45, gridcolor: 'rgba(255,255,255,0.06)' },
                yaxis: { title: 'Target Token (query)', autorange: 'reversed', gridcolor: 'rgba(255,255,255,0.06)' },
                height: Math.max(400, tokens.length * 30 + 120),
                margin: { t: 44, b: 100, l: 100 },
            }), window.PLOTLY_CONFIG);

            statusDiv.className = 'summary-text';
            statusDiv.textContent = 'Rollout computed across ' + window.N_LAYERS + ' layers with 0.5 residual weighting. Click cells to see source contributions for each target token.';
        })
        .catch(function (err) {
            showError('attn-flow-chart', 'Flow analysis failed: ' + err.message);
        });
}
