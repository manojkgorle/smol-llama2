/**
 * activations.js — Activation Explorer tab (SwiGLU-aware).
 *
 * Charts:
 *   1. Residual Stream Norms (bar, per-position for selected layer)
 *   2. Attention vs FFN Output Norms (grouped bar, mean over positions, all layers)
 *   3. SwiGLU Gate Histogram (selected layer)
 *   4. SwiGLU Up Projection Histogram (selected layer)
 *   5. SwiGLU Gated Product Histogram (selected layer)
 *   6. Gate Sparsity Across Layers (bar, all layers)
 */

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
var _actData = null;      // cached API response
var _actLayer = 0;        // currently selected layer
var _actViewMode = 'distributions';
var _actPrompt = '';      // current prompt for neuron browser

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function initActivations() {
    var controls = document.getElementById('activations-controls');
    if (!controls) return;

    createPromptInput('activations-controls', function (prompt) {
        analyzeActivations(prompt);
    }, { placeholder: 'Enter a prompt to explore activations...', buttonLabel: 'Analyze' });

    var layerOpts = [];
    for (var i = 0; i < window.N_LAYERS; i++) {
        layerOpts.push({ value: String(i), label: 'Layer ' + i });
    }
    var layerSel = createDropdown('activations-controls', 'Layer:', 'act-layer-select', layerOpts);
    if (layerSel) {
        layerSel.addEventListener('change', function () {
            _actLayer = parseInt(this.value, 10);
            if (_actData) _renderLayerCharts();
        });
    }

    // View switcher: Distributions | Neuron Browser
    var viewSwitcher = document.createElement('div');
    viewSwitcher.className = 'view-switcher';
    viewSwitcher.id = 'act-view-switcher';

    var btnDist = document.createElement('button');
    btnDist.className = 'active';
    btnDist.textContent = 'Distributions';
    btnDist.dataset.view = 'distributions';

    var btnNeuron = document.createElement('button');
    btnNeuron.textContent = 'Neuron Browser';
    btnNeuron.dataset.view = 'neuron-browser';

    var btnNetwork = document.createElement('button');
    btnNetwork.textContent = 'Network View';
    btnNetwork.dataset.view = 'network';

    viewSwitcher.appendChild(btnDist);
    viewSwitcher.appendChild(btnNeuron);
    viewSwitcher.appendChild(btnNetwork);
    controls.appendChild(viewSwitcher);

    viewSwitcher.querySelectorAll('button').forEach(function (btn) {
        btn.addEventListener('click', function () {
            viewSwitcher.querySelectorAll('button').forEach(function (b) { b.classList.remove('active'); });
            btn.classList.add('active');
            _actViewMode = btn.dataset.view;
            if (_actData || _actPrompt) {
                if (_actViewMode === 'neuron-browser') {
                    _renderNeuronBrowser();
                } else if (_actViewMode === 'network') {
                    _renderNetworkDiagram();
                } else {
                    _buildChartContainers();
                    _renderAllCharts();
                }
            }
        });
    });

    // Build chart containers
    var chartsRoot = document.getElementById('activations-charts');
    if (!chartsRoot) return;

    // Guide text shown before first analysis
    chartsRoot.innerHTML =
        '<div id="act-guide" class="tab-guide">' +
            '<div class="guide-title">Activation Explorer</div>' +
            '<p>Inspect internal activations at every layer of the model. ' +
            'This model uses <strong>SwiGLU</strong> feed-forward networks — a gated activation that learns which neurons to activate.</p>' +
            '<div class="guide-features">' +
                '<div class="guide-item"><span class="guide-tag">residual</span>L2 norms of the residual stream at each token position per layer.</div>' +
                '<div class="guide-item"><span class="guide-tag">attn vs ffn</span>Compare how much each sub-layer contributes to the output.</div>' +
                '<div class="guide-item"><span class="guide-tag">swiglu</span>Gate, up-projection, and gated-product histograms revealing activation distributions.</div>' +
                '<div class="guide-item"><span class="guide-tag">sparsity</span>What fraction of gate neurons are near-zero — higher means more selective computation.</div>' +
            '</div>' +
            '<p class="guide-hint">Enter a prompt above and click <strong>Analyze</strong> to begin.</p>' +
        '</div>';
}

function _buildChartContainers() {
    var chartsRoot = document.getElementById('activations-charts');
    if (!chartsRoot) return;

    // Residual norms — full width
    chartsRoot.innerHTML = '';

    // Residual card with help icon
    var residualCard = document.createElement('div');
    residualCard.className = 'chart-card full-width';
    residualCard.id = 'act-residual-card';
    var residualTitle = document.createElement('div');
    residualTitle.className = 'chart-title';
    residualTitle.textContent = 'Residual Stream Norms';
    residualTitle.appendChild(createHelpIcon('Residual Stream',
        'The <strong>residual stream</strong> is the main information highway through the model. ' +
        'Each layer reads from and writes to it. L2 norms show how much information flows through each position. ' +
        'Growing norms across layers may indicate accumulating representations.'
    ));
    residualCard.appendChild(residualTitle);
    var residualChart = document.createElement('div');
    residualChart.className = 'chart-container';
    residualChart.id = 'act-residual-chart';
    residualCard.appendChild(residualChart);
    chartsRoot.appendChild(residualCard);

    // Attn vs FFN card with help icon
    var attnFfnCard = document.createElement('div');
    attnFfnCard.className = 'chart-card full-width';
    attnFfnCard.id = 'act-attn-ffn-card';
    var attnFfnTitle = document.createElement('div');
    attnFfnTitle.className = 'chart-title';
    attnFfnTitle.textContent = 'Attention vs FFN Output Norms (mean over positions)';
    attnFfnTitle.appendChild(createHelpIcon('Attn vs FFN',
        'Compares how much each sub-layer contributes to the output at each layer. ' +
        '<strong>Attention</strong> handles token-to-token interactions. ' +
        '<strong>FFN</strong> handles per-token transformations. ' +
        'Layers where FFN dominates often perform factual recall.'
    ));
    attnFfnCard.appendChild(attnFfnTitle);
    var attnFfnChart = document.createElement('div');
    attnFfnChart.className = 'chart-container';
    attnFfnChart.id = 'act-attn-ffn-chart';
    attnFfnCard.appendChild(attnFfnChart);
    chartsRoot.appendChild(attnFfnCard);

    // SwiGLU grid
    var swigluGrid = document.createElement('div');
    swigluGrid.className = 'chart-grid two-col';
    swigluGrid.id = 'act-swiglu-grid';

    var gateCard = document.createElement('div');
    gateCard.className = 'chart-card';
    gateCard.id = 'act-gate-card';
    var gateTitle = document.createElement('div');
    gateTitle.className = 'chart-title';
    gateTitle.textContent = 'SwiGLU Gate Histogram';
    gateTitle.appendChild(createHelpIcon('SwiGLU Gate',
        '<strong>SwiGLU</strong> uses a learned gate to control which neurons activate. ' +
        'Gate near <strong>0</strong> = blocked, near <strong>1+</strong> = allowed. ' +
        'High sparsity means the model is <strong>selective</strong> — only a few neurons fire per input.'
    ));
    gateCard.appendChild(gateTitle);
    gateCard.innerHTML += '<div class="chart-container" id="act-gate-chart"></div>';
    swigluGrid.appendChild(gateCard);

    swigluGrid.innerHTML +=
        '<div class="chart-card" id="act-up-card">' +
            '<div class="chart-title">SwiGLU Up Projection Histogram</div>' +
            '<div class="chart-container" id="act-up-chart"></div>' +
        '</div>' +
        '<div class="chart-card" id="act-gated-card">' +
            '<div class="chart-title">SwiGLU Gated Product Histogram</div>' +
            '<div class="chart-container" id="act-gated-chart"></div>' +
        '</div>';
    chartsRoot.appendChild(swigluGrid);

    // Sparsity card with help icon
    var sparsityCard = document.createElement('div');
    sparsityCard.className = 'chart-card full-width';
    sparsityCard.id = 'act-sparsity-card';
    var sparsityTitle = document.createElement('div');
    sparsityTitle.className = 'chart-title';
    sparsityTitle.textContent = 'Gate Sparsity Across Layers';
    sparsityTitle.appendChild(createHelpIcon('Gate Sparsity',
        'Fraction of gate neurons with activation <strong>near zero</strong> (|gate| < 0.01). ' +
        'Higher sparsity = more selective computation. ' +
        'Deeper layers often show higher sparsity as the model becomes more specialized.'
    ));
    sparsityCard.appendChild(sparsityTitle);
    var sparsityChart = document.createElement('div');
    sparsityChart.className = 'chart-container';
    sparsityChart.id = 'act-sparsity-chart';
    sparsityCard.appendChild(sparsityChart);
    chartsRoot.appendChild(sparsityCard);
}

// ---------------------------------------------------------------------------
// Fetch
// ---------------------------------------------------------------------------
async function analyzeActivations(prompt) {
    _actPrompt = prompt;
    if (_actViewMode === 'neuron-browser') {
        _renderNeuronBrowser();
        return;
    }
    if (_actViewMode === 'network') {
        _renderNetworkDiagram();
        return;
    }
    _buildChartContainers();
    showLoading('activations-charts');
    try {
        var data = await apiFetch('/api/activations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt }),
        });
        _actData = data;
        _actLayer = parseInt(document.getElementById('act-layer-select').value, 10);
        _renderAllCharts();
    } catch (err) {
        showError('activations-charts', err.message);
    } finally {
        hideLoading('activations-charts');
    }
}

// ---------------------------------------------------------------------------
// Render helpers
// ---------------------------------------------------------------------------

function _renderAllCharts() {
    _renderLayerCharts();
    _renderAttnVsFfn();
    _renderGateSparsity();
}

/** Charts that depend on the selected layer */
function _renderLayerCharts() {
    var layer = _actData.layers[String(_actLayer)];
    if (!layer) return;

    _renderResidualNorms(layer);
    _renderSwiGLUHistogram('act-gate-chart', layer, 'gate', '#f87171');
    _renderSwiGLUHistogram('act-up-chart', layer, 'up', '#34d399');
    _renderSwiGLUHistogram('act-gated-chart', layer, 'gated', '#fbbf24');
}

function _renderResidualNorms(layer) {
    var norms = layer.residual_norms;
    if (!norms) return;
    var tokens = _actData.tokens;
    var x = tokens.map(function (t, i) { return i + ': ' + t; });

    Plotly.react('act-residual-chart', [{
        type: 'bar',
        x: x,
        y: norms,
        marker: { color: window.LAYER_COLORS[_actLayer] },
    }], darkLayout({
        title: { text: 'Residual Stream L2 Norms — Layer ' + _actLayer, font: { size: 14 } },
        xaxis: { title: 'Position', gridcolor: 'rgba(255,255,255,0.06)', tickangle: -45 },
        yaxis: { title: 'L2 Norm', gridcolor: 'rgba(255,255,255,0.06)' },
        margin: { t: 44, b: 80 },
    }), window.PLOTLY_CONFIG);
}

function _renderSwiGLUHistogram(containerId, layer, key, color) {
    var stats = layer.swiglu_stats && layer.swiglu_stats[key];
    if (!stats || !stats.histogram) {
        Plotly.react(containerId, [], darkLayout({
            title: { text: 'No data', font: { size: 13 } },
        }), window.PLOTLY_CONFIG);
        return;
    }
    var hist = stats.histogram;

    var label = key.charAt(0).toUpperCase() + key.slice(1);
    var subtitle = 'mean=' + stats.mean.toFixed(3) + '  std=' + stats.std.toFixed(3);

    Plotly.react(containerId, [{
        type: 'bar',
        x: hist.bins,
        y: hist.counts,
        marker: { color: color, opacity: 0.85 },
        hovertemplate: 'bin: %{x:.3f}<br>count: %{y}<extra></extra>',
    }], darkLayout({
        title: { text: label + ' — Layer ' + _actLayer + '<br><sub>' + subtitle + '</sub>', font: { size: 13 } },
        xaxis: { title: 'Activation Value', gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Count', gridcolor: 'rgba(255,255,255,0.06)' },
        bargap: 0.05,
        margin: { t: 56, b: 48 },
    }), window.PLOTLY_CONFIG);
}

function _renderAttnVsFfn() {
    var attnMeans = [];
    var ffnMeans = [];
    var labels = [];

    for (var i = 0; i < window.N_LAYERS; i++) {
        var ld = _actData.layers[String(i)];
        labels.push('Layer ' + i);

        if (ld && ld.attn_output_norms) {
            var sum = 0;
            for (var j = 0; j < ld.attn_output_norms.length; j++) sum += ld.attn_output_norms[j];
            attnMeans.push(sum / ld.attn_output_norms.length);
        } else {
            attnMeans.push(0);
        }

        if (ld && ld.ffn_output_norms) {
            var sum2 = 0;
            for (var k = 0; k < ld.ffn_output_norms.length; k++) sum2 += ld.ffn_output_norms[k];
            ffnMeans.push(sum2 / ld.ffn_output_norms.length);
        } else {
            ffnMeans.push(0);
        }
    }

    Plotly.react('act-attn-ffn-chart', [
        {
            type: 'bar',
            name: 'Attention',
            x: labels,
            y: attnMeans,
            marker: { color: '#818cf8' },
        },
        {
            type: 'bar',
            name: 'FFN',
            x: labels,
            y: ffnMeans,
            marker: { color: '#a78bfa' },
        },
    ], darkLayout({
        title: { text: 'Attention vs FFN Output Norms (mean over positions)', font: { size: 14 } },
        barmode: 'group',
        xaxis: { title: 'Layer', gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Mean L2 Norm', gridcolor: 'rgba(255,255,255,0.06)' },
        margin: { t: 44, b: 48 },
    }), window.PLOTLY_CONFIG);
}

function _renderGateSparsity() {
    var sparsities = [];
    var labels = [];
    var colors = [];

    for (var i = 0; i < window.N_LAYERS; i++) {
        labels.push('Layer ' + i);
        var ld = _actData.layers[String(i)];
        var sp = 0;
        if (ld && ld.swiglu_stats && ld.swiglu_stats.gate && ld.swiglu_stats.gate.sparsity != null) {
            sp = ld.swiglu_stats.gate.sparsity;
        }
        sparsities.push(sp);
        colors.push(window.LAYER_COLORS[i]);
    }

    Plotly.react('act-sparsity-chart', [{
        type: 'bar',
        x: labels,
        y: sparsities,
        marker: { color: colors },
        hovertemplate: '%{x}<br>Sparsity: %{y:.4f}<extra></extra>',
    }], darkLayout({
        title: { text: 'Gate Sparsity Across Layers (fraction |gate| < 0.01)', font: { size: 14 } },
        xaxis: { title: 'Layer', gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Sparsity', gridcolor: 'rgba(255,255,255,0.06)', range: [0, 1] },
        margin: { t: 44, b: 48 },
    }), window.PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Neuron Browser
// ---------------------------------------------------------------------------
async function _renderNeuronBrowser() {
    var chartsRoot = document.getElementById('activations-charts');
    if (!chartsRoot) return;
    chartsRoot.innerHTML = '';

    if (!_actPrompt) {
        chartsRoot.innerHTML = '<div class="placeholder-message">Enter a prompt and click Analyze to browse neurons.</div>';
        return;
    }

    var layer = parseInt(document.getElementById('act-layer-select').value, 10);

    // Overview card
    var overviewCard = document.createElement('div');
    overviewCard.className = 'chart-card full-width';
    var overviewTitle = document.createElement('div');
    overviewTitle.className = 'chart-title';
    overviewTitle.textContent = 'Neuron Browser — Layer ' + layer;
    overviewTitle.appendChild(createHelpIcon('Neuron Browser',
        'Each FFN neuron is a <strong>learned feature detector</strong>. The gate activation shows how strongly it fires per token. ' +
        '<strong>Sparse neurons</strong> detect specific patterns. Click a neuron bar to see its per-token activation profile.'
    ));
    overviewCard.appendChild(overviewTitle);

    var overviewChart = document.createElement('div');
    overviewChart.className = 'chart-container';
    overviewChart.id = 'neuron-overview-chart';
    overviewCard.appendChild(overviewChart);
    chartsRoot.appendChild(overviewCard);

    // Detail panel
    var detailPanel = document.createElement('div');
    detailPanel.id = 'neuron-detail-panel';
    chartsRoot.appendChild(detailPanel);

    showInlineLoading('neuron-overview-chart');

    try {
        var data = await apiFetch('/api/neurons/overview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: _actPrompt, layer: layer }),
        });

        var neurons = data.neurons;
        if (!neurons || neurons.length === 0) {
            showError('neuron-overview-chart', 'No neuron data available.');
            return;
        }

        var labels = neurons.map(function (n) { return 'N' + n.idx; });
        var values = neurons.map(function (n) { return n.mean_activation; });
        var colors = values.map(function (v) {
            return v >= 0 ? '#34d399' : '#f87171';
        });

        Plotly.newPlot('neuron-overview-chart', [{
            x: labels,
            y: values,
            type: 'bar',
            marker: { color: colors },
            hovertemplate: 'Neuron %{x}<br>Mean Activation: %{y:.4f}<extra></extra>',
        }], darkLayout({
            title: { text: 'Top ' + neurons.length + ' Most Active Neurons (by |mean gate activation|)', font: { size: 14 } },
            xaxis: { title: 'Neuron Index', gridcolor: 'rgba(255,255,255,0.06)' },
            yaxis: { title: 'Mean Gate Activation', gridcolor: 'rgba(255,255,255,0.06)' },
            margin: { t: 44, b: 60 },
        }), window.PLOTLY_CONFIG);

        // Click on bar to show detail
        var chartEl = document.getElementById('neuron-overview-chart');
        chartEl.on('plotly_click', function (clickData) {
            if (clickData.points && clickData.points.length > 0) {
                var pointIdx = clickData.points[0].pointIndex;
                var neuronIdx = neurons[pointIdx].idx;
                _showNeuronDetail(layer, neuronIdx);
            }
        });

    } catch (err) {
        showError('neuron-overview-chart', err.message);
    }
}

async function _showNeuronDetail(layer, neuronIdx) {
    var panel = document.getElementById('neuron-detail-panel');
    if (!panel) return;

    panel.innerHTML = '';
    var detailCard = document.createElement('div');
    detailCard.className = 'chart-card full-width';

    var detailTitle = document.createElement('div');
    detailTitle.className = 'chart-title';
    detailTitle.textContent = 'Neuron ' + neuronIdx + ' — Layer ' + layer + ' (per-token activation)';
    detailCard.appendChild(detailTitle);

    var detailChart = document.createElement('div');
    detailChart.className = 'chart-container';
    detailChart.id = 'neuron-detail-chart';
    detailCard.appendChild(detailChart);
    panel.appendChild(detailCard);

    showInlineLoading('neuron-detail-chart');

    try {
        var data = await apiFetch('/api/neurons/detail', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: _actPrompt, layer: layer, neuron_idx: neuronIdx }),
        });

        var tokens = data.tokens;
        var activations = data.activations;

        var x = tokens.map(function (t, i) { return i + ': ' + t; });
        var colors = activations.map(function (a) {
            return a >= 0 ? '#34d399' : '#f87171';
        });

        Plotly.newPlot('neuron-detail-chart', [{
            x: x,
            y: activations,
            type: 'bar',
            marker: { color: colors },
            hovertemplate: '%{x}<br>Activation: %{y:.4f}<extra></extra>',
        }], darkLayout({
            title: { text: 'Gate Activation for Neuron ' + neuronIdx, font: { size: 14 } },
            xaxis: { title: 'Token', tickangle: -45, gridcolor: 'rgba(255,255,255,0.06)' },
            yaxis: { title: 'Gate Activation', gridcolor: 'rgba(255,255,255,0.06)' },
            margin: { t: 44, b: 80 },
        }), window.PLOTLY_CONFIG);

        // Token strip colored by activation
        if (tokens.length > 0) {
            var strip = document.createElement('div');
            strip.className = 'token-strip';
            strip.style.marginTop = '12px';

            var maxAbs = 0;
            activations.forEach(function (a) { if (Math.abs(a) > maxAbs) maxAbs = Math.abs(a); });
            if (maxAbs === 0) maxAbs = 1;

            tokens.forEach(function (tok, i) {
                var span = document.createElement('span');
                span.className = 'token-span';
                span.textContent = tok;
                span.title = tok + ': ' + activations[i].toFixed(4);
                var intensity = Math.abs(activations[i]) / maxAbs;
                var color = activations[i] >= 0 ? '52, 211, 153' : '248, 113, 113';
                span.style.background = 'rgba(' + color + ',' + (intensity * 0.6 + 0.05).toFixed(2) + ')';
                strip.appendChild(span);
            });
            detailCard.appendChild(strip);
        }
    } catch (err) {
        showError('neuron-detail-chart', err.message);
    }
}

// ---------------------------------------------------------------------------
// Network Diagram — D3 visualization of FFN structure
// ---------------------------------------------------------------------------
async function _renderNetworkDiagram() {
    var chartsRoot = document.getElementById('activations-charts');
    if (!chartsRoot) return;
    chartsRoot.innerHTML = '';

    if (!_actPrompt) {
        chartsRoot.innerHTML = '<div class="placeholder-message">Enter a prompt and click Analyze to view the network.</div>';
        return;
    }

    var layer = parseInt(document.getElementById('act-layer-select').value, 10);

    var card = document.createElement('div');
    card.className = 'chart-card full-width';
    var title = document.createElement('div');
    title.className = 'chart-title';
    title.textContent = 'FFN Network — Layer ' + layer;
    title.appendChild(createHelpIcon('Network View',
        'Shows the <strong>SwiGLU feed-forward network</strong> structure as a connected graph. ' +
        'Each node is a neuron, colored by activation strength (<strong>cyan</strong> = positive, <strong>red</strong> = negative). ' +
        'Connection thickness shows weight magnitude. Only the top-K most active neurons are displayed. ' +
        'The FFN computes: output = (SiLU(gate(x)) * up(x)) @ W_down.'
    ));
    card.appendChild(title);

    var diagramDiv = document.createElement('div');
    diagramDiv.id = 'ffn-network-diagram';
    diagramDiv.className = 'nn-diagram-container';
    diagramDiv.style.minHeight = '500px';
    card.appendChild(diagramDiv);

    var detailDiv = document.createElement('div');
    detailDiv.id = 'network-neuron-detail';
    card.appendChild(detailDiv);

    chartsRoot.appendChild(card);
    showInlineLoading('ffn-network-diagram');

    try {
        var data = await apiFetch('/api/neurons/network', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: _actPrompt, layer: layer, top_k: 20 }),
        });

        _drawFFNDiagram('ffn-network-diagram', data);
    } catch (err) {
        showError('ffn-network-diagram', 'Network diagram failed: ' + err.message);
    }
}

function _drawFFNDiagram(containerId, data) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    var layers = data.layers || [];
    var connections = data.connections || [];
    if (layers.length === 0) {
        container.innerHTML = '<div class="placeholder-message">No network data available.</div>';
        return;
    }

    // Layout constants
    var margin = { top: 50, right: 30, bottom: 40, left: 30 };
    var maxNeurons = 0;
    layers.forEach(function (l) { if (l.neurons.length > maxNeurons) maxNeurons = l.neurons.length; });

    var nodeRadius = Math.max(5, Math.min(12, Math.floor(220 / maxNeurons)));
    var nodeSpacing = nodeRadius * 2 + 6;
    var layerSpacing = 160;
    var width = (layers.length - 1) * layerSpacing + margin.left + margin.right + 40;
    var height = maxNeurons * nodeSpacing + margin.top + margin.bottom + 20;

    // Find max absolute activation for color scale
    var maxAbs = 0;
    layers.forEach(function (layer) {
        layer.neurons.forEach(function (n) {
            if (Math.abs(n.activation) > maxAbs) maxAbs = Math.abs(n.activation);
        });
    });
    if (maxAbs === 0) maxAbs = 1;

    // Color function: negative = red, zero = dark, positive = cyan
    function neuronColor(activation) {
        var t = activation / maxAbs; // -1 to 1
        if (t >= 0) {
            var r = Math.round(9 + t * (34 - 9));
            var g = Math.round(9 + t * (211 - 9));
            var b = Math.round(11 + t * (238 - 11));
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        } else {
            var absT = -t;
            var rn = Math.round(9 + absT * (248 - 9));
            var gn = Math.round(9 + absT * (113 - 9));
            var bn = Math.round(11 + absT * (113 - 11));
            return 'rgb(' + rn + ',' + gn + ',' + bn + ')';
        }
    }

    function glowOpacity(activation) {
        return Math.abs(activation) / maxAbs * 0.4;
    }

    var svg = d3.select('#' + containerId)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    var g = svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    // Precompute node positions
    var nodeMap = {};
    layers.forEach(function (layer, li) {
        var x = li * layerSpacing + 20;
        var nNeurons = layer.neurons.length;
        var totalH = nNeurons * nodeSpacing;
        var startY = (height - margin.top - margin.bottom - totalH) / 2;

        layer.neurons.forEach(function (neuron, ni) {
            var y = startY + ni * nodeSpacing + nodeSpacing / 2;
            var key = li + '-' + neuron.idx;
            nodeMap[key] = { x: x, y: y, neuron: neuron, layerIdx: li };
        });
    });

    // Draw connections
    var maxWeight = 0;
    connections.forEach(function (c) {
        if (Math.abs(c.weight) > maxWeight) maxWeight = Math.abs(c.weight);
    });
    if (maxWeight === 0) maxWeight = 1;

    // Sort connections so stronger ones render on top
    connections.sort(function (a, b) { return Math.abs(a.weight) - Math.abs(b.weight); });

    connections.forEach(function (conn) {
        var srcKey = conn.source_layer + '-' + conn.source_idx;
        var tgtKey = conn.target_layer + '-' + conn.target_idx;
        var src = nodeMap[srcKey];
        var tgt = nodeMap[tgtKey];
        if (!src || !tgt) return;

        var strength = Math.abs(conn.weight) / maxWeight;

        g.append('path')
            .attr('d', 'M' + src.x + ',' + src.y +
                ' C' + (src.x + layerSpacing * 0.4) + ',' + src.y +
                ' ' + (tgt.x - layerSpacing * 0.4) + ',' + tgt.y +
                ' ' + tgt.x + ',' + tgt.y)
            .attr('fill', 'none')
            .attr('stroke', conn.weight >= 0 ? 'rgba(34,211,238,' + (0.08 + strength * 0.35).toFixed(2) + ')' : 'rgba(248,113,113,' + (0.08 + strength * 0.35).toFixed(2) + ')')
            .attr('stroke-width', Math.max(0.5, strength * 3))
            .attr('class', 'nn-link')
            .attr('data-source', srcKey)
            .attr('data-target', tgtKey);
    });

    // Draw layer labels
    layers.forEach(function (layer, li) {
        var x = li * layerSpacing + 20;
        g.append('text')
            .attr('x', x)
            .attr('y', -20)
            .attr('text-anchor', 'middle')
            .attr('fill', '#71717a')
            .attr('font-size', '10px')
            .attr('font-family', 'JetBrains Mono, monospace')
            .attr('font-weight', '600')
            .text(layer.name);
    });

    // Draw neurons
    var tooltip = document.getElementById('nn-diagram-tooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'nn-diagram-tooltip';
        tooltip.className = 'heatmap-tooltip';
        tooltip.style.display = 'none';
        document.body.appendChild(tooltip);
    }

    layers.forEach(function (layer, li) {
        var x = li * layerSpacing + 20;
        var nNeurons = layer.neurons.length;
        var totalH = nNeurons * nodeSpacing;
        var startY = (height - margin.top - margin.bottom - totalH) / 2;

        layer.neurons.forEach(function (neuron, ni) {
            var y = startY + ni * nodeSpacing + nodeSpacing / 2;
            var color = neuronColor(neuron.activation);
            var nodeKey = li + '-' + neuron.idx;

            var nodeGroup = g.append('g')
                .attr('transform', 'translate(' + x + ',' + y + ')')
                .style('cursor', 'pointer');

            // Glow halo
            nodeGroup.append('circle')
                .attr('r', nodeRadius + 3)
                .attr('fill', color)
                .attr('opacity', glowOpacity(neuron.activation))
                .style('filter', 'blur(3px)');

            // Main circle
            nodeGroup.append('circle')
                .attr('r', nodeRadius)
                .attr('fill', color)
                .attr('stroke', 'rgba(255,255,255,0.12)')
                .attr('stroke-width', 1);

            // Neuron index label (only for larger sizes)
            if (nodeRadius >= 8) {
                nodeGroup.append('text')
                    .attr('text-anchor', 'middle')
                    .attr('dy', '0.35em')
                    .attr('fill', Math.abs(neuron.activation) / maxAbs > 0.3 ? '#09090b' : '#71717a')
                    .attr('font-size', '7px')
                    .attr('font-family', 'JetBrains Mono, monospace')
                    .attr('font-weight', '600')
                    .text(neuron.idx);
            }

            // Hover: highlight connected edges + show tooltip
            nodeGroup.on('mouseover', function (event) {
                // Highlight connections
                g.selectAll('.nn-link').each(function () {
                    var link = d3.select(this);
                    if (link.attr('data-source') === nodeKey || link.attr('data-target') === nodeKey) {
                        link.attr('stroke-opacity', 1).attr('stroke-width', 3);
                    } else {
                        link.attr('stroke-opacity', 0.03);
                    }
                });

                // Show tooltip
                tooltip.style.display = 'block';
                tooltip.style.left = (event.pageX + 14) + 'px';
                tooltip.style.top = (event.pageY - 10) + 'px';
                tooltip.innerHTML =
                    '<strong>' + escapeHtml(layer.name) + '</strong> &middot; Neuron ' + neuron.idx +
                    '<br>Activation: <span style="color:var(--highlight);">' + neuron.activation.toFixed(4) + '</span>';
            })
            .on('mouseout', function () {
                // Reset connections
                g.selectAll('.nn-link').each(function () {
                    var link = d3.select(this);
                    var w = parseFloat(link.attr('stroke-width'));
                    link.attr('stroke-opacity', null);
                });
                tooltip.style.display = 'none';
            })
            .on('click', function () {
                // Drill into neuron detail (only for gate neurons)
                if (li === 1 || li === 2) {
                    _showNeuronDetail(parseInt(document.getElementById('act-layer-select').value, 10), neuron.idx);
                }
            });
        });
    });

    // Legend
    var legendG = g.append('g')
        .attr('transform', 'translate(0,' + (height - margin.top - margin.bottom + 10) + ')');

    var defs = svg.append('defs');
    var gradient = defs.append('linearGradient')
        .attr('id', 'nn-act-grad');
    gradient.append('stop').attr('offset', '0%').attr('stop-color', '#f87171');
    gradient.append('stop').attr('offset', '50%').attr('stop-color', '#09090b');
    gradient.append('stop').attr('offset', '100%').attr('stop-color', '#22d3ee');

    legendG.append('rect')
        .attr('x', 0).attr('y', 0)
        .attr('width', 140).attr('height', 8)
        .attr('rx', 4)
        .attr('fill', 'url(#nn-act-grad)');

    legendG.append('text').attr('x', 0).attr('y', -4)
        .attr('fill', '#71717a').attr('font-size', '9px').attr('font-family', 'JetBrains Mono, monospace')
        .text('−' + maxAbs.toFixed(2));
    legendG.append('text').attr('x', 140).attr('y', -4).attr('text-anchor', 'end')
        .attr('fill', '#71717a').attr('font-size', '9px').attr('font-family', 'JetBrains Mono, monospace')
        .text('+' + maxAbs.toFixed(2));
    legendG.append('text').attr('x', 70).attr('y', -4).attr('text-anchor', 'middle')
        .attr('fill', '#71717a').attr('font-size', '9px').attr('font-family', 'JetBrains Mono, monospace')
        .text('activation');

    // Right side: connection legend
    legendG.append('text').attr('x', 200).attr('y', 6)
        .attr('fill', '#71717a').attr('font-size', '9px').attr('font-family', 'JetBrains Mono, monospace')
        .text('Lines: weight magnitude (thick = strong)');

    // Data summary
    legendG.append('text').attr('x', 0).attr('y', 22)
        .attr('fill', '#52525b').attr('font-size', '9px').attr('font-family', 'JetBrains Mono, monospace')
        .text('Top ' + data.top_k + ' of ' + data.hidden_dim + ' hidden neurons | ' + connections.length + ' connections (top-3 per target)');
}
