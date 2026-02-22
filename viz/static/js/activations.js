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
var _actData = null;   // cached API response
var _actLayer = 0;     // currently selected layer

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

    // Build chart containers
    var chartsRoot = document.getElementById('activations-charts');
    if (!chartsRoot) return;

    // Residual norms — full width
    chartsRoot.innerHTML =
        '<div class="chart-card full-width" id="act-residual-card">' +
            '<div class="chart-title">Residual Stream Norms</div>' +
            '<div class="chart-container" id="act-residual-chart"></div>' +
        '</div>' +

        '<div class="chart-card full-width" id="act-attn-ffn-card">' +
            '<div class="chart-title">Attention vs FFN Output Norms (mean over positions)</div>' +
            '<div class="chart-container" id="act-attn-ffn-chart"></div>' +
        '</div>' +

        '<div class="chart-grid two-col" id="act-swiglu-grid">' +
            '<div class="chart-card" id="act-gate-card">' +
                '<div class="chart-title">SwiGLU Gate Histogram</div>' +
                '<div class="chart-container" id="act-gate-chart"></div>' +
            '</div>' +
            '<div class="chart-card" id="act-up-card">' +
                '<div class="chart-title">SwiGLU Up Projection Histogram</div>' +
                '<div class="chart-container" id="act-up-chart"></div>' +
            '</div>' +
            '<div class="chart-card" id="act-gated-card">' +
                '<div class="chart-title">SwiGLU Gated Product Histogram</div>' +
                '<div class="chart-container" id="act-gated-chart"></div>' +
            '</div>' +
        '</div>' +

        '<div class="chart-card full-width" id="act-sparsity-card">' +
            '<div class="chart-title">Gate Sparsity Across Layers</div>' +
            '<div class="chart-container" id="act-sparsity-chart"></div>' +
        '</div>';
}

// ---------------------------------------------------------------------------
// Fetch
// ---------------------------------------------------------------------------
async function analyzeActivations(prompt) {
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
    _renderSwiGLUHistogram('act-gate-chart', layer, 'gate', '#e94560');
    _renderSwiGLUHistogram('act-up-chart', layer, 'up', '#4ecca3');
    _renderSwiGLUHistogram('act-gated-chart', layer, 'gated', '#f0a500');
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
        xaxis: { title: 'Position', gridcolor: '#2a2a4a', tickangle: -45 },
        yaxis: { title: 'L2 Norm', gridcolor: '#2a2a4a' },
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
        xaxis: { title: 'Activation Value', gridcolor: '#2a2a4a' },
        yaxis: { title: 'Count', gridcolor: '#2a2a4a' },
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
            marker: { color: '#3282b8' },
        },
        {
            type: 'bar',
            name: 'FFN',
            x: labels,
            y: ffnMeans,
            marker: { color: '#bb86fc' },
        },
    ], darkLayout({
        title: { text: 'Attention vs FFN Output Norms (mean over positions)', font: { size: 14 } },
        barmode: 'group',
        xaxis: { title: 'Layer', gridcolor: '#2a2a4a' },
        yaxis: { title: 'Mean L2 Norm', gridcolor: '#2a2a4a' },
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
        xaxis: { title: 'Layer', gridcolor: '#2a2a4a' },
        yaxis: { title: 'Sparsity', gridcolor: '#2a2a4a', range: [0, 1] },
        margin: { t: 44, b: 48 },
    }), window.PLOTLY_CONFIG);
}
