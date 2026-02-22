/**
 * captum.js — Captum Attribution tab for LLaMA-2 dashboard.
 *
 * Provides three attribution methods:
 *   1. Gradient Saliency (fast)
 *   2. Integrated Gradients (medium)
 *   3. Layer Conductance (slow)
 */

// ---------------------------------------------------------------------------
// Helper: build a colored token strip from scores and a d3 color interpolator
// ---------------------------------------------------------------------------
function buildCaptumTokenStrip(container, tokens, scores, interpolator, diverging) {
    container.innerHTML = '';

    var strip = document.createElement('div');
    strip.className = 'token-strip';

    if (!tokens || tokens.length === 0) return;

    // Normalize scores for coloring
    var minScore = Infinity;
    var maxScore = -Infinity;
    scores.forEach(function (s) {
        if (s < minScore) minScore = s;
        if (s > maxScore) maxScore = s;
    });

    tokens.forEach(function (tok, i) {
        var span = document.createElement('span');
        span.className = 'token-span';
        span.textContent = tok;
        span.title = tok + ': ' + scores[i].toFixed(4);

        var t;
        if (diverging) {
            // Map score range to [0, 1] centered at 0
            var absMax = Math.max(Math.abs(minScore), Math.abs(maxScore));
            t = absMax === 0 ? 0.5 : (scores[i] / (2 * absMax)) + 0.5;
            // For reversed RdBu: high positive = red, high negative = blue
            // d3.interpolateRdBu: 0 = red, 1 = blue. We reverse so positive = red.
            t = 1 - t;
        } else {
            // Normalize to [0, 1]
            var range = maxScore - minScore;
            t = range === 0 ? 0 : (scores[i] - minScore) / range;
        }

        var color = interpolator(t);
        span.style.background = color;

        // Determine text color for readability
        var rgb = d3.color(color);
        if (rgb) {
            var lum = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
            span.style.color = lum > 140 ? '#09090b' : '#ffffff';
        }

        strip.appendChild(span);
    });

    container.appendChild(strip);
}

// ---------------------------------------------------------------------------
// Main init
// ---------------------------------------------------------------------------
function initCaptum() {
    var controlsRoot = document.getElementById('captum-controls');
    var resultsRoot = document.getElementById('captum-results');
    if (!controlsRoot || !resultsRoot) return;

    controlsRoot.innerHTML = '';
    resultsRoot.innerHTML = '<div class="placeholder-message">Select an attribution method and enter a prompt to begin.</div>';

    // Prompt input
    var promptGroup = document.createElement('div');
    promptGroup.className = 'prompt-group';

    var promptInput = document.createElement('input');
    promptInput.type = 'text';
    promptInput.className = 'prompt-input';
    promptInput.placeholder = 'Enter a prompt for attribution analysis...';
    promptInput.id = 'captum-prompt-input';

    promptGroup.appendChild(promptInput);
    controlsRoot.appendChild(promptGroup);

    // Help icons for Captum methods
    controlsRoot.appendChild(createHelpIcon('Captum Attribution',
        '<strong>Gradient Saliency</strong>: Computes the gradient magnitude for each token embedding. Fast but noisy. ' +
        '<strong>Integrated Gradients</strong>: Accumulates gradients along the path from a baseline to the input. More reliable but slower. ' +
        '<strong>Layer Conductance</strong>: Measures how much each layer contributes to the output for each token. Slowest but most informative.'
    ));

    // Three method buttons
    var btnSaliency = document.createElement('button');
    btnSaliency.className = 'btn btn-primary';
    btnSaliency.textContent = 'Gradient Saliency';
    btnSaliency.title = 'Fast';

    var btnIG = document.createElement('button');
    btnIG.className = 'btn btn-primary';
    btnIG.textContent = 'Integrated Gradients';
    btnIG.title = 'Medium';

    var btnConductance = document.createElement('button');
    btnConductance.className = 'btn btn-primary';
    btnConductance.textContent = 'Layer Conductance';
    btnConductance.title = 'Slow';

    // Speed hints next to buttons
    function addSpeedTag(btn, speed) {
        var tag = document.createElement('span');
        tag.style.fontSize = '11px';
        tag.style.marginLeft = '6px';
        tag.style.opacity = '0.7';
        tag.textContent = '(' + speed + ')';
        btn.appendChild(tag);
    }
    addSpeedTag(btnSaliency, 'fast');
    addSpeedTag(btnIG, 'medium');
    addSpeedTag(btnConductance, 'slow');

    controlsRoot.appendChild(btnSaliency);
    controlsRoot.appendChild(btnIG);
    controlsRoot.appendChild(btnConductance);

    var resultsId = 'captum-results';

    // Disable all buttons during a run
    function setButtonsDisabled(disabled) {
        btnSaliency.disabled = disabled;
        btnIG.disabled = disabled;
        btnConductance.disabled = disabled;
    }

    // -----------------------------------------------------------------------
    // Gradient Saliency
    // -----------------------------------------------------------------------
    btnSaliency.addEventListener('click', function () {
        var prompt = promptInput.value.trim();
        if (!prompt) return;

        resultsRoot.innerHTML = '';
        setButtonsDisabled(true);
        showInlineLoading(resultsId);

        apiFetch('/api/captum/saliency', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt }),
        })
            .then(function (data) {
                setButtonsDisabled(false);
                resultsRoot.innerHTML = '';

                // data.tokens: string array
                // data.saliency: number array (per-token saliency scores)
                var tokens = data.tokens;
                var saliency = data.saliency;

                // Token strip colored by saliency (YlOrRd)
                var stripContainer = document.createElement('div');
                stripContainer.id = 'captum-saliency-strip';
                resultsRoot.appendChild(stripContainer);
                buildCaptumTokenStrip(stripContainer, tokens, saliency, d3.interpolateYlOrRd, false);

                // Bar chart
                var chartCard = document.createElement('div');
                chartCard.className = 'chart-card';
                var chartTitle = document.createElement('div');
                chartTitle.className = 'chart-title';
                chartTitle.textContent = 'Gradient Saliency per Token';
                chartTitle.appendChild(createHelpIcon('Gradient Saliency',
                    'Shows how much each token\'s embedding affects the output. ' +
                    'Computed as the <strong>L2 norm of the gradient</strong> of the output with respect to the embedding. ' +
                    'Higher values = the model is more sensitive to changes in that token.'
                ));
                chartCard.appendChild(chartTitle);

                var chartDiv = document.createElement('div');
                chartDiv.id = 'captum-saliency-chart';
                chartDiv.className = 'chart-container';
                chartCard.appendChild(chartDiv);
                resultsRoot.appendChild(chartCard);

                // Normalize saliency to [0, 1] for display
                var maxSal = Math.max.apply(null, saliency);
                var minSal = Math.min.apply(null, saliency);
                var rangeSal = maxSal - minSal || 1;
                var normalizedSaliency = saliency.map(function (s) {
                    return (s - minSal) / rangeSal;
                });

                // Bar colors from YlOrRd
                var barColors = normalizedSaliency.map(function (ns) {
                    return d3.interpolateYlOrRd(ns);
                });

                var trace = {
                    x: tokens,
                    y: saliency,
                    type: 'bar',
                    marker: { color: barColors },
                    hovertemplate: '%{x}<br>Saliency: %{y:.4f}<extra></extra>',
                };

                var layout = darkLayout({
                    title: { text: 'Gradient Saliency', font: { size: 14 } },
                    xaxis: { title: 'Token', tickangle: -45 },
                    yaxis: { title: 'Saliency Score' },
                    height: 350,
                });

                Plotly.newPlot('captum-saliency-chart', [trace], layout, window.PLOTLY_CONFIG);
            })
            .catch(function (err) {
                setButtonsDisabled(false);
                showError(resultsId, err.message);
            });
    });

    // -----------------------------------------------------------------------
    // Integrated Gradients
    // -----------------------------------------------------------------------
    btnIG.addEventListener('click', function () {
        var prompt = promptInput.value.trim();
        if (!prompt) return;

        resultsRoot.innerHTML = '';
        setButtonsDisabled(true);
        showInlineLoading(resultsId);

        apiFetch('/api/captum/integrated_gradients', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt }),
        })
            .then(function (data) {
                setButtonsDisabled(false);
                resultsRoot.innerHTML = '';

                // data.tokens: string array
                // data.attributions: number array (per-token, can be negative)
                // data.convergence_delta: number
                var tokens = data.tokens;
                var attributions = data.attributions;

                // Token strip colored by attribution (diverging: reversed RdBu)
                var stripContainer = document.createElement('div');
                stripContainer.id = 'captum-ig-strip';
                resultsRoot.appendChild(stripContainer);

                // Reversed RdBu: positive = red, negative = blue
                function reversedRdBu(t) {
                    return d3.interpolateRdBu(1 - t);
                }
                buildCaptumTokenStrip(stripContainer, tokens, attributions, reversedRdBu, true);

                // Convergence delta badge
                if (data.convergence_delta !== undefined) {
                    var deltaBadge = document.createElement('div');
                    deltaBadge.className = 'kl-badge';
                    deltaBadge.style.marginBottom = '16px';
                    deltaBadge.innerHTML =
                        '<span class="kl-label">Convergence Delta</span>' +
                        '<span class="kl-value">' + escapeHtml(data.convergence_delta.toFixed(6)) + '</span>';
                    resultsRoot.appendChild(deltaBadge);
                }

                // Bar chart
                var chartCard = document.createElement('div');
                chartCard.className = 'chart-card';
                var chartTitle = document.createElement('div');
                chartTitle.className = 'chart-title';
                chartTitle.textContent = 'Integrated Gradients Attribution per Token';
                chartTitle.appendChild(createHelpIcon('Integrated Gradients',
                    'Accumulates gradients along a straight-line path from a <strong>zero baseline</strong> to the actual input. ' +
                    '<strong>Positive</strong> (red) = token pushes toward the predicted output. ' +
                    '<strong>Negative</strong> (blue) = token pushes away. ' +
                    'The convergence delta measures approximation accuracy (lower = better).'
                ));
                chartCard.appendChild(chartTitle);

                var chartDiv = document.createElement('div');
                chartDiv.id = 'captum-ig-chart';
                chartDiv.className = 'chart-container';
                chartCard.appendChild(chartDiv);
                resultsRoot.appendChild(chartCard);

                // Color bars by sign: positive = red, negative = blue
                var barColors = attributions.map(function (a) {
                    return a >= 0 ? '#f87171' : '#818cf8';
                });

                var trace = {
                    x: tokens,
                    y: attributions,
                    type: 'bar',
                    marker: { color: barColors },
                    hovertemplate: '%{x}<br>Attribution: %{y:.4f}<extra></extra>',
                };

                var layout = darkLayout({
                    title: { text: 'Integrated Gradients', font: { size: 14 } },
                    xaxis: { title: 'Token', tickangle: -45 },
                    yaxis: { title: 'Attribution' },
                    height: 350,
                });

                Plotly.newPlot('captum-ig-chart', [trace], layout, window.PLOTLY_CONFIG);
            })
            .catch(function (err) {
                setButtonsDisabled(false);
                showError(resultsId, err.message);
            });
    });

    // -----------------------------------------------------------------------
    // Layer Conductance
    // -----------------------------------------------------------------------
    btnConductance.addEventListener('click', function () {
        var prompt = promptInput.value.trim();
        if (!prompt) return;

        resultsRoot.innerHTML = '';
        setButtonsDisabled(true);
        showInlineLoading(resultsId);

        apiFetch('/api/captum/layer_conductance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt }),
        })
            .then(function (data) {
                setButtonsDisabled(false);
                resultsRoot.innerHTML = '';

                // API returns layer_conductance: {layer_idx: [per_token_scores]}
                // or conductance: 2D array. Convert dict to 2D array [n_tokens x n_layers].
                var tokens = data.tokens;
                var rawConductance = data.conductance || data.layer_conductance;
                var conductance;
                var nLayers = window.N_LAYERS;

                if (rawConductance && !Array.isArray(rawConductance)) {
                    // Dict format: {0: [scores], 1: [scores], ...} → transpose to [tokens x layers]
                    var nTokens = tokens.length;
                    conductance = [];
                    for (var ti = 0; ti < nTokens; ti++) {
                        var row = [];
                        for (var li = 0; li < nLayers; li++) {
                            var layerScores = rawConductance[String(li)] || rawConductance[li];
                            row.push(layerScores ? (layerScores[ti] || 0) : 0);
                        }
                        conductance.push(row);
                    }
                } else {
                    conductance = rawConductance || [];
                }

                var layerLabels = [];
                for (var l = 0; l < nLayers; l++) {
                    layerLabels.push('Layer ' + l);
                }

                // Heatmap card
                var chartCard = document.createElement('div');
                chartCard.className = 'chart-card full-width';
                var chartTitle = document.createElement('div');
                chartTitle.className = 'chart-title';
                chartTitle.textContent = 'Layer Conductance (per token, per layer)';
                chartTitle.appendChild(createHelpIcon('Layer Conductance',
                    'Measures how much each <strong>layer contributes</strong> to the output for each token. ' +
                    'Computed by decomposing gradients through each layer\'s hidden representation. ' +
                    'Hot spots reveal which layer-token combinations are most important for the prediction.'
                ));
                chartCard.appendChild(chartTitle);

                var chartDiv = document.createElement('div');
                chartDiv.id = 'captum-conductance-heatmap';
                chartDiv.className = 'chart-container';
                chartCard.appendChild(chartDiv);
                resultsRoot.appendChild(chartCard);

                // Transpose conductance for Plotly: z[layer][token] -> z[token][layer] already correct
                // conductance is [n_tokens x n_layers], plotly heatmap wants z[y][x]
                // We want y = tokens, x = layers
                var trace = {
                    z: conductance,
                    x: layerLabels,
                    y: tokens,
                    type: 'heatmap',
                    colorscale: 'RdBu',
                    reversescale: true,
                    zmid: 0,
                    colorbar: { title: 'Conductance' },
                    hovertemplate: 'Token: %{y}<br>%{x}<br>Conductance: %{z:.4f}<extra></extra>',
                };

                var layout = darkLayout({
                    title: { text: 'Layer Conductance', font: { size: 14 } },
                    xaxis: { title: 'Layer', side: 'bottom' },
                    yaxis: { title: 'Token', autorange: 'reversed' },
                    height: Math.max(400, tokens.length * 28 + 120),
                });

                Plotly.newPlot('captum-conductance-heatmap', [trace], layout, window.PLOTLY_CONFIG);
            })
            .catch(function (err) {
                setButtonsDisabled(false);
                showError(resultsId, err.message);
            });
    });
}
