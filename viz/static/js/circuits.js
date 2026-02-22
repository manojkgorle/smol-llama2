/**
 * circuits.js — Circuits tab with 4 collapsible subsections for LLaMA-2 dashboard.
 *
 * Subsections:
 *   1. Activation Patching
 *   2. Activation Steering
 *   3. Activation Swapping
 *   4. Pre-computation Detection
 */

// ---------------------------------------------------------------------------
// Helper: render side-by-side prediction comparison panels
// ---------------------------------------------------------------------------
function renderPredictionComparison(containerId, baselinePreds, comparedPreds, baselineLabel, comparedLabel) {
    var container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    var wrapper = document.createElement('div');
    wrapper.className = 'side-by-side';

    // Build a single comparison panel
    function buildPanel(preds, label, barColor) {
        var panel = document.createElement('div');
        panel.className = 'comparison-panel';

        var title = document.createElement('div');
        title.className = 'panel-title';
        title.textContent = label;
        panel.appendChild(title);

        if (!preds || preds.length === 0) {
            var empty = document.createElement('div');
            empty.className = 'placeholder-message';
            empty.textContent = 'No predictions available';
            panel.appendChild(empty);
            return panel;
        }

        // Find max probability for bar scaling (API may use 'prob' or 'probability')
        var maxProb = 0;
        preds.forEach(function (p) {
            var v = p.probability !== undefined ? p.probability : (p.prob || 0);
            if (v > maxProb) maxProb = v;
        });
        if (maxProb === 0) maxProb = 1;

        preds.forEach(function (p) {
            var row = document.createElement('div');
            row.className = 'pred-row';

            var prob = p.probability !== undefined ? p.probability : p.prob;
            if (prob === undefined || prob === null || isNaN(prob)) prob = 0;

            var tokenEl = document.createElement('span');
            tokenEl.className = 'pred-token';
            tokenEl.textContent = p.token;

            var barContainer = document.createElement('div');
            barContainer.className = 'pred-bar-container';
            var bar = document.createElement('div');
            bar.className = 'pred-bar';
            bar.style.width = ((prob / maxProb) * 100).toFixed(1) + '%';
            bar.style.background = barColor;
            barContainer.appendChild(bar);

            var probEl = document.createElement('span');
            probEl.className = 'pred-prob';
            probEl.textContent = (prob * 100).toFixed(1) + '%';

            row.appendChild(tokenEl);
            row.appendChild(barContainer);
            row.appendChild(probEl);
            panel.appendChild(row);
        });

        return panel;
    }

    wrapper.appendChild(buildPanel(baselinePreds, baselineLabel, 'var(--success)'));
    wrapper.appendChild(buildPanel(comparedPreds, comparedLabel, 'var(--warning)'));
    container.appendChild(wrapper);
}

// ---------------------------------------------------------------------------
// Helper: create a collapsible subsection
// ---------------------------------------------------------------------------
function createSubsection(parentEl, title, tag, description, startExpanded) {
    var section = document.createElement('div');
    section.className = 'circuits-subsection' + (startExpanded ? ' expanded' : '');

    var header = document.createElement('div');
    header.className = 'circuits-subsection-header';

    var headerLeft = document.createElement('div');
    headerLeft.style.display = 'flex';
    headerLeft.style.alignItems = 'center';
    headerLeft.style.gap = '10px';

    var h3 = document.createElement('h3');
    h3.textContent = title;

    var tagEl = document.createElement('span');
    tagEl.className = 'subsection-tag';
    tagEl.textContent = tag;

    var expandIcon = document.createElement('span');
    expandIcon.className = 'expand-icon';
    expandIcon.textContent = '\u25BC';

    headerLeft.appendChild(h3);
    headerLeft.appendChild(tagEl);
    header.appendChild(headerLeft);
    header.appendChild(expandIcon);

    var body = document.createElement('div');
    body.className = 'circuits-subsection-body';

    if (description) {
        var desc = document.createElement('div');
        desc.className = 'circuits-subsection-desc';
        desc.textContent = description;
        body.appendChild(desc);
    }

    header.addEventListener('click', function () {
        section.classList.toggle('expanded');
    });

    section.appendChild(header);
    section.appendChild(body);
    parentEl.appendChild(section);

    return body;
}

// ---------------------------------------------------------------------------
// Main init
// ---------------------------------------------------------------------------
function initCircuits() {
    var root = document.getElementById('circuits-container');
    if (!root) return;
    root.innerHTML = '';

    // -----------------------------------------------------------------------
    // Subsection 1: Activation Patching
    // -----------------------------------------------------------------------
    var patchBody = createSubsection(
        root,
        'Activation Patching',
        'causal tracing',
        'Patch activations from a clean run into a corrupted run to measure each component\'s causal contribution. The heatmap shows what fraction of the output is recovered by patching each component at each layer.',
        true
    );

    // Dual prompt inputs
    var dualGroup = document.createElement('div');
    dualGroup.className = 'dual-prompt-group';

    var cleanCol = document.createElement('div');
    cleanCol.className = 'prompt-col';
    var cleanLabel = document.createElement('label');
    cleanLabel.textContent = 'Clean Prompt';
    var cleanInput = document.createElement('input');
    cleanInput.type = 'text';
    cleanInput.placeholder = 'The capital of France is';
    cleanInput.id = 'circuits-patch-clean';
    cleanCol.appendChild(cleanLabel);
    cleanCol.appendChild(cleanInput);

    var corrCol = document.createElement('div');
    corrCol.className = 'prompt-col';
    var corrLabel = document.createElement('label');
    corrLabel.textContent = 'Corrupted Prompt';
    var corrInput = document.createElement('input');
    corrInput.type = 'text';
    corrInput.placeholder = 'The capital of Germany is';
    corrInput.id = 'circuits-patch-corrupted';
    corrCol.appendChild(corrLabel);
    corrCol.appendChild(corrInput);

    dualGroup.appendChild(cleanCol);
    dualGroup.appendChild(corrCol);
    patchBody.appendChild(dualGroup);

    var patchBtn = document.createElement('button');
    patchBtn.className = 'btn btn-primary';
    patchBtn.textContent = 'Run Patching';
    patchBtn.style.marginBottom = '16px';
    patchBody.appendChild(patchBtn);

    var patchResultsId = 'circuits-patch-results';
    var patchResults = document.createElement('div');
    patchResults.className = 'circuits-results';
    patchResults.id = patchResultsId;
    patchBody.appendChild(patchResults);

    patchBtn.addEventListener('click', function () {
        var clean = cleanInput.value.trim();
        var corrupted = corrInput.value.trim();
        if (!clean || !corrupted) return;

        patchResults.innerHTML = '';
        var chartDiv = document.createElement('div');
        chartDiv.id = 'circuits-patch-heatmap';
        chartDiv.style.minHeight = '300px';
        patchResults.appendChild(chartDiv);

        var summaryDiv = document.createElement('div');
        summaryDiv.id = 'circuits-patch-summary';
        patchResults.appendChild(summaryDiv);

        showLoading(patchResultsId);

        apiFetch('/api/circuits/patching', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ clean_prompt: clean, corrupted_prompt: corrupted }),
        })
            .then(function (data) {
                hideLoading(patchResultsId);

                // API returns patching_results: [{layer, residual, attn, ffn}, ...]
                // Convert to 2D array [n_layers x 3 components]
                var components = ['residual', 'attn', 'ffn'];
                var results = data.patching_results || data.recovery || [];
                var recovery;
                if (results.length > 0 && typeof results[0] === 'object' && !Array.isArray(results[0])) {
                    recovery = results.map(function (r) {
                        return [r.residual || 0, r.attn || 0, r.ffn || 0];
                    });
                } else {
                    recovery = results;
                }
                var layerLabels = [];
                for (var i = 0; i < recovery.length; i++) layerLabels.push('Layer ' + i);

                var trace = {
                    z: recovery,
                    x: components,
                    y: layerLabels,
                    type: 'heatmap',
                    colorscale: 'YlOrRd',
                    colorbar: { title: 'Recovery', tickformat: '.0%' },
                    hovertemplate: 'Layer %{y}<br>Component: %{x}<br>Recovery: %{z:.3f}<extra></extra>',
                };

                var layout = darkLayout({
                    title: { text: 'Activation Patching Recovery', font: { size: 14 } },
                    xaxis: { title: 'Component', tickvals: [0, 1, 2], ticktext: components },
                    yaxis: { title: 'Layer', autorange: 'reversed' },
                    height: 400,
                });

                Plotly.newPlot('circuits-patch-heatmap', [trace], layout, window.PLOTLY_CONFIG);

                // Summary: use API max_recovery or compute from matrix
                var maxVal = 0;
                var maxLayer = 0;
                var maxComp = '';
                if (data.max_recovery) {
                    maxVal = data.max_recovery.recovery || 0;
                    maxLayer = data.max_recovery.layer || 0;
                    maxComp = data.max_recovery.component || '';
                } else {
                    for (var li = 0; li < recovery.length; li++) {
                        for (var ci = 0; ci < recovery[li].length; ci++) {
                            if (Math.abs(recovery[li][ci]) > Math.abs(maxVal)) {
                                maxVal = recovery[li][ci];
                                maxLayer = li;
                                maxComp = components[ci];
                            }
                        }
                    }
                }

                summaryDiv.className = 'summary-text';
                summaryDiv.textContent = 'Max recovery: ' + (maxVal * 100).toFixed(1) + '% at Layer ' + maxLayer + ' (' + maxComp + '). ' +
                    'Logit gap (clean \u2212 corrupted): ' + (data.logit_gap || 0).toFixed(3) + ' for target "' + (data.target_token || '') + '".';
            })
            .catch(function (err) {
                hideLoading(patchResultsId);
                showError(patchResultsId, err.message);
            });
    });

    // -----------------------------------------------------------------------
    // Subsection 2: Activation Steering
    // -----------------------------------------------------------------------
    var steerBody = createSubsection(
        root,
        'Activation Steering',
        'intervention',
        'Scale a specific attention head or FFN component\'s activation to observe how it steers model predictions. Compare the baseline output against the steered output.'
    );

    // Prompt input
    var steerPromptGroup = document.createElement('div');
    steerPromptGroup.className = 'prompt-group';
    steerPromptGroup.style.marginBottom = '12px';

    var steerInput = document.createElement('input');
    steerInput.type = 'text';
    steerInput.className = 'prompt-input';
    steerInput.placeholder = 'Enter a prompt for steering...';
    steerInput.id = 'circuits-steer-prompt';
    steerPromptGroup.appendChild(steerInput);
    steerBody.appendChild(steerPromptGroup);

    // Controls row
    var steerControls = document.createElement('div');
    steerControls.className = 'circuits-controls-row';
    steerControls.id = 'circuits-steer-controls';
    steerBody.appendChild(steerControls);

    // Layer dropdown
    var layerOptions = [];
    for (var li = 0; li < window.N_LAYERS; li++) {
        layerOptions.push({ value: String(li), label: 'Layer ' + li });
    }
    var steerLayerSel = createDropdown('circuits-steer-controls', 'Layer', 'circuits-steer-layer', layerOptions);

    // Component radio: head / ffn
    var compRadioGroup = document.createElement('div');
    compRadioGroup.className = 'radio-group';

    var radioHead = document.createElement('input');
    radioHead.type = 'radio';
    radioHead.name = 'steer-component';
    radioHead.value = 'head';
    radioHead.id = 'steer-comp-head';
    radioHead.checked = true;
    var radioHeadLabel = document.createElement('label');
    radioHeadLabel.setAttribute('for', 'steer-comp-head');
    radioHeadLabel.appendChild(radioHead);
    radioHeadLabel.appendChild(document.createTextNode(' Head'));

    var radioFfn = document.createElement('input');
    radioFfn.type = 'radio';
    radioFfn.name = 'steer-component';
    radioFfn.value = 'ffn';
    radioFfn.id = 'steer-comp-ffn';
    var radioFfnLabel = document.createElement('label');
    radioFfnLabel.setAttribute('for', 'steer-comp-ffn');
    radioFfnLabel.appendChild(radioFfn);
    radioFfnLabel.appendChild(document.createTextNode(' FFN'));

    compRadioGroup.appendChild(radioHeadLabel);
    compRadioGroup.appendChild(radioFfnLabel);
    steerControls.appendChild(compRadioGroup);

    // Head dropdown
    var headOptions = [];
    for (var hi = 0; hi < window.N_HEADS; hi++) {
        headOptions.push({ value: String(hi), label: 'Head ' + hi });
    }
    var steerHeadSel = createDropdown('circuits-steer-controls', 'Head', 'circuits-steer-head', headOptions);

    // Scale slider
    var sliderGroup = document.createElement('div');
    sliderGroup.className = 'slider-group';

    var sliderLabel = document.createElement('label');
    sliderLabel.textContent = 'Scale';

    var slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = '2';
    slider.step = '0.1';
    slider.value = '0';
    slider.id = 'circuits-steer-scale';

    var sliderVal = document.createElement('span');
    sliderVal.className = 'slider-value';
    sliderVal.textContent = '0.0';

    slider.addEventListener('input', function () {
        sliderVal.textContent = parseFloat(slider.value).toFixed(1);
    });

    sliderGroup.appendChild(sliderLabel);
    sliderGroup.appendChild(slider);
    sliderGroup.appendChild(sliderVal);
    steerControls.appendChild(sliderGroup);

    // Steer button
    var steerBtn = document.createElement('button');
    steerBtn.className = 'btn btn-primary';
    steerBtn.textContent = 'Run Steering';
    steerBtn.style.marginBottom = '16px';
    steerBody.appendChild(steerBtn);

    var steerResultsId = 'circuits-steer-results';
    var steerResults = document.createElement('div');
    steerResults.className = 'circuits-results';
    steerResults.id = steerResultsId;
    steerBody.appendChild(steerResults);

    // Toggle head dropdown visibility based on component radio
    function updateHeadDropdownVisibility() {
        var isHead = document.querySelector('input[name="steer-component"]:checked').value === 'head';
        steerHeadSel.parentElement.style.display = isHead ? '' : 'none';
    }
    radioHead.addEventListener('change', updateHeadDropdownVisibility);
    radioFfn.addEventListener('change', updateHeadDropdownVisibility);
    updateHeadDropdownVisibility();

    steerBtn.addEventListener('click', function () {
        var prompt = steerInput.value.trim();
        if (!prompt) return;

        var layer = parseInt(steerLayerSel.value, 10);
        var component = document.querySelector('input[name="steer-component"]:checked').value;
        var head = parseInt(steerHeadSel.value, 10);
        var scale = parseFloat(slider.value);

        steerResults.innerHTML = '';
        showLoading(steerResultsId);

        apiFetch('/api/circuits/steering', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                layer: layer,
                component: component,
                head: head,
                scale: scale,
            }),
        })
            .then(function (data) {
                hideLoading(steerResultsId);

                // KL divergence badge
                if (data.kl_divergence !== undefined) {
                    var klBadge = document.createElement('div');
                    klBadge.className = 'kl-badge';
                    klBadge.innerHTML =
                        '<span class="kl-label">KL Divergence</span>' +
                        '<span class="kl-value">' + escapeHtml(data.kl_divergence.toFixed(4)) + '</span>';
                    steerResults.appendChild(klBadge);
                }

                // Side-by-side comparison
                var comparisonId = 'circuits-steer-comparison';
                var compDiv = document.createElement('div');
                compDiv.id = comparisonId;
                steerResults.appendChild(compDiv);

                renderPredictionComparison(
                    comparisonId,
                    data.baseline_predictions,
                    data.steered_predictions,
                    'Baseline',
                    'Steered (scale=' + scale.toFixed(1) + ')'
                );
            })
            .catch(function (err) {
                hideLoading(steerResultsId);
                showError(steerResultsId, err.message);
            });
    });

    // -----------------------------------------------------------------------
    // Subsection 3: Activation Swapping
    // -----------------------------------------------------------------------
    var swapBody = createSubsection(
        root,
        'Activation Swapping',
        'transfer',
        'Swap activations from a source prompt into a target prompt at a chosen layer and component. Observe how the target prompt\'s predictions change when receiving the source\'s internal representations.'
    );

    // Dual prompt inputs
    var swapDualGroup = document.createElement('div');
    swapDualGroup.className = 'dual-prompt-group';

    var srcCol = document.createElement('div');
    srcCol.className = 'prompt-col';
    var srcLabel = document.createElement('label');
    srcLabel.textContent = 'Source Prompt';
    var srcInput = document.createElement('input');
    srcInput.type = 'text';
    srcInput.placeholder = 'The cat sat on the';
    srcInput.id = 'circuits-swap-source';
    srcCol.appendChild(srcLabel);
    srcCol.appendChild(srcInput);

    var tgtCol = document.createElement('div');
    tgtCol.className = 'prompt-col';
    var tgtLabel = document.createElement('label');
    tgtLabel.textContent = 'Target Prompt';
    var tgtInput = document.createElement('input');
    tgtInput.type = 'text';
    tgtInput.placeholder = 'The dog ran through the';
    tgtInput.id = 'circuits-swap-target';
    tgtCol.appendChild(tgtLabel);
    tgtCol.appendChild(tgtInput);

    swapDualGroup.appendChild(srcCol);
    swapDualGroup.appendChild(tgtCol);
    swapBody.appendChild(swapDualGroup);

    // Controls row
    var swapControls = document.createElement('div');
    swapControls.className = 'circuits-controls-row';
    swapControls.id = 'circuits-swap-controls';
    swapBody.appendChild(swapControls);

    // Layer dropdown
    var swapLayerOptions = [];
    for (var sli = 0; sli < window.N_LAYERS; sli++) {
        swapLayerOptions.push({ value: String(sli), label: 'Layer ' + sli });
    }
    var swapLayerSel = createDropdown('circuits-swap-controls', 'Layer', 'circuits-swap-layer', swapLayerOptions);

    // Component radio: residual / attn / ffn
    var swapRadioGroup = document.createElement('div');
    swapRadioGroup.className = 'radio-group';

    var swapRadioResid = document.createElement('input');
    swapRadioResid.type = 'radio';
    swapRadioResid.name = 'swap-component';
    swapRadioResid.value = 'residual';
    swapRadioResid.id = 'swap-comp-residual';
    swapRadioResid.checked = true;
    var swapResidLabel = document.createElement('label');
    swapResidLabel.setAttribute('for', 'swap-comp-residual');
    swapResidLabel.appendChild(swapRadioResid);
    swapResidLabel.appendChild(document.createTextNode(' Residual'));

    var swapRadioAttn = document.createElement('input');
    swapRadioAttn.type = 'radio';
    swapRadioAttn.name = 'swap-component';
    swapRadioAttn.value = 'attn';
    swapRadioAttn.id = 'swap-comp-attn';
    var swapAttnLabel = document.createElement('label');
    swapAttnLabel.setAttribute('for', 'swap-comp-attn');
    swapAttnLabel.appendChild(swapRadioAttn);
    swapAttnLabel.appendChild(document.createTextNode(' Attention'));

    var swapRadioFfn = document.createElement('input');
    swapRadioFfn.type = 'radio';
    swapRadioFfn.name = 'swap-component';
    swapRadioFfn.value = 'ffn';
    swapRadioFfn.id = 'swap-comp-ffn';
    var swapFfnLabel = document.createElement('label');
    swapFfnLabel.setAttribute('for', 'swap-comp-ffn');
    swapFfnLabel.appendChild(swapRadioFfn);
    swapFfnLabel.appendChild(document.createTextNode(' FFN'));

    swapRadioGroup.appendChild(swapResidLabel);
    swapRadioGroup.appendChild(swapAttnLabel);
    swapRadioGroup.appendChild(swapFfnLabel);
    swapControls.appendChild(swapRadioGroup);

    // Swap button
    var swapBtn = document.createElement('button');
    swapBtn.className = 'btn btn-primary';
    swapBtn.textContent = 'Run Swapping';
    swapBtn.style.marginBottom = '16px';
    swapBody.appendChild(swapBtn);

    var swapResultsId = 'circuits-swap-results';
    var swapResults = document.createElement('div');
    swapResults.className = 'circuits-results';
    swapResults.id = swapResultsId;
    swapBody.appendChild(swapResults);

    swapBtn.addEventListener('click', function () {
        var source = srcInput.value.trim();
        var target = tgtInput.value.trim();
        if (!source || !target) return;

        var layer = parseInt(swapLayerSel.value, 10);
        var component = document.querySelector('input[name="swap-component"]:checked').value;

        swapResults.innerHTML = '';
        showLoading(swapResultsId);

        apiFetch('/api/circuits/swapping', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_prompt: source,
                target_prompt: target,
                layer: layer,
                component: component,
            }),
        })
            .then(function (data) {
                hideLoading(swapResultsId);

                // KL divergence badge
                if (data.kl_divergence !== undefined) {
                    var klBadge = document.createElement('div');
                    klBadge.className = 'kl-badge';
                    klBadge.innerHTML =
                        '<span class="kl-label">KL Divergence</span>' +
                        '<span class="kl-value">' + escapeHtml(data.kl_divergence.toFixed(4)) + '</span>';
                    swapResults.appendChild(klBadge);
                }

                // Side-by-side comparison
                var comparisonId = 'circuits-swap-comparison';
                var compDiv = document.createElement('div');
                compDiv.id = comparisonId;
                swapResults.appendChild(compDiv);

                renderPredictionComparison(
                    comparisonId,
                    data.baseline_predictions,
                    data.swapped_predictions,
                    'Baseline (target)',
                    'Swapped (source \u2192 target, L' + layer + ' ' + component + ')'
                );
            })
            .catch(function (err) {
                hideLoading(swapResultsId);
                showError(swapResultsId, err.message);
            });
    });

    // -----------------------------------------------------------------------
    // Subsection 4: Pre-computation Detection
    // -----------------------------------------------------------------------
    var precompBody = createSubsection(
        root,
        'Pre-computation Detection',
        'lookahead',
        'Detect whether the model pre-computes future tokens in earlier layers. The heatmap shows, for each position, the earliest layer where future tokens (+2 to +5 ahead) can be linearly decoded from the residual stream.'
    );

    // Prompt input
    var precompPromptGroup = document.createElement('div');
    precompPromptGroup.className = 'prompt-group';
    precompPromptGroup.style.marginBottom = '12px';

    var precompInput = document.createElement('input');
    precompInput.type = 'text';
    precompInput.className = 'prompt-input';
    precompInput.placeholder = 'Enter a prompt for pre-computation detection...';
    precompInput.id = 'circuits-precomp-prompt';
    precompPromptGroup.appendChild(precompInput);
    precompBody.appendChild(precompPromptGroup);

    var precompBtn = document.createElement('button');
    precompBtn.className = 'btn btn-primary';
    precompBtn.textContent = 'Detect';
    precompBtn.style.marginBottom = '16px';
    precompBody.appendChild(precompBtn);

    var precompResultsId = 'circuits-precomp-results';
    var precompResults = document.createElement('div');
    precompResults.className = 'circuits-results';
    precompResults.id = precompResultsId;
    precompBody.appendChild(precompResults);

    precompBtn.addEventListener('click', function () {
        var prompt = precompInput.value.trim();
        if (!prompt) return;

        precompResults.innerHTML = '';
        var chartDiv = document.createElement('div');
        chartDiv.id = 'circuits-precomp-heatmap';
        chartDiv.style.minHeight = '300px';
        precompResults.appendChild(chartDiv);

        var findingsDiv = document.createElement('div');
        findingsDiv.id = 'circuits-precomp-findings';
        precompResults.appendChild(findingsDiv);

        showLoading(precompResultsId);

        apiFetch('/api/circuits/precomputation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt }),
        })
            .then(function (data) {
                hideLoading(precompResultsId);

                // API returns precomputation_matrix (or heatmap) and findings
                var heatmap = data.precomputation_matrix || data.heatmap || [];
                var tokens = data.tokens || [];
                var offsets = (data.future_offsets || [2, 3, 4, 5]).map(function (o) { return '+' + o; });

                var posLabels = [];
                for (var ti = 0; ti < heatmap.length; ti++) {
                    posLabels.push(tokens[ti] ? escapeHtml(tokens[ti]) : 'pos ' + ti);
                }

                var trace = {
                    z: heatmap,
                    x: offsets,
                    y: posLabels,
                    type: 'heatmap',
                    colorscale: [
                        [0, '#111113'],
                        [0.25, '#1e1e2e'],
                        [0.5, '#f87171'],
                        [0.75, '#fbbf24'],
                        [1, '#34d399'],
                    ],
                    colorbar: { title: 'Earliest Layer' },
                    hovertemplate: 'Position: %{y}<br>Offset: %{x}<br>Earliest Layer: %{z}<extra></extra>',
                };

                var layout = darkLayout({
                    title: { text: 'Pre-computation Detection: Earliest Decodable Layer', font: { size: 14 } },
                    xaxis: { title: 'Future Offset' },
                    yaxis: { title: 'Position', autorange: 'reversed' },
                    height: Math.max(350, heatmap.length * 28 + 100),
                });

                Plotly.newPlot('circuits-precomp-heatmap', [trace], layout, window.PLOTLY_CONFIG);

                // Findings list (up to 10)
                var findings = data.findings || [];
                if (findings.length > 0) {
                    var findingsContainer = document.getElementById('circuits-precomp-findings');
                    findingsContainer.style.marginTop = '16px';

                    var findingsTitle = document.createElement('div');
                    findingsTitle.className = 'summary-text';
                    findingsTitle.style.marginBottom = '12px';
                    findingsTitle.textContent = 'Detected ' + findings.length + ' pre-computation signal' + (findings.length > 1 ? 's' : '') + ':';
                    findingsContainer.appendChild(findingsTitle);

                    var ul = document.createElement('ul');
                    ul.className = 'findings-list';

                    var displayCount = Math.min(findings.length, 10);
                    for (var fi = 0; fi < displayCount; fi++) {
                        var f = findings[fi];
                        var li = document.createElement('li');

                        // Depth badge — API uses first_depth_idx or earliest_layer
                        var depthIdx = f.first_depth_idx !== undefined ? f.first_depth_idx : (f.earliest_layer || 0);
                        var badge = document.createElement('span');
                        badge.className = 'finding-depth-badge';
                        var depthClass = 'early';
                        if (depthIdx >= Math.floor(window.N_LAYERS / 3) && depthIdx < Math.floor(2 * window.N_LAYERS / 3)) {
                            depthClass = 'mid';
                        } else if (depthIdx >= Math.floor(2 * window.N_LAYERS / 3)) {
                            depthClass = 'late';
                        }
                        badge.classList.add(depthClass);
                        var depthLabel = f.first_depth || ('L' + depthIdx);
                        badge.textContent = depthLabel + ' (' + depthClass + ')';

                        var offset = f.future_offset || f.offset || '?';
                        var text = document.createElement('span');
                        text.textContent = f.text || (
                            'Position ' + f.position + ' "' + (f.token || '') + '" pre-computes "' + (f.future_token || '') + '" at offset +' + offset
                        );

                        li.appendChild(badge);
                        li.appendChild(text);
                        ul.appendChild(li);
                    }

                    findingsContainer.appendChild(ul);
                }
            })
            .catch(function (err) {
                hideLoading(precompResultsId);
                showError(precompResultsId, err.message);
            });
    });
}
