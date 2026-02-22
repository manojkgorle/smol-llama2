/**
 * embeddings.js — Embedding Space + Token Explorer tab.
 *
 * Views:
 *   1. Embedding Space — PCA scatter of vocab tokens + prompt trajectory
 *   2. Token Explorer — tokenization breakdown, vocabulary search, nearest neighbors
 */

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------
var _embState = {
    initialized: false,
    data: null,
    currentView: 'embedding-space',
};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function initEmbeddings() {
    if (_embState.initialized) return;
    _embState.initialized = true;

    var controls = document.getElementById('embeddings-controls');
    if (!controls) return;

    // Prompt input
    createPromptInput('embeddings-controls', function (prompt) {
        analyzeEmbeddings(prompt);
    }, { placeholder: 'Enter a prompt for embedding analysis...', buttonLabel: 'Analyze' });

    // View switcher
    var switcher = document.createElement('div');
    switcher.className = 'view-switcher';
    switcher.id = 'emb-view-switcher';

    var btnSpace = document.createElement('button');
    btnSpace.className = 'active';
    btnSpace.textContent = 'Embedding Space';
    btnSpace.dataset.view = 'embedding-space';

    var btnExplorer = document.createElement('button');
    btnExplorer.textContent = 'Token Explorer';
    btnExplorer.dataset.view = 'token-explorer';

    switcher.appendChild(btnSpace);
    switcher.appendChild(btnExplorer);
    controls.appendChild(switcher);

    // Help icons
    controls.appendChild(createHelpIcon('Embedding Space',
        'PCA projects the model\'s <strong>' + (window.vizState.model_info ? window.vizState.model_info.dim : 384) +
        '-dimensional</strong> embedding vectors into 2D for visualization. ' +
        'Each point represents a token in the vocabulary. Prompt tokens are highlighted and their trajectory through layers is shown.'
    ));

    switcher.querySelectorAll('button').forEach(function (btn) {
        btn.addEventListener('click', function () {
            switcher.querySelectorAll('button').forEach(function (b) { b.classList.remove('active'); });
            btn.classList.add('active');
            _embState.currentView = btn.dataset.view;
            _renderEmbeddingView();
        });
    });

    // Build view area
    var viewArea = document.getElementById('embeddings-view');
    if (viewArea) {
        viewArea.innerHTML =
            '<div id="emb-guide" class="tab-guide">' +
                '<div class="guide-title">Embedding Space & Token Explorer</div>' +
                '<p>Visualize how the model represents tokens in <strong>embedding space</strong> and explore the tokenizer\'s vocabulary.</p>' +
                '<div class="guide-features">' +
                    '<div class="guide-item"><span class="guide-tag">PCA</span>2D projection of all vocabulary embeddings with prompt token trajectory through layers.</div>' +
                    '<div class="guide-item"><span class="guide-tag">explorer</span>Tokenize text, search the vocabulary, and find nearest neighbor tokens.</div>' +
                '</div>' +
                '<p class="guide-hint">Enter a prompt above and click <strong>Analyze</strong> to begin.</p>' +
            '</div>' +
            '<div id="emb-main-area" class="chart-grid"></div>';
    }
}

// ---------------------------------------------------------------------------
// Fetch
// ---------------------------------------------------------------------------
async function analyzeEmbeddings(prompt) {
    var guide = document.getElementById('emb-guide');
    if (guide) guide.style.display = 'none';

    var mainArea = document.getElementById('emb-main-area');
    if (!mainArea) return;
    mainArea.innerHTML = '';

    showLoading('embeddings-view');

    try {
        var data = await apiFetch('/api/embeddings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt }),
        });
        _embState.data = data;
        _embState.prompt = prompt;
        hideLoading('embeddings-view');
        _renderEmbeddingView();
    } catch (err) {
        hideLoading('embeddings-view');
        showError('emb-main-area', 'Analysis failed: ' + err.message);
    }
}

// ---------------------------------------------------------------------------
// View router
// ---------------------------------------------------------------------------
function _renderEmbeddingView() {
    var mainArea = document.getElementById('emb-main-area');
    if (!mainArea) return;
    mainArea.innerHTML = '';

    if (_embState.currentView === 'embedding-space') {
        _renderEmbeddingSpace(mainArea);
    } else {
        _renderTokenExplorer(mainArea);
    }
}

// ---------------------------------------------------------------------------
// VIEW 1: Embedding Space — PCA scatter + trajectory
// ---------------------------------------------------------------------------
function _renderEmbeddingSpace(container) {
    var data = _embState.data;
    if (!data || !data.vocab_points) {
        container.innerHTML = '<div class="placeholder-message">Submit a prompt to visualize embedding space.</div>';
        return;
    }

    // Info badges row
    var badgeRow = document.createElement('div');
    badgeRow.style.cssText = 'display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px;';

    if (data.explained_variance) {
        var varBadge = document.createElement('div');
        varBadge.className = 'kl-badge';
        var totalVar = (data.explained_variance[0] + data.explained_variance[1]) * 100;
        varBadge.innerHTML =
            '<span class="kl-label">PCA Variance</span>' +
            '<span class="kl-value">' + totalVar.toFixed(1) + '%</span>' +
            '<span class="kl-label" style="margin-left:8px;">PC1: ' + (data.explained_variance[0] * 100).toFixed(1) + '% PC2: ' + (data.explained_variance[1] * 100).toFixed(1) + '%</span>';
        badgeRow.appendChild(varBadge);
    }

    var vocabBadge = document.createElement('div');
    vocabBadge.className = 'kl-badge';
    vocabBadge.innerHTML =
        '<span class="kl-label">Tokens</span>' +
        '<span class="kl-value">' + data.vocab_points.length + ' vocab</span>' +
        '<span class="kl-label" style="margin-left:8px;">' + data.tokens.length + ' prompt</span>';
    badgeRow.appendChild(vocabBadge);
    container.appendChild(badgeRow);

    // Main scatter plot card
    var scatterCard = document.createElement('div');
    scatterCard.className = 'chart-card full-width';
    var scatterTitle = document.createElement('div');
    scatterTitle.className = 'chart-title';
    scatterTitle.textContent = 'Vocabulary Embedding Space (PCA)';
    scatterTitle.appendChild(createHelpIcon('Embedding Space',
        'Each gray point is a token from the vocabulary projected to 2D via PCA. ' +
        '<strong>Colored lines</strong> show how prompt tokens move through the model\'s layers — from the initial embedding through each transformer block. ' +
        'Large jumps indicate layers where the model significantly transforms the representation. ' +
        'Numbered markers show the layer depth (E=embedding, 0-7=layers). Use the scroll wheel to zoom.'
    ));
    scatterCard.appendChild(scatterTitle);

    var scatterDiv = document.createElement('div');
    scatterDiv.id = 'emb-scatter-chart';
    scatterDiv.style.minHeight = '600px';
    scatterCard.appendChild(scatterDiv);
    container.appendChild(scatterCard);

    // Compute density-based sizing for vocab points
    var vocabX = data.vocab_points.map(function (p) { return p[0]; });
    var vocabY = data.vocab_points.map(function (p) { return p[1]; });

    var traces = [];

    // Vocab background: use density2d contour as a background + scatter overlay
    traces.push({
        x: vocabX,
        y: vocabY,
        type: 'histogram2dcontour',
        colorscale: [
            [0, 'rgba(9,9,11,0)'],
            [0.2, 'rgba(34,211,238,0.03)'],
            [0.5, 'rgba(34,211,238,0.08)'],
            [0.8, 'rgba(34,211,238,0.15)'],
            [1.0, 'rgba(34,211,238,0.25)'],
        ],
        showscale: false,
        ncontours: 20,
        line: { width: 0 },
        name: 'Density',
        hoverinfo: 'skip',
    });

    // Vocab scatter points
    traces.push({
        x: vocabX,
        y: vocabY,
        text: data.vocab_labels,
        type: 'scattergl',
        mode: 'markers',
        marker: {
            size: 2.5,
            color: 'rgba(113, 113, 122, 0.25)',
        },
        name: 'Vocab (' + vocabX.length + ')',
        hovertemplate: '<b>%{text}</b><extra></extra>',
        showlegend: true,
    });

    // Prompt token trajectories — the key visualization
    if (data.trajectory && data.trajectory.length > 0) {
        var tokens = data.tokens;
        var nDepths = data.trajectory.length;

        for (var ti = 0; ti < tokens.length; ti++) {
            var trajX = [];
            var trajY = [];
            var trajLabels = [];
            var markerSizes = [];
            var markerSymbols = [];

            for (var di = 0; di < nDepths; di++) {
                var pts = data.trajectory[di].points;
                if (ti < pts.length) {
                    trajX.push(pts[ti][0]);
                    trajY.push(pts[ti][1]);
                    var depthLabel = data.trajectory[di].depth;
                    var readableDepth = depthLabel === 'embedding' ? 'Emb' : 'L' + depthLabel;
                    trajLabels.push(tokens[ti] + ' @ ' + readableDepth);
                    // Make embedding point smaller, layer points progressively larger
                    markerSizes.push(di === 0 ? 7 : 8 + di * 0.5);
                    markerSymbols.push(di === 0 ? 'diamond' : 'circle');
                }
            }

            if (trajX.length === 0) continue;

            var color = window.LAYER_COLORS[ti % window.LAYER_COLORS.length];

            // Trajectory line (semi-transparent, thicker)
            traces.push({
                x: trajX,
                y: trajY,
                text: trajLabels,
                type: 'scatter',
                mode: 'lines',
                line: { color: color, width: 2.5, shape: 'spline' },
                opacity: 0.6,
                name: tokens[ti] + ' (path)',
                hoverinfo: 'skip',
                showlegend: false,
                legendgroup: 'tok-' + ti,
            });

            // Trajectory markers with layer annotations
            var markerColors = [];
            for (var mi = 0; mi < trajX.length; mi++) {
                markerColors.push(mi === 0 ? '#ffffff' : color);
            }

            traces.push({
                x: trajX,
                y: trajY,
                text: trajLabels,
                customdata: trajLabels.map(function (_, idx) {
                    return idx === 0 ? 'E' : String(idx - 1);
                }),
                type: 'scatter',
                mode: 'markers+text',
                marker: {
                    size: markerSizes,
                    color: markerColors,
                    line: { color: color, width: 2 },
                    symbol: markerSymbols,
                },
                textposition: 'top center',
                textfont: { size: 9, color: color, family: 'JetBrains Mono, monospace' },
                texttemplate: '%{customdata}',
                name: tokens[ti],
                hovertemplate: '<b>%{text}</b><extra></extra>',
                showlegend: true,
                legendgroup: 'tok-' + ti,
            });

            // Start marker annotation (arrow from embedding to first layer)
            if (trajX.length > 1) {
                // Add an arrowhead annotation from embedding to layer 0
                // This will be added as a layout annotation below
            }
        }
    }

    // Build layout annotations for start points
    var annotations = [];
    if (data.trajectory && data.trajectory.length > 1 && data.tokens) {
        for (var ai = 0; ai < data.tokens.length; ai++) {
            var emPts = data.trajectory[0].points;
            if (ai < emPts.length) {
                annotations.push({
                    x: emPts[ai][0],
                    y: emPts[ai][1],
                    text: data.tokens[ai],
                    showarrow: true,
                    arrowhead: 0,
                    arrowwidth: 1,
                    arrowcolor: window.LAYER_COLORS[ai % window.LAYER_COLORS.length],
                    ax: 0,
                    ay: -30,
                    font: {
                        size: 11,
                        color: window.LAYER_COLORS[ai % window.LAYER_COLORS.length],
                        family: 'JetBrains Mono, monospace',
                    },
                    bgcolor: 'rgba(9,9,11,0.8)',
                    borderpad: 3,
                });
            }
        }
    }

    var layout = darkLayout({
        xaxis: {
            title: 'PC1',
            gridcolor: 'rgba(255,255,255,0.04)',
            zeroline: false,
        },
        yaxis: {
            title: 'PC2',
            gridcolor: 'rgba(255,255,255,0.04)',
            zeroline: false,
            scaleanchor: 'x',
            scaleratio: 1,
        },
        height: 650,
        hovermode: 'closest',
        annotations: annotations,
        legend: {
            x: 1.02, y: 1,
            bgcolor: 'rgba(9,9,11,0.92)',
            bordercolor: 'rgba(255,255,255,0.08)',
            borderwidth: 1,
            font: { size: 10, color: '#a1a1aa' },
        },
    });

    var config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        displaylogo: false,
        scrollZoom: true,
    };

    Plotly.newPlot('emb-scatter-chart', traces, layout, config);

    // Click handler for vocab points — find neighbors
    var chartEl = document.getElementById('emb-scatter-chart');
    chartEl.on('plotly_click', function (clickData) {
        if (clickData.points && clickData.points.length > 0) {
            var pt = clickData.points[0];
            // Only handle vocab scatter clicks (trace index 1)
            if (pt.curveNumber === 1 && pt.pointIndex < data.vocab_labels.length) {
                _findNeighbors(pt.pointIndex);
            }
        }
    });

    // Trajectory distance chart
    _renderTrajectoryDistances(container, data);
}

// ---------------------------------------------------------------------------
// Trajectory distances — how much each token moves per layer
// ---------------------------------------------------------------------------
function _renderTrajectoryDistances(container, data) {
    if (!data.trajectory || data.trajectory.length < 2) return;

    var trajCard = document.createElement('div');
    trajCard.className = 'chart-card full-width';
    var trajTitle = document.createElement('div');
    trajTitle.className = 'chart-title';
    trajTitle.textContent = 'Token Movement Per Layer (Euclidean distance in PCA space)';
    trajTitle.appendChild(createHelpIcon('Movement Distance',
        'Shows how much each token\'s representation <strong>moves in embedding space</strong> at each layer. ' +
        'Large spikes indicate layers where the model significantly transforms that token\'s meaning. ' +
        'This reveals where the "work" happens for each token.'
    ));
    trajCard.appendChild(trajTitle);

    var chartDiv = document.createElement('div');
    chartDiv.id = 'emb-trajectory-dist-chart';
    chartDiv.className = 'chart-container';
    trajCard.appendChild(chartDiv);
    container.appendChild(trajCard);

    var tokens = data.tokens;
    var nDepths = data.trajectory.length;
    var traces = [];

    for (var ti = 0; ti < tokens.length; ti++) {
        var distances = [];
        var labels = [];

        for (var di = 1; di < nDepths; di++) {
            var prev = data.trajectory[di - 1].points;
            var curr = data.trajectory[di].points;
            if (ti < prev.length && ti < curr.length) {
                var dx = curr[ti][0] - prev[ti][0];
                var dy = curr[ti][1] - prev[ti][1];
                distances.push(Math.sqrt(dx * dx + dy * dy));
                var depthLabel = data.trajectory[di].depth;
                labels.push(depthLabel === 'embedding' ? 'Emb' : 'L' + depthLabel);
            }
        }

        var color = window.LAYER_COLORS[ti % window.LAYER_COLORS.length];
        traces.push({
            x: labels,
            y: distances,
            type: 'scatter',
            mode: 'lines+markers',
            name: tokens[ti],
            line: { color: color, width: 2 },
            marker: { size: 6, color: color },
            hovertemplate: tokens[ti] + ' @ %{x}<br>Distance: %{y:.4f}<extra></extra>',
        });
    }

    Plotly.newPlot('emb-trajectory-dist-chart', traces, darkLayout({
        xaxis: { title: 'Layer', gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'Distance Moved', gridcolor: 'rgba(255,255,255,0.06)' },
        height: 350,
        legend: {
            x: 1.02, y: 1,
            bgcolor: 'rgba(9,9,11,0.9)',
            bordercolor: 'rgba(255,255,255,0.06)',
            borderwidth: 1,
            font: { size: 10 },
        },
    }), window.PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// VIEW 2: Token Explorer
// ---------------------------------------------------------------------------
function _renderTokenExplorer(container) {
    // Tokenize section
    var tokenizeCard = document.createElement('div');
    tokenizeCard.className = 'chart-card full-width';

    var tokenizeTitle = document.createElement('div');
    tokenizeTitle.className = 'chart-title';
    tokenizeTitle.textContent = 'Tokenizer';
    tokenizeTitle.appendChild(createHelpIcon('Tokenizer',
        'The tokenizer breaks text into <strong>tokens</strong> — the basic units the model processes. ' +
        'Common words become single tokens, rare words are split into pieces. ' +
        'The BOS token (&lt;s&gt;) is always prepended.'
    ));
    tokenizeCard.appendChild(tokenizeTitle);

    var tokenizeInput = document.createElement('div');
    tokenizeInput.className = 'token-explorer-input';

    var tokenizeText = document.createElement('input');
    tokenizeText.type = 'text';
    tokenizeText.placeholder = 'Enter text to tokenize...';
    tokenizeText.id = 'token-explorer-text';
    if (_embState.prompt) tokenizeText.value = _embState.prompt;

    var tokenizeBtn = document.createElement('button');
    tokenizeBtn.className = 'btn btn-primary';
    tokenizeBtn.textContent = 'Tokenize';

    tokenizeInput.appendChild(tokenizeText);
    tokenizeInput.appendChild(tokenizeBtn);
    tokenizeCard.appendChild(tokenizeInput);

    var tokenizeResults = document.createElement('div');
    tokenizeResults.id = 'token-explorer-results';
    tokenizeCard.appendChild(tokenizeResults);
    container.appendChild(tokenizeCard);

    tokenizeBtn.addEventListener('click', function () {
        var text = tokenizeText.value.trim();
        if (!text) return;
        _tokenizeText(text);
    });

    tokenizeText.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
            var text = tokenizeText.value.trim();
            if (text) _tokenizeText(text);
        }
    });

    // Search section
    var searchCard = document.createElement('div');
    searchCard.className = 'chart-card full-width';

    var searchTitle = document.createElement('div');
    searchTitle.className = 'chart-title';
    searchTitle.textContent = 'Vocabulary Search';
    searchCard.appendChild(searchTitle);

    var searchInput = document.createElement('div');
    searchInput.className = 'token-explorer-input';

    var searchText = document.createElement('input');
    searchText.type = 'text';
    searchText.placeholder = 'Search vocabulary (e.g. "cat", "the")...';
    searchText.id = 'token-explorer-search';

    var searchBtn = document.createElement('button');
    searchBtn.className = 'btn btn-primary';
    searchBtn.textContent = 'Search';

    searchInput.appendChild(searchText);
    searchInput.appendChild(searchBtn);
    searchCard.appendChild(searchInput);

    var searchResults = document.createElement('div');
    searchResults.id = 'token-explorer-search-results';
    searchCard.appendChild(searchResults);
    container.appendChild(searchCard);

    searchBtn.addEventListener('click', function () {
        var query = searchText.value.trim();
        if (!query) return;
        _searchVocab(query);
    });

    searchText.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
            var query = searchText.value.trim();
            if (query) _searchVocab(query);
        }
    });

    // Neighbors section
    var neighborsCard = document.createElement('div');
    neighborsCard.className = 'chart-card full-width';

    var neighborsTitle = document.createElement('div');
    neighborsTitle.className = 'chart-title';
    neighborsTitle.textContent = 'Nearest Neighbors';
    neighborsTitle.appendChild(createHelpIcon('Nearest Neighbors',
        'Shows tokens whose embedding vectors are closest (by <strong>cosine similarity</strong>) to the selected token. ' +
        'Tokens that are near neighbors often share semantic or syntactic properties.'
    ));
    neighborsCard.appendChild(neighborsTitle);

    var neighborsInput = document.createElement('div');
    neighborsInput.className = 'token-explorer-input';

    var neighborsText = document.createElement('input');
    neighborsText.type = 'text';
    neighborsText.placeholder = 'Enter a token ID (e.g. 42)...';
    neighborsText.id = 'token-explorer-neighbor-id';

    var neighborsBtn = document.createElement('button');
    neighborsBtn.className = 'btn btn-primary';
    neighborsBtn.textContent = 'Find Neighbors';

    neighborsInput.appendChild(neighborsText);
    neighborsInput.appendChild(neighborsBtn);
    neighborsCard.appendChild(neighborsInput);

    var neighborsResults = document.createElement('div');
    neighborsResults.id = 'token-explorer-neighbors-results';
    neighborsCard.appendChild(neighborsResults);
    container.appendChild(neighborsCard);

    neighborsBtn.addEventListener('click', function () {
        var tokenId = parseInt(neighborsText.value.trim(), 10);
        if (isNaN(tokenId)) return;
        _findNeighbors(tokenId);
    });

    // Auto-tokenize if we already have a prompt
    if (_embState.prompt) {
        _tokenizeText(_embState.prompt);
    }
}

async function _tokenizeText(text) {
    var results = document.getElementById('token-explorer-results');
    if (!results) return;
    showInlineLoading('token-explorer-results');

    try {
        var data = await apiFetch('/api/tokenizer/encode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text }),
        });

        var html = '<div style="margin-bottom:8px;font-size:12px;color:var(--text-muted);font-family:var(--font-mono);">' +
            data.num_tokens + ' tokens</div>';
        html += '<div class="token-strip">';

        var colors = window.LAYER_COLORS;
        for (var i = 0; i < data.token_strings.length; i++) {
            var color = colors[i % colors.length];
            html += '<span class="token-span" style="background:' + color + '22;border-color:' + color + '44;cursor:pointer;" ' +
                'title="ID: ' + data.token_ids[i] + '" data-token-id="' + data.token_ids[i] + '">' +
                escapeHtml(data.token_strings[i]) +
                '<span style="font-size:9px;color:var(--text-muted);margin-left:4px;">' + data.token_ids[i] + '</span></span>';
        }
        html += '</div>';
        results.innerHTML = html;

        // Click token to find neighbors
        results.querySelectorAll('.token-span[data-token-id]').forEach(function (span) {
            span.addEventListener('click', function () {
                var tid = parseInt(span.dataset.tokenId, 10);
                var input = document.getElementById('token-explorer-neighbor-id');
                if (input) input.value = tid;
                _findNeighbors(tid);
            });
        });
    } catch (err) {
        showError('token-explorer-results', err.message);
    }
}

async function _searchVocab(query) {
    var results = document.getElementById('token-explorer-search-results');
    if (!results) return;
    showInlineLoading('token-explorer-search-results');

    try {
        var data = await apiFetch('/api/tokenizer/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query }),
        });

        if (data.results.length === 0) {
            results.innerHTML = '<div class="placeholder-message">No results found for "' + escapeHtml(query) + '"</div>';
            return;
        }

        var html = '<div class="token-strip">';
        for (var i = 0; i < data.results.length; i++) {
            var r = data.results[i];
            html += '<span class="token-span" style="background:rgba(255,255,255,0.04);cursor:pointer;" ' +
                'title="ID: ' + r.id + '" data-token-id="' + r.id + '">' +
                escapeHtml(r.token) +
                '<span style="font-size:9px;color:var(--text-muted);margin-left:4px;">' + r.id + '</span></span>';
        }
        html += '</div>';
        results.innerHTML = html;

        results.querySelectorAll('.token-span[data-token-id]').forEach(function (span) {
            span.addEventListener('click', function () {
                var tid = parseInt(span.dataset.tokenId, 10);
                var input = document.getElementById('token-explorer-neighbor-id');
                if (input) input.value = tid;
                _findNeighbors(tid);
            });
        });
    } catch (err) {
        showError('token-explorer-search-results', err.message);
    }
}

async function _findNeighbors(tokenId) {
    var results = document.getElementById('token-explorer-neighbors-results');
    if (!results) return;
    showInlineLoading('token-explorer-neighbors-results');

    try {
        var data = await apiFetch('/api/tokenizer/neighbors', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token_id: tokenId }),
        });

        var html = '<div style="margin-bottom:8px;font-size:13px;color:var(--text-primary);">Neighbors of <strong style="color:var(--highlight);">' +
            escapeHtml(data.token) + '</strong> (ID: ' + data.token_id + ')</div>';

        html += '<ul class="neighbors-list">';
        for (var i = 0; i < data.neighbors.length; i++) {
            var n = data.neighbors[i];
            html += '<li>' +
                '<span class="neighbor-token">' + escapeHtml(n.token) + '</span>' +
                '<span class="neighbor-score">cosine: ' + n.similarity.toFixed(4) + '</span>' +
            '</li>';
        }
        html += '</ul>';
        results.innerHTML = html;
    } catch (err) {
        showError('token-explorer-neighbors-results', err.message);
    }
}
