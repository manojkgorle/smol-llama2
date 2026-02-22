/**
 * app.js — Tab router, WebSocket init, and shared infrastructure for LLaMA-2 dashboard.
 */

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
window.vizState = {
    socket: null,
    model_info: null,
    activeTab: 'dashboard',
    tabInitialized: {
        dashboard: false,
        attention: false,
        activations: false,
        attribution: false,
        circuits: false,
        captum: false,
    },
};

// LLaMA model constants (updated after model info loads)
window.N_LAYERS = 8;
window.N_HEADS = 6;
window.N_KV_HEADS = 2;
window.N_KV_GROUPS = 3;

// GQA group colors: warm for group 0, cool for group 1
window.GQA_COLORS = {
    group0: ['#ff6b6b', '#ff8787', '#ffa3a3'],  // heads 0,1,2
    group1: ['#4ecdc4', '#72ddd6', '#96ede8'],   // heads 3,4,5
};

// 8-layer color palette
window.LAYER_COLORS = [
    '#e94560', '#f0a500', '#4ecca3', '#3282b8',
    '#bb86fc', '#ff6b6b', '#4ecdc4', '#f7d794',
];

// ---------------------------------------------------------------------------
// Shared Plotly layout / config
// ---------------------------------------------------------------------------
window.PLOTLY_DARK_LAYOUT = {
    paper_bgcolor: '#1a1a2e',
    plot_bgcolor: '#16213e',
    font: { color: '#e0e0e0', family: '-apple-system, BlinkMacSystemFont, sans-serif' },
    margin: { t: 36, r: 24, b: 48, l: 56 },
    xaxis: { gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
    yaxis: { gridcolor: '#2a2a4a', zerolinecolor: '#2a2a4a' },
    legend: { bgcolor: 'rgba(22,33,62,0.8)', bordercolor: '#2a2a4a', borderwidth: 1, font: { size: 11 } },
};

window.PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

// ---------------------------------------------------------------------------
// Tab routing
// ---------------------------------------------------------------------------
function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(function (el) { el.classList.remove('active'); });
    document.querySelectorAll('.nav-btn').forEach(function (btn) { btn.classList.remove('active'); });

    var section = document.getElementById('tab-' + tabName);
    if (section) section.classList.add('active');

    var btn = document.querySelector('.nav-btn[data-tab="' + tabName + '"]');
    if (btn) btn.classList.add('active');

    window.vizState.activeTab = tabName;

    if (!window.vizState.tabInitialized[tabName]) {
        window.vizState.tabInitialized[tabName] = true;
        switch (tabName) {
            case 'dashboard': if (typeof initDashboard === 'function') initDashboard(); break;
            case 'attention': if (typeof initAttention === 'function') initAttention(); break;
            case 'activations': if (typeof initActivations === 'function') initActivations(); break;
            case 'attribution': if (typeof initAttribution === 'function') initAttribution(); break;
            case 'circuits': if (typeof initCircuits === 'function') initCircuits(); break;
            case 'captum': if (typeof initCaptum === 'function') initCaptum(); break;
        }
    }

    setTimeout(function () {
        if (section) {
            section.querySelectorAll('.js-plotly-plot').forEach(function (plot) {
                Plotly.Plots.resize(plot);
            });
        }
    }, 50);
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
function initWebSocket() {
    try {
        var socket = io({ transports: ['polling'], upgrade: false });
        socket.on('connect', function () { console.log('[ws] connected'); });
        socket.on('disconnect', function () { console.log('[ws] disconnected'); });
        socket.on('connect_error', function (err) { console.warn('[ws] error:', err.message); });
        socket.on('step_update', function (data) {
            if (typeof handleStepUpdate === 'function') handleStepUpdate(data);
        });
        socket.on('val_update', function (data) {
            if (typeof handleValUpdate === 'function') handleValUpdate(data);
        });
        window.vizState.socket = socket;
    } catch (err) {
        console.warn('[ws] failed:', err.message);
    }
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------
function showLoading(containerId) {
    var c = document.getElementById(containerId);
    if (!c || c.querySelector('.loading-overlay')) return;
    c.style.position = 'relative';
    var ov = document.createElement('div');
    ov.className = 'loading-overlay';
    ov.innerHTML = '<div class="spinner"></div>';
    c.appendChild(ov);
}

function hideLoading(containerId) {
    var c = document.getElementById(containerId);
    if (!c) return;
    var ov = c.querySelector('.loading-overlay');
    if (ov) ov.remove();
}

function showInlineLoading(containerId) {
    var c = document.getElementById(containerId);
    if (!c) return;
    c.innerHTML = '<div class="loading-inline"><div class="spinner"></div><span>Loading...</span></div>';
}

// ---------------------------------------------------------------------------
// Error display
// ---------------------------------------------------------------------------
function showError(containerId, message) {
    var c = document.getElementById(containerId);
    if (!c) return;
    c.innerHTML = '<div class="error-message">' + escapeHtml(message) + '</div>';
}

function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Reusable prompt input component
// ---------------------------------------------------------------------------
function createPromptInput(containerId, onSubmit, options) {
    var opts = options || {};
    var placeholder = opts.placeholder || 'Enter a prompt...';
    var btnLabel = opts.buttonLabel || 'Analyze';

    var container = document.getElementById(containerId);
    if (!container) return null;

    var group = document.createElement('div');
    group.className = 'prompt-group';

    var input = document.createElement('input');
    input.type = 'text';
    input.className = 'prompt-input';
    input.placeholder = placeholder;
    input.id = containerId + '-prompt-input';

    var btn = document.createElement('button');
    btn.className = 'btn btn-primary';
    btn.textContent = btnLabel;
    btn.id = containerId + '-prompt-btn';

    group.appendChild(input);
    group.appendChild(btn);
    container.appendChild(group);

    function submit() {
        var value = input.value.trim();
        if (!value) return;
        onSubmit(value);
    }

    btn.addEventListener('click', submit);
    input.addEventListener('keydown', function (e) { if (e.key === 'Enter') submit(); });

    return { input: input, button: btn, group: group };
}

// ---------------------------------------------------------------------------
// Dropdown helper
// ---------------------------------------------------------------------------
function createDropdown(containerId, label, id, optionsList) {
    var container = document.getElementById(containerId);
    if (!container) return null;

    var group = document.createElement('div');
    group.className = 'select-group';

    var lbl = document.createElement('label');
    lbl.setAttribute('for', id);
    lbl.textContent = label;

    var sel = document.createElement('select');
    sel.className = 'select-input';
    sel.id = id;

    optionsList.forEach(function (opt) {
        var o = document.createElement('option');
        o.value = opt.value;
        o.textContent = opt.label;
        sel.appendChild(o);
    });

    group.appendChild(lbl);
    group.appendChild(sel);
    container.appendChild(group);
    return sel;
}

// ---------------------------------------------------------------------------
// Fetch helper
// ---------------------------------------------------------------------------
async function apiFetch(url, options) {
    var resp = await fetch(url, options);
    if (!resp.ok) {
        var errText = await resp.text();
        throw new Error('API error ' + resp.status + ': ' + errText);
    }
    return resp.json();
}

function darkLayout(overrides) {
    var base = JSON.parse(JSON.stringify(window.PLOTLY_DARK_LAYOUT));
    return Object.assign(base, overrides || {});
}

// ---------------------------------------------------------------------------
// GQA helper: get group index for a head
// ---------------------------------------------------------------------------
function getGQAGroup(headIdx) {
    return Math.floor(headIdx / window.N_KV_GROUPS);
}

function getGQAColor(headIdx) {
    var group = getGQAGroup(headIdx);
    var inGroupIdx = headIdx % window.N_KV_GROUPS;
    return group === 0 ? window.GQA_COLORS.group0[inGroupIdx] : window.GQA_COLORS.group1[inGroupIdx];
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.nav-btn').forEach(function (btn) {
        btn.addEventListener('click', function () { switchTab(btn.dataset.tab); });
    });

    initWebSocket();

    apiFetch('/api/model/info')
        .then(function (data) {
            window.vizState.model_info = data;
            if (data.loaded) {
                window.N_LAYERS = data.n_layers;
                window.N_HEADS = data.n_heads;
                window.N_KV_HEADS = data.n_kv_heads;
                window.N_KV_GROUPS = data.n_heads / data.n_kv_heads;
            }
            console.log('[app] model info:', data);
        })
        .catch(function (err) {
            console.warn('[app] model info fetch failed:', err.message);
        });

    switchTab('dashboard');
});
