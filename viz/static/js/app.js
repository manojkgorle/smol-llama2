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
    group0: ['#f87171', '#fca5a5', '#fecaca'],  // heads 0,1,2
    group1: ['#22d3ee', '#67e8f9', '#a5f3fc'],   // heads 3,4,5
};

// 8-layer color palette
window.LAYER_COLORS = [
    '#f87171', '#fb923c', '#fbbf24', '#34d399',
    '#22d3ee', '#818cf8', '#a78bfa', '#f472b6',
];

// ---------------------------------------------------------------------------
// Shared Plotly layout / config
// ---------------------------------------------------------------------------
window.PLOTLY_DARK_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(255,255,255,0.012)',
    font: { color: '#a1a1aa', family: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', size: 12 },
    margin: { t: 36, r: 24, b: 48, l: 56 },
    xaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.08)' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.08)' },
    legend: { bgcolor: 'rgba(9,9,11,0.9)', bordercolor: 'rgba(255,255,255,0.07)', borderwidth: 1, font: { size: 11, color: '#a1a1aa' } },
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
    if (!overrides) return base;
    // Deep merge one level so axis/legend overrides inherit base gridcolor etc.
    Object.keys(overrides).forEach(function (key) {
        if (overrides[key] && typeof overrides[key] === 'object' && !Array.isArray(overrides[key]) &&
            base[key] && typeof base[key] === 'object' && !Array.isArray(base[key])) {
            Object.assign(base[key], overrides[key]);
        } else {
            base[key] = overrides[key];
        }
    });
    return base;
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
