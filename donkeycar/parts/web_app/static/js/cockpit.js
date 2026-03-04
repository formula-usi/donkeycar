
// ============================================================
// Cockpit Display
// Read-only view: receives state from the server via WebSocket
// and updates the gauge, steering wheel, and circuit info.
// ============================================================

const state = {
    circuit: 'Default',
    circuit_icon: '/static/images/default_circuit.png',
    surface: 'dry',
    surface_icon: '/static/images/weather/dry.svg',
    tele: {
        user:  { angle: 0, throttle: 0 },
        pilot: { angle: 0, throttle: 0 }
    }
};

// Seed values from server-rendered state if available.
if (globalThis.serverState) {
    if (globalThis.serverState.circuit !== undefined) state.circuit = globalThis.serverState.circuit;
    if (globalThis.serverState.current_surface !== undefined) state.surface = globalThis.serverState.current_surface;
    if (globalThis.serverState.surface_icon !== undefined) state.surface_icon = globalThis.serverState.surface_icon;
}

// ============================================================
// State & UI
// ============================================================

// Update fields that already exist in the state object.
// Also maps flat server keys (angle, throttle) into state.tele.user.
function updateState(state, data) {
    let changed = false;
    if (typeof data === 'object') {
        Object.keys(data).forEach(function(key) {
            if (state.hasOwnProperty(key) && state[key] !== data[key]) {
                if (typeof state[key] === 'object') {
                    changed = updateState(state[key], data[key]) || changed;
                } else {
                    state[key] = data[key];
                    changed = true;
                }
            }
            if (state.tele?.user.hasOwnProperty(key) && state.tele.user[key] !== data[key]) {
                if (typeof state.tele.user[key] === 'object') {
                    changed = updateState(state.tele.user[key], data[key]) && changed;
                } else {
                    state.tele.user[key] = data[key];
                    changed = true;
                }
            }
        });
    }
    return changed;
}

function updateUI() {
    // Circuit name
    document.getElementById('circuit_display').textContent = state.circuit;

    // Circuit minimap
    const circuitImage = document.getElementById('circuit_image');
    if (state.circuit_icon) {
        circuitImage.src = state.circuit_icon;
        circuitImage.alt = state.circuit + ' Circuit';
    }

    // Weather / surface display
    const weatherDisplay = document.getElementById('weather_display');
    console.log('Updating surface display:', state.surface, state.surface_icon);
    if (weatherDisplay) {
        const img = document.getElementById('surface_icon');
        if (img) img.src = state.surface_icon;
        const surfaceText = document.getElementById('surface_text');
        if (surfaceText) surfaceText.textContent = state.surface;
    }

    // Steering wheel rotation (angle is -1..1)
    const rotationDegrees = state.tele.user.angle * 180 / Math.PI;
    document.getElementById('steering_wheel').style.transform = 'rotate(' + rotationDegrees + 'deg)';

    // Speedometer — show whichever throttle (user or pilot) is larger
    const gauge = document.getElementById('speedometer_gauge');
    if (gauge) {
        const throttle = Math.abs(state.tele.user.throttle) >= Math.abs(state.tele.pilot.throttle)
            ? state.tele.user.throttle
            : state.tele.pilot.throttle;
        gauge.setAttribute('value', (Math.abs(throttle) * 100).toFixed(0));
    }
}

// ============================================================
// WebSocket
// ============================================================

document.addEventListener('DOMContentLoaded', function() {
    const socket = new WebSocket('ws://' + location.host + '/wsDrive');
    globalThis.donkeySocket = socket;

    socket.onopen  = function()  { console.log('Cockpit WebSocket connected'); };
    socket.onerror = function(e) { console.error('Cockpit WebSocket error:', e); };
    socket.onclose = function()  { console.log('Cockpit WebSocket closed'); };

    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (updateState(state, data)) {
            updateUI();
        }
    };

    updateUI();
});

// ============================================================
// Analog Gauge Web Component (Speedometer)
// Styles are loaded into the shadow root from cockpit.css.
// ============================================================

class AnalogGauge extends HTMLElement {
    static get observedAttributes() { return ['value']; }

    #root; #units; #value;

    constructor() {
        super();
        this.#root = this.attachShadow({ mode: 'open' });

        const computedStyle = getComputedStyle(this);
        this.#units = {
            defaultMark:   90,
            defaultNeedle: 270,
            max:    Number.parseInt(this.getAttribute('max') || 100),
            min:    Number.parseInt(this.getAttribute('min') || 0),
            range:  Number.parseFloat(computedStyle.getPropertyValue('--analog-gauge-range')) || 250,
            suffix: this.getAttribute('suffix') || '',
            start:  Number.parseFloat(computedStyle.getPropertyValue('--analog-gauge-start-angle')) || 235,
            value:  Number.parseFloat(this.getAttribute('value') || 0)
        };
        this.#units.minDegree  = this.#units.start - this.#units.defaultNeedle;
        this.#units.totalRange = this.#units.range;

        this.#root.innerHTML = `
            <link rel="stylesheet" href="/static/styles/cockpit.css">
            <div part="bezel"></div>
            <div part="gauge"></div>
            <div part="ticks"></div>
            ${this.#generateValueMarks()}
            <div part="needle"></div>
            <div part="hub"></div>
            <div part="value"></div>
            <div part="label">${this.getAttribute('label') || ''}</div>
            <div part="label-min">${this.getAttribute('min-label') || ''}</div>
            <div part="label-max">${this.getAttribute('max-label') || ''}</div>
            <div part="glass"></div>
        `;

        this.#value = this.#root.querySelector('[part="value"]');
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (name === 'value' && oldValue !== newValue) {
            this.#units.value = Number.parseFloat(newValue || 0);
            this.#update();
        }
    }

    #generateValueMarks() {
        const values = this.getAttribute('values');
        if (!values) return '';

        let valueArray = [];
        let count = 0;

        if (/^\s*\d+\s*$/.test(values)) {
            count = Number.parseInt(values.trim());
            if (Number.isNaN(count) || count <= 0) return '';
            valueArray = Array.from({ length: count }, (_, i) =>
                Math.round(this.#units.min + (i * (this.#units.max - this.#units.min) / (count - 1 || 1)))
            );
        } else {
            valueArray = values.split(',').map(v => v.trim());
            count = valueArray.length;
            if (count <= 0) return '';
        }

        const degreeStep = this.#units.range / (count - 1 || 1);
        return `
        <ul part="value-marks">
            ${valueArray.map((value, i) => {
                const degree = this.#units.start - this.#units.defaultMark + (i * degreeStep);
                return `<li style="--_d:${degree}deg" part="value-mark">${value}</li>`;
            }).join('')}
        </ul>`;
    }

    #update() {
        const normalizedValue = Math.max(this.#units.min, Math.min(this.#units.max, this.#units.value));
        const valuePercentage = (normalizedValue - this.#units.min) / (this.#units.max - this.#units.min);
        const degree          = this.#units.minDegree + (valuePercentage * this.#units.totalRange);
        this.style.setProperty('--analog-gauge-value', `${valuePercentage * this.#units.range}deg`);
        this.style.setProperty('--_d', `${degree}deg`);
        this.#value.textContent = this.#units.value + this.#units.suffix;
    }
}

customElements.define('analog-gauge', AnalogGauge);
