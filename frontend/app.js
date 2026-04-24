const healthStatus = document.querySelector("#health-status");
const healthVersion = document.querySelector("#health-version");
const lastSync = document.querySelector("#last-sync");
const predictionResult = document.querySelector("#prediction-result");
const driftStatus = document.querySelector("#drift-status");
const driftError = document.querySelector("#drift-error");
const retrainResult = document.querySelector("#retrain-result");
const riskScore = document.querySelector("#risk-score");
const riskCaption = document.querySelector("#risk-caption");
const gaugeRing = document.querySelector("#gauge-ring");
const historyList = document.querySelector("#history-list");
const summaryRisk = document.querySelector("#summary-risk");
const summaryDrift = document.querySelector("#summary-drift");
const summaryFeatures = document.querySelector("#summary-features");
const summaryRetrain = document.querySelector("#summary-retrain");

const presets = {
    stable: {
        temperature_c: 68,
        vibration_mm_s: 5.6,
        pressure_bar: 92,
        runtime_hours: 2800,
    },
    watch: {
        temperature_c: 94,
        vibration_mm_s: 11.2,
        pressure_bar: 119,
        runtime_hours: 6400,
    },
    critical: {
        temperature_c: 128,
        vibration_mm_s: 22.5,
        pressure_bar: 158,
        runtime_hours: 10100,
    },
};

const recentChecks = [];

function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
}

function timestampLabel() {
    return new Date().toLocaleTimeString();
}

function setResultState(element, state, html) {
    element.classList.remove("muted", "ok", "alert");
    element.classList.add(state);
    element.innerHTML = html;
}

function setGauge(probability, maintenanceRequired) {
    const percent = Math.round(probability * 100);
    const angle = `${Math.max(8, probability * 360)}deg`;
    gaugeRing.style.setProperty("--gauge-value", angle);
    riskScore.textContent = `${percent}%`;
    riskCaption.textContent = maintenanceRequired ? "Action window open" : "Within safe zone";
    summaryRisk.textContent = maintenanceRequired
        ? `High risk at ${percent}%`
        : `Stable at ${percent}%`;
}

function renderHistory() {
    if (!recentChecks.length) {
        historyList.innerHTML = `<div class="history-empty">Recent predictions will appear here.</div>`;
        return;
    }

    historyList.innerHTML = recentChecks
        .map((entry) => `
            <div class="history-item">
                <strong>${entry.title}</strong>
                <div class="history-meta">
                    ${entry.time}<br>
                    Temp ${entry.payload.temperature_c}C | Vib ${entry.payload.vibration_mm_s} mm/s<br>
                    Pressure ${entry.payload.pressure_bar} bar | Runtime ${entry.payload.runtime_hours} h
                </div>
            </div>
        `)
        .join("");
}

function addHistoryEntry(payload, probability, maintenanceRequired) {
    recentChecks.unshift({
        title: `${maintenanceRequired ? "Maintenance recommended" : "Stable reading"} · ${formatPercent(probability)}`,
        time: timestampLabel(),
        payload,
    });

    if (recentChecks.length > 4) {
        recentChecks.pop();
    }

    renderHistory();
}

function applyPreset(name) {
    const preset = presets[name];
    if (!preset) {
        return;
    }

    for (const [key, value] of Object.entries(preset)) {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) {
            input.value = value;
        }
    }
}

function randomBetween(min, max, digits = 1) {
    return Number((Math.random() * (max - min) + min).toFixed(digits));
}

function randomizeSensors() {
    const samples = [
        {
            temperature_c: randomBetween(60, 82),
            vibration_mm_s: randomBetween(4, 8),
            pressure_bar: randomBetween(84, 100),
            runtime_hours: randomBetween(1800, 4200, 0),
        },
        {
            temperature_c: randomBetween(84, 105),
            vibration_mm_s: randomBetween(8, 14),
            pressure_bar: randomBetween(104, 130),
            runtime_hours: randomBetween(4200, 7600, 0),
        },
        {
            temperature_c: randomBetween(105, 135),
            vibration_mm_s: randomBetween(14, 26),
            pressure_bar: randomBetween(130, 168),
            runtime_hours: randomBetween(7600, 11200, 0),
        },
    ];

    const sample = samples[Math.floor(Math.random() * samples.length)];
    for (const [key, value] of Object.entries(sample)) {
        document.querySelector(`[name="${key}"]`).value = value;
    }
}

async function loadHealth() {
    const response = await fetch("/health");
    const data = await response.json();
    healthStatus.textContent = data.status.toUpperCase();
    healthVersion.textContent = data.model_version;
    lastSync.textContent = timestampLabel();
}

async function loadDrift() {
    driftError.textContent = "";

    try {
        const response = await fetch("/drift/status");
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to load drift status.");
        }

        summaryDrift.textContent = data.drift_detected ? "Detected" : "Stable";
        summaryFeatures.textContent = data.drifted_features.length
            ? data.drifted_features.join(", ")
            : "None";

        driftStatus.innerHTML = `
            <div class="metric-row"><span>Dataset drift</span><strong>${data.drift_detected ? "Detected" : "Stable"}</strong></div>
            <div class="metric-row"><span>Drifted share</span><strong>${formatPercent(data.share_drifted_features)}</strong></div>
            <div class="metric-row"><span>Reference rows</span><strong>${data.n_reference}</strong></div>
            <div class="metric-row"><span>Current rows</span><strong>${data.n_current}</strong></div>
            <div class="metric-row"><span>Features in drift</span><strong>${data.drifted_features.length ? data.drifted_features.join(", ") : "None"}</strong></div>
        `;
    } catch (error) {
        summaryDrift.textContent = "Unavailable";
        summaryFeatures.textContent = "--";
        driftStatus.innerHTML = `<div class="metric-row"><span>State</span><strong>Unavailable</strong></div>`;
        driftError.textContent = error.message;
    }
}

async function submitPrediction(event) {
    event.preventDefault();

    const formData = new FormData(event.currentTarget);
    const payload = Object.fromEntries(formData.entries());

    for (const key of Object.keys(payload)) {
        payload[key] = Number(payload[key]);
    }

    setResultState(predictionResult, "muted", "Running maintenance check...");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        const state = data.maintenance_required ? "alert" : "ok";
        const title = data.maintenance_required ? "Maintenance recommended" : "Machine looks stable";

        setGauge(data.probability, data.maintenance_required);
        addHistoryEntry(payload, data.probability, data.maintenance_required);

        setResultState(
            predictionResult,
            state,
            `<strong>${title}</strong>
            Risk probability: ${formatPercent(data.probability)}<br>
            Model version: ${data.model_version}<br>
            Next step: ${data.maintenance_required ? "Schedule inspection and prepare intervention." : "Keep monitoring current operating conditions."}`
        );

        await Promise.all([loadHealth(), loadDrift()]);
    } catch (error) {
        setResultState(predictionResult, "alert", `<strong>Prediction error</strong>${error.message}`);
    }
}

async function triggerRetrain() {
    setResultState(retrainResult, "muted", "Retraining in progress...");
    summaryRetrain.textContent = "Running";

    try {
        const response = await fetch("/retrain", { method: "POST" });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Retrain failed.");
        }

        const message = data.status === "already_running"
            ? `<strong>Retrain already running</strong>Another training job is still active.`
            : `<strong>Retrain complete</strong>New model version: ${data.model_version}`;

        setResultState(retrainResult, data.status === "ok" ? "ok" : "muted", message);
        summaryRetrain.textContent = data.status === "ok" ? `Updated to ${data.model_version}` : "Already running";
        await loadHealth();
    } catch (error) {
        summaryRetrain.textContent = "Error";
        setResultState(retrainResult, "alert", `<strong>Retrain error</strong>${error.message}`);
    }
}

async function refreshAll() {
    await Promise.all([loadHealth(), loadDrift()]);
}

document.querySelector("#predict-form").addEventListener("submit", submitPrediction);
document.querySelector("#refresh-health").addEventListener("click", loadHealth);
document.querySelector("#refresh-drift").addEventListener("click", loadDrift);
document.querySelector("#retrain-button").addEventListener("click", triggerRetrain);
document.querySelector("#refresh-all").addEventListener("click", refreshAll);
document.querySelector("#randomize-sensors").addEventListener("click", randomizeSensors);
document.querySelector("#load-watchlist").addEventListener("click", () => applyPreset("watch"));

document.querySelectorAll("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => applyPreset(button.dataset.preset));
});

Promise.all([loadHealth(), loadDrift()]).catch((error) => {
    healthStatus.textContent = "ERROR";
    healthVersion.textContent = "--";
    driftError.textContent = error.message;
});
