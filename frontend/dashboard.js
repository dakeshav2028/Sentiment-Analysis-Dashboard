// Global chart instances to destroy/recreate on data update
let trendChart = null;
let distChart = null;
let cachedKeywords = {};

document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const btnReanalyze = document.getElementById("btn-reanalyze");
    const textReview = document.getElementById("input-review");
    const btnPredict = document.getElementById("btn-predict");
    const predictResult = document.getElementById("predict-result");

    // Initial fetch to check status
    checkAppStatus();

    // Set up button event listeners
    btnReanalyze.addEventListener("click", () => triggerAnalysis(true));
    btnPredict.addEventListener("click", runPrediction);

    // Live review text prediction
    textReview.addEventListener("keypress", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            runPrediction();
        }
    });

    // Set up keyword tab listeners
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
        btn.addEventListener("click", (e) => {
            tabBtns.forEach(b => b.classList.remove("active"));
            e.target.classList.add("active");
            const sentiment = e.target.getAttribute("data-tab");
            renderKeywords(sentiment);
        });
    });
});

async function checkAppStatus() {
    try {
        const response = await fetch("/status");
        const statusData = await response.json();

        if (statusData.status === "pending_analysis") {
            // Do NOT auto-trigger batch analysis — too slow for live server.
            // Data should already be precomputed and shipped with the deployment.
            showToast("Dashboard data not precomputed. Contact site owner.", "error");
        } else {
            fetchDashboardData();
        }
    } catch (err) {
        console.error("Error checking app status:", err);
        showToast("Error connecting to backend API.", "error");
    }
}

async function triggerAnalysis(manual = false) {
    const overlay = document.getElementById("analysis-overlay");
    const spinner = document.getElementById("analyze-spinner");

    overlay.classList.remove("hidden");
    if (manual) {
        spinner.style.display = "inline-block";
    }

    try {
        const response = await fetch("/analyze", { method: "POST" });
        const result = await response.json();

        if (result.status === "success") {
            showToast("Analysis completed successfully!");
            fetchDashboardData();
        } else {
            showToast("Analysis failed: " + result.message, "error");
        }
    } catch (err) {
        console.error("Error running analysis:", err);
        showToast("Network error running batch analysis.", "error");
    } finally {
        overlay.classList.add("hidden");
        spinner.style.display = "none";
    }
}

async function fetchDashboardData() {
    try {
        const response = await fetch("/dashboard-data");
        const data = await response.json();

        if (data.status === "pending_analysis") {
            showToast("Dashboard data not precomputed yet.", "error");
            return;
        }

        // 1. Update stats elements
        document.getElementById("val-total-reviews").innerText = Number(data.summary.total_reviews).toLocaleString();

        const accuracyPct = (data.summary.accuracy * 100).toFixed(1);
        document.getElementById("val-accuracy").innerText = accuracyPct + "%";

        const posCount = data.summary.predicted_distribution.positive || 0;
        const neuCount = data.summary.predicted_distribution.neutral || 0;
        const negCount = data.summary.predicted_distribution.negative || 0;
        const total = data.summary.total_reviews || 3000;

        document.getElementById("val-pos-count").innerText = Number(posCount).toLocaleString();
        document.getElementById("val-pos-pct").innerText = ((posCount / total) * 100).toFixed(1) + "% of total";

        document.getElementById("val-neg-count").innerText = Number(negCount).toLocaleString();
        document.getElementById("val-neg-pct").innerText = ((negCount / total) * 100).toFixed(1) + "% of total";

        // Update model badge
        const modelBadge = document.getElementById("model-badge");
        modelBadge.innerText = data.model_name || "DistilBERT Classifier";
        if (data.model_name && data.model_name.includes("Fallback")) {
            modelBadge.style.borderColor = "rgba(239, 68, 68, 0.4)";
            modelBadge.style.color = "#fca5a5";
            modelBadge.style.background = "rgba(239, 68, 68, 0.1)";
        } else {
            modelBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
            modelBadge.style.color = "#a7f3d0";
            modelBadge.style.background = "rgba(16, 185, 129, 0.1)";
        }

        // 2. Render charts
        renderDistributionChart(posCount, neuCount, negCount);
        renderTrendChart(data.trends);

        // 3. Render Keywords
        cachedKeywords = data.keywords;
        // Find active tab and render
        const activeTab = document.querySelector(".tab-btn.active").getAttribute("data-tab");
        renderKeywords(activeTab);

    } catch (err) {
        console.error("Error loading dashboard data:", err);
        showToast("Error loading dashboard data.", "error");
    }
}

function renderDistributionChart(pos, neu, neg) {
    const ctx = document.getElementById("distributionChart").getContext("2d");

    if (distChart) {
        distChart.destroy();
    }

    distChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [pos, neu, neg],
                backgroundColor: ['#10b981', '#8b5cf6', '#ef4444'],
                borderWidth: 1,
                borderColor: 'rgba(255, 255, 255, 0.08)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 11 },
                        padding: 15
                    }
                }
            },
            cutout: '65%'
        }
    });
}

function renderTrendChart(trends) {
    const ctx = document.getElementById("trendChart").getContext("2d");

    if (trendChart) {
        trendChart.destroy();
    }

    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: trends.dates,
            datasets: [
                {
                    label: 'Positive',
                    data: trends.positive,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.03)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5
                },
                {
                    label: 'Neutral',
                    data: trends.neutral,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.03)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5
                },
                {
                    label: 'Negative',
                    data: trends.negative,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.03)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 2,
                    pointHoverRadius: 5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { family: 'Plus Jakarta Sans', size: 10 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { family: 'Plus Jakarta Sans', size: 10 } }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Plus Jakarta Sans', size: 11 }
                    }
                }
            }
        }
    });
}

function renderKeywords(sentiment) {
    const listElement = document.getElementById("keywords-list");
    listElement.innerHTML = "";

    const words = cachedKeywords[sentiment] || [];

    if (words.length === 0) {
        listElement.innerHTML = `<p style="grid-column: 1/-1; text-align: center; color: var(--text-dim); padding: 1rem;">No keywords extracted.</p>`;
        return;
    }

    words.forEach(item => {
        const wordEl = document.createElement("div");
        wordEl.className = "keyword-item";
        wordEl.innerHTML = `
            <span class="keyword-word">${item.word}</span>
            <span class="keyword-score">${item.score.toFixed(1)}</span>
        `;
        listElement.appendChild(wordEl);
    });
}

async function runPrediction() {
    const textarea = document.getElementById("input-review");
    const text = textarea.value.trim();
    const resultDiv = document.getElementById("predict-result");

    if (!text) {
        showToast("Please enter text to analyze.", "error");
        return;
    }

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });
        const result = await response.json();

        // Show result panel
        resultDiv.classList.remove("hidden");

        // Update badge
        const badge = document.getElementById("result-badge");
        badge.innerText = result.sentiment;
        badge.className = "badge"; // Reset classes
        badge.classList.add("badge-" + result.sentiment);

        // Update confidence
        const confidencePct = Math.round(result.confidence * 100);
        document.getElementById("result-confidence").innerText = confidencePct + "%";

        const barFill = document.getElementById("confidence-bar-fill");
        barFill.style.width = confidencePct + "%";
        barFill.className = "confidence-bar-fill"; // Reset classes
        barFill.classList.add("bg-" + result.sentiment);

    } catch (err) {
        console.error("Error predicting review:", err);
        showToast("Error classification endpoint.", "error");
    }
}

// Toast Helper
function showToast(message, type = "success") {
    // Simple modern toast popup
    const toast = document.createElement("div");
    toast.style.position = "fixed";
    toast.style.bottom = "20px";
    toast.style.right = "20px";
    toast.style.padding = "10px 20px";
    toast.style.borderRadius = "8px";
    toast.style.color = "white";
    toast.style.fontFamily = "var(--font-body)";
    toast.style.fontSize = "0.85rem";
    toast.style.fontWeight = "600";
    toast.style.zIndex = "2000";
    toast.style.boxShadow = "0 4px 12px rgba(0,0,0,0.5)";

    if (type === "success") {
        toast.style.background = "#10b981";
    } else {
        toast.style.background = "#ef4444";
    }

    toast.innerText = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transition = "opacity 0.5s ease";
        setTimeout(() => toast.remove(), 500);
    }, 3000);
}
