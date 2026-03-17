// ==========================
// TRAIT SUMMARY MAP (NEW)
// ==========================
const TRAIT_SUMMARY_MAP = {
    "Agreeableness": "Cooperative, kind, empathetic nature.",
    "Conscientiousness": "Organized, responsible, disciplined.",
    "Extraversion": "Energetic, talkative, socially expressive.",
    "Neuroticism": "Emotionally sensitive, easily stressed.",
    "Openness": "Creative, imaginative, open to new ideas."
};

// ==========================
// ERROR POPUP
// ==========================
function showError(message) {
    const old = document.getElementById("error-popup");
    if (old) old.remove();

    const popup = document.createElement("div");
    popup.id = "error-popup";
    popup.className = "error-popup";
    popup.innerHTML = `<span>${message}</span>`;
    document.body.appendChild(popup);

    setTimeout(() => popup.classList.add("show"), 20);
    setTimeout(() => popup.remove(), 3000);
}

const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");

let selectedFile = null;
let chartInstance = null;

fileInput.addEventListener("change", () => {
    selectedFile = fileInput.files[0];
    if (selectedFile) {
        preview.src = URL.createObjectURL(selectedFile);
        preview.classList.remove("d-none");
        analyzeBtn.classList.remove("d-none");
    }
});

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return showError("Please upload an image!");

    const formData = new FormData();
    formData.append("file", selectedFile);

    document.querySelector(".upload-hero").insertAdjacentHTML(
        "beforeend",
        `<div id="loader" class="mt-3 text-center">
            <div class="spinner-border text-primary"></div>
            <p class="text-muted mt-2">Analyzing...</p>
        </div>`
    );

    try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        document.getElementById("loader").remove();

        if (data.error) return showError(data.error);

        showResult(data.trait, data.summary, data.scores);

    } catch (err) {
        showError("Backend not responding!");
        document.getElementById("loader").remove();
    }
});

// ==========================
// SHOW RESULT + CLICKABLE CHART
// ==========================
function showResult(trait, summary, scores) {
    const old = document.getElementById("result-card");
    if (old) old.remove();

    const html = `
        <div id="result-card" class="card shadow p-4 mt-4">
            <h3 class="text-primary fw-bold">Predicted Trait: ${trait}</h3>
            <p>${summary}</p>

            <canvas id="traitChart" class="mt-4"></canvas>

            <!-- CLICKED TRAIT SUMMARY -->
            <div id="traitDetailBox" class="alert alert-info mt-3 d-none"></div>

            <button id="downloadPdfBtn" class="btn btn-success w-100 mt-4">
                Download PDF Report
            </button>
        </div>
    `;

    document.querySelector(".upload-hero").insertAdjacentHTML("beforeend", html);

    const ctx = document.getElementById("traitChart");
    if (chartInstance) chartInstance.destroy();

    const labels = Object.keys(scores);
    const values = Object.values(scores);

    chartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Trait Strength (%)",
                data: values,
                backgroundColor: "#6c63ff",
                borderWidth: 2
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true, max: 100 }
            },

            // 🔥 BAR CLICK EVENT
            onClick: (evt, elements) => {
                if (!elements.length) return;

                const index = elements[0].index;
                const traitName = labels[index];
                const score = values[index];

                const box = document.getElementById("traitDetailBox");
                box.classList.remove("d-none");
                box.innerHTML = `
                    <strong>${traitName}</strong><br>
                    Score: <b>${score}%</b><br>
                    ${TRAIT_SUMMARY_MAP[traitName]}
                `;
            }
        }
    });

    document.getElementById("downloadPdfBtn").onclick = () =>
        downloadPDF(trait, summary);
}

// ==========================
// PDF DOWNLOAD FUNCTION
// ==========================
function downloadPDF(trait, summary) {
    const pdf = new jsPDF("p", "mm", "a4");

    pdf.setFontSize(20);
    pdf.text("HandwritingAI Personality Report", 20, 20);

    pdf.setFontSize(14);
    pdf.text(`Predicted Trait: ${trait}`, 20, 40);

    pdf.setFontSize(12);
    pdf.text(summary, 20, 55, { maxWidth: 170 });

    const canvas = document.getElementById("traitChart");
    const chartData = canvas.toDataURL("image/png", 1.0);

    pdf.addImage(chartData, "PNG", 20, 80, 170, 90);

    pdf.save("HandwritingAI_Report.pdf");
}
