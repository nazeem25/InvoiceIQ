
"use strict";

const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const imagePreview = document.getElementById("imagePreview");
const uploadVisual = document.getElementById("uploadVisual");
const dropTitle = document.getElementById("dropTitle");
const dropSubtitle = document.getElementById("dropSubtitle");
const selectedFile = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFileButton = document.getElementById("removeFile");
const extractButton = document.getElementById("extractButton");
const progressWrap = document.getElementById("progressWrap");
const progressMessage = document.getElementById("progressMessage");

const emptyState = document.getElementById("emptyState");
const resultContent = document.getElementById("resultContent");
const resultBadge = document.getElementById("resultBadge");
const resultSubtitle = document.getElementById("resultSubtitle");
const fieldsBody = document.getElementById("fieldsBody");

const reviewAlert = document.getElementById("reviewAlert");
const reviewTitle = document.getElementById("reviewTitle");
const reviewMessage = document.getElementById("reviewMessage");

const validationBox = document.getElementById("validationBox");
const validationList = document.getElementById("validationList");
const recognizedText = document.getElementById("recognizedText");
const toast = document.getElementById("toast");

const exportCsvButton = document.getElementById("exportCsvButton");
const exportJsonButton = document.getElementById("exportJsonButton");

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_TYPES = ["image/png", "image/jpeg"];

const FIELD_LABELS = {
    invoice_number: "Invoice number",
    vendor: "Vendor",
    invoice_date: "Invoice date",
    item: "Line item",
    quantity: "Quantity",
    unit_price: "Unit price",
    subtotal: "Subtotal",
    tax: "Tax",
    grand_total: "Grand total",
};

let currentFile = null;
let previewUrl = null;
let latestResult = null;
let toastTimer = null;
let isProcessing = false;

// --------------------------------------------------
// Notifications and helpers
// --------------------------------------------------

function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");

    if (toastTimer !== null) {
        clearTimeout(toastTimer);
    }

    toastTimer = setTimeout(() => {
        toast.classList.remove("show");
        toastTimer = null;
    }, 4000);
}

function formatFileSize(bytes) {
    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setBadge(text, type = "neutral") {
    resultBadge.textContent = text;
    resultBadge.className = `result-badge ${type}`;
}

function setExportButtonsEnabled(enabled) {
    exportCsvButton.disabled = !enabled;
    exportJsonButton.disabled = !enabled;
}

function resetResults() {
    latestResult = null;

    emptyState.hidden = false;
    resultContent.hidden = true;
    reviewAlert.hidden = true;
    validationBox.hidden = true;

    setBadge("AWAITING INPUT");
    setExportButtonsEnabled(false);

    resultSubtitle.textContent =
        "Your structured invoice data will appear here.";
}

// --------------------------------------------------
// File selection and preview
// --------------------------------------------------

function selectFile(file) {
    if (!file) return;

    if (isProcessing) {
        showToast("Please wait for the current extraction to finish.");
        return;
    }

    if (!ALLOWED_TYPES.includes(file.type)) {
        showToast("Please choose a PNG, JPG, or JPEG image.");
        return;
    }

    if (file.size === 0) {
        showToast("The selected file is empty.");
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        showToast("The image exceeds the 10 MB upload limit.");
        return;
    }

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    currentFile = file;
    previewUrl = URL.createObjectURL(file);

    imagePreview.src = previewUrl;
    imagePreview.hidden = false;
    uploadVisual.hidden = true;

    dropTitle.textContent = "Invoice selected";
    dropSubtitle.textContent = "Ready to process this image";

    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    selectedFile.hidden = false;
    extractButton.disabled = false;

    resetResults();
}

function clearFile() {
    if (isProcessing) {
        showToast("Wait for extraction to finish before removing the file.");
        return;
    }

    currentFile = null;
    fileInput.value = "";

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    imagePreview.removeAttribute("src");
    imagePreview.hidden = true;
    uploadVisual.hidden = false;

    dropTitle.textContent = "Drop your invoice here";
    dropSubtitle.innerHTML =
        'or <span class="text-link">browse files</span> from your computer';

    selectedFile.hidden = true;
    extractButton.disabled = true;
    progressWrap.hidden = true;

    resetResults();
}

// --------------------------------------------------
// Display extraction results
// --------------------------------------------------

function renderResults(data) {
    const fields = data.extracted_fields || {};
    const fieldResults = data.field_results || {};

    const validationIssues = Array.isArray(data.validation_issues)
        ? data.validation_issues
        : [];

    const missingFields = Object.keys(FIELD_LABELS).filter((key) => {
        const value = fields[key];

        return value === null ||
            value === undefined ||
            String(value).trim() === "";
    });

    const flaggedFields = Object.values(fieldResults).filter(
        (field) => field && field.needs_review === true
    );

    const validationValid = data.validation_valid === true;

    const needsReview =
        data.needs_human_review === true ||
        missingFields.length > 0 ||
        flaggedFields.length > 0 ||
        !validationValid;

    // Retain the successful response for export.
    latestResult = data;
    setExportButtonsEnabled(true);

    emptyState.hidden = true;
    resultContent.hidden = false;

    resultSubtitle.textContent = `Results for ${currentFile.name}`;

    const confidence = Number(data.mean_ocr_confidence);

    document.getElementById("confidenceValue").textContent =
        data.mean_ocr_confidence !== null &&
        data.mean_ocr_confidence !== undefined &&
        Number.isFinite(confidence)
            ? `${(confidence * 100).toFixed(2)}%`
            : "N/A";

    const processingTime = Number(data.processing_time_seconds);

    document.getElementById("timeValue").textContent =
        data.processing_time_seconds !== null &&
        data.processing_time_seconds !== undefined &&
        Number.isFinite(processingTime)
            ? `${processingTime.toFixed(2)} s`
            : "N/A";

    document.getElementById("segmentsValue").textContent =
        data.recognized_segments ?? "N/A";

    const validationValue =
        document.getElementById("validationValue");

    validationValue.textContent = validationValid
        ? "Passed"
        : "Issues";

    validationValue.className = validationValid
        ? "valid"
        : "invalid";

    setBadge(
        needsReview ? "REVIEW RECOMMENDED" : "VALIDATION PASSED",
        needsReview ? "warning" : "success"
    );

    // Build the field table safely.
    fieldsBody.replaceChildren();

    for (const [key, label] of Object.entries(FIELD_LABELS)) {
        const value = fields[key];

        const missing =
            value === null ||
            value === undefined ||
            String(value).trim() === "";

        const details = fieldResults[key] || {};
        const percent = details.confidence_percent;
        const needsFieldReview = details.needs_review === true;

        const row = document.createElement("tr");
        const labelCell = document.createElement("td");
        const valueCell = document.createElement("td");
        const confidenceCell = document.createElement("td");
        const statusCell = document.createElement("td");

        labelCell.textContent = label;
        valueCell.textContent = missing
            ? "Not detected"
            : String(value);

        const confidenceNumber = Number(percent);

        if (
            percent !== null &&
            percent !== undefined &&
            Number.isFinite(confidenceNumber)
        ) {
            confidenceCell.textContent =
                `${confidenceNumber.toFixed(2)}%`;
        } else {
            confidenceCell.textContent = "Unavailable";
        }

        const status = document.createElement("span");

        status.className = `field-status ${
            missing || needsFieldReview
                ? "missing"
                : "present"
        }`;

        status.textContent = missing
            ? "Missing"
            : needsFieldReview
                ? "Review"
                : "Extracted";

        statusCell.appendChild(status);

        row.append(
            labelCell,
            valueCell,
            confidenceCell,
            statusCell
        );

        fieldsBody.appendChild(row);
    }

    // Display manual-review reasons.
    reviewAlert.hidden = !needsReview;

    if (needsReview) {
        reviewTitle.textContent = "Manual review recommended";

        const reasons = [];

        if (missingFields.length > 0) {
            reasons.push(`${missingFields.length} missing field(s)`);
        }

        if (flaggedFields.length > 0) {
            reasons.push(
                `${flaggedFields.length} field(s) flagged by OCR review`
            );
        }

        if (!validationValid) {
            reasons.push("validation checks failed");
        }

        reviewMessage.textContent =
            `${reasons.join("; ")}. Verify the values against the original invoice.`;
    }

    // Display validation issues.
    validationBox.hidden = validationIssues.length === 0;
    validationList.replaceChildren();

    for (const issue of validationIssues) {
        const item = document.createElement("li");
        item.textContent = String(issue).replaceAll("_", " ");
        validationList.appendChild(item);
    }

    recognizedText.textContent =
        data.recognized_text || "No OCR text returned.";

    document.getElementById("results").scrollIntoView({
        behavior: "smooth",
        block: "start",
    });
}

// --------------------------------------------------
// Call the Flask API
// --------------------------------------------------

async function extractInvoice() {
    if (!currentFile) {
        showToast("Choose an invoice image first.");
        return;
    }

    if (isProcessing) return;

    const fileBeingProcessed = currentFile;
    const formData = new FormData();

    formData.append("file", fileBeingProcessed);

    isProcessing = true;
    extractButton.disabled = true;
    progressWrap.hidden = false;

    progressMessage.textContent =
        "Running OCR, extracting fields, and checking confidence...";

    setBadge("PROCESSING");
    setExportButtonsEnabled(false);

    try {
        const response = await fetch("/api/extract", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || `Request failed (${response.status}).`
            );
        }

        if (data.ocr_status !== "success") {
            throw new Error(
                data.error || "OCR could not process this image."
            );
        }

        // Ignore results if the selected file changed.
        if (currentFile !== fileBeingProcessed) return;

        renderResults(data);
    } catch (error) {
        setBadge("PROCESSING FAILED", "error");
        showToast(
            error.message || "An unexpected error occurred."
        );
    } finally {
        isProcessing = false;
        progressWrap.hidden = true;
        extractButton.disabled = !currentFile;
    }
}

// --------------------------------------------------
// Download helpers
// --------------------------------------------------

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], {
        type: `${mimeType};charset=utf-8`,
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = filename;

    document.body.appendChild(link);
    link.click();
    link.remove();

    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function getExportFilename(extension) {
    const originalName = currentFile?.name || "invoice";
    const baseName = originalName.replace(/\.[^.]+$/, "");

    const safeName = baseName
        .replace(/[^a-zA-Z0-9_-]/g, "_")
        .slice(0, 100) || "invoice";

    return `${safeName}_extraction.${extension}`;
}

// --------------------------------------------------
// JSON export: complete API response
// --------------------------------------------------

function exportJson() {
    if (!latestResult) {
        showToast("Extract an invoice before exporting.");
        return;
    }

    downloadFile(
        JSON.stringify(latestResult, null, 2),
        getExportFilename("json"),
        "application/json"
    );

    showToast("JSON export downloaded.");
}

// --------------------------------------------------
// CSV export: fields, confidence, and validation
// --------------------------------------------------

function exportCsv() {
    if (!latestResult) {
        showToast("Extract an invoice before exporting.");
        return;
    }

    const fields = latestResult.extracted_fields || {};
    const fieldResults = latestResult.field_results || {};

    const rows = [
        [
            "Field",
            "Extracted Value",
            "OCR Confidence (%)",
            "Needs Review",
            "Review Reasons",
        ],
    ];

    for (const [key, label] of Object.entries(FIELD_LABELS)) {
        const value = fields[key];
        const result = fieldResults[key] || {};
        const confidence = result.confidence_percent;

        rows.push([
            label,
            value === null || value === undefined
                ? ""
                : String(value),
            confidence === null || confidence === undefined
                ? ""
                : String(confidence),
            result.needs_review ? "Yes" : "No",
            Array.isArray(result.review_reasons)
                ? result.review_reasons.join("; ")
                : "",
        ]);
    }

    rows.push([]);
    rows.push(["OCR Status", latestResult.ocr_status || ""]);
    rows.push([
        "Mean OCR Confidence",
        latestResult.mean_ocr_confidence ?? "",
    ]);
    rows.push([
        "Processing Time (seconds)",
        latestResult.processing_time_seconds ?? "",
    ]);
    rows.push([
        "Validation Passed",
        latestResult.validation_valid ? "Yes" : "No",
    ]);
    rows.push([
        "Validation Issues",
        Array.isArray(latestResult.validation_issues)
            ? latestResult.validation_issues.join("; ")
            : "",
    ]);
    rows.push([
        "Needs Human Review",
        latestResult.needs_human_review ? "Yes" : "No",
    ]);

    const csv = rows
        .map((row) =>
            row.map((cell) => {
                let value = String(cell ?? "");

                // Reduce spreadsheet formula-injection risk.
                if (/^[\s]*[=+\-@]/.test(value)) {
                    value = "'" + value;
                }

                return `"${value.replace(/"/g, '""')}"`;
            }).join(",")
        )
        .join("\r\n");

    downloadFile(
        "\uFEFF" + csv,
        getExportFilename("csv"),
        "text/csv"
    );

    showToast("CSV export downloaded.");
}

// --------------------------------------------------
// Event listeners
// --------------------------------------------------

dropzone.addEventListener("click", (event) => {
    if (event.target.closest("button")) return;
    fileInput.click();
});

dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
    }
});

fileInput.addEventListener("change", () => {
    selectFile(fileInput.files[0]);
});

removeFileButton.addEventListener("click", clearFile);
extractButton.addEventListener("click", extractInvoice);

exportCsvButton.addEventListener("click", exportCsv);
exportJsonButton.addEventListener("click", exportJson);

["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("drag-over");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("drag-over");
    });
});

dropzone.addEventListener("drop", (event) => {
    selectFile(event.dataTransfer.files[0]);
});

// Initial dashboard state.
resetResults();
