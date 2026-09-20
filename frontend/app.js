/**
 * AI Image Restoration — Frontend Logic
 * Phase 5: Complete Upload, Analyze, Pipeline Routing, & Restoration
 */

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');

let selectedFile = null;

// --- Drag & Drop ---
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// --- File Selection ---
function handleFileSelect(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        alert('Unsupported file type. Please use JPG, PNG, WebP, or BMP.');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        alert('File too large. Maximum size is 10 MB.');
        return;
    }

    selectedFile = file;
    dropZone.innerHTML = `<p>✅ <strong>${file.name}</strong><br>(${(file.size / 1024 / 1024).toFixed(2)} MB)</p>`;
    dropZone.classList.add('has-file');
    uploadBtn.disabled = false;
}

// --- Upload, Analyze, & Restore Flow ---
uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading & Validating...';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        // 1. Upload Image
        const uploadResp = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        const uploadData = await uploadResp.json();

        if (!uploadResp.ok || !uploadData.success) {
            alert(`Upload failed: ${uploadData.detail || uploadData.error || 'Unknown error'}`);
            return;
        }

        const imageId = uploadData.image_id;

        // 2. Analyze Image Degradations
        uploadBtn.textContent = 'Running Degradation Analyzer...';
        const analyzeResp = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: imageId }),
        });

        const analyzeData = await analyzeResp.json();
        renderAnalysisReport(uploadData, analyzeData);

        // 3. Execute Restoration Pipeline
        const isUpscale4k = document.getElementById('upscale-4k-toggle')?.checked || false;
        uploadBtn.textContent = 'Running Pipeline Restoration...';
        const restoreResp = await fetch('/api/restore', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_id: imageId,
                upscale_4k: isUpscale4k,
            }),
        });

        const restoreData = await restoreResp.json();

        if (restoreResp.ok && restoreData.success) {
            renderPipelineSteps(restoreData.pipeline_steps, restoreData.inference_time_seconds);
            renderResultComparison(imageId, restoreData);
        } else {
            alert(`Restoration error: ${restoreData.detail || 'Failed to restore image'}`);
        }

    } catch (err) {
        console.error('Error during upload, analysis, or restoration:', err);
        alert('Error connecting to backend API: ' + err.message);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Restore Image';
    }
});

// --- Render Functions ---
function renderAnalysisReport(uploadData, analyzeData) {
    const analysisSection = document.getElementById('analysis-section');
    const reportDiv = document.getElementById('degradation-report');

    analysisSection.classList.remove('hidden');

    const meta = uploadData.meta;
    const degradations = analyzeData.degradations || [];
    const metrics = analyzeData.raw_metrics || {};

    let degradationsHtml = '';
    if (degradations.length === 0) {
        degradationsHtml = `<p style="color: var(--success); font-weight: 600;">✨ No major degradations detected (Image appears clean)</p>`;
    } else {
        degradationsHtml = '<div class="degradation-list">';
        degradations.forEach(deg => {
            const sevClass = deg.severity.toLowerCase();
            degradationsHtml += `
                <div class="deg-card deg-${sevClass}">
                    <span class="deg-name">${deg.name.replace(/_/g, ' ').toUpperCase()}</span>
                    <span class="deg-badge badge-${sevClass}">${deg.severity} (${(deg.score * 100).toFixed(0)}%)</span>
                    ${deg.confidence ? `<span style="display:block; font-size: 0.8rem; margin-top:2px; opacity: 0.8;">Confidence: ${(deg.confidence * 100).toFixed(0)}%</span>` : ''}
                </div>
            `;
        });
        degradationsHtml += '</div>';
    }

    let metricsHtml = '<div class="metrics-grid">';
    // Always show dimensions first
    metricsHtml += `
        <div class="metric-box">
            <span class="metric-label">Dimensions</span>
            <span class="metric-val">${meta.width} x ${meta.height} px</span>
        </div>
    `;

    // Render all raw metrics dynamically directly from backend payload
    for (const [key, val] of Object.entries(metrics)) {
        if (key === "dimensions") continue;
        const formattedKey = key.replace(/_/g, ' ').toUpperCase();
        const formattedVal = (typeof val === 'number') ? val.toFixed(3) : val;
        metricsHtml += `
            <div class="metric-box">
                <span class="metric-label">${formattedKey}</span>
                <span class="metric-val">${formattedVal}</span>
            </div>
        `;
    }
    metricsHtml += '</div>';

    reportDiv.innerHTML = `
        <div class="analysis-container">
            <div class="info-banner">
                <p><strong>Status:</strong> ✅ Image Uploaded & Analyzed</p>
                <p><strong>Image ID:</strong> <code>${uploadData.image_id}</code> | <strong>Filename:</strong> ${meta.filename} | <strong>Size:</strong> ${(meta.file_size_bytes / 1024).toFixed(1)} KB</p>
            </div>

            <h3 style="margin-top: 1rem; font-size: 1rem; color: var(--text-secondary);">Detected Degradations</h3>
            ${degradationsHtml}

            <h3 style="margin-top: 1.2rem; font-size: 1rem; color: var(--text-secondary);">All Detector Metrics</h3>
            ${metricsHtml}
        </div>
    `;
}

function renderPipelineSteps(steps, totalTime) {
    const pipelineSection = document.getElementById('pipeline-section');
    const stepsDiv = document.getElementById('pipeline-steps');

    pipelineSection.classList.remove('hidden');

    if (!steps || steps.length === 0) {
        stepsDiv.innerHTML = `<p style="color: var(--text-secondary);">No restoration steps required. Original image retained.</p>`;
        return;
    }

    let stepsHtml = '<div class="pipeline-flow">';
    steps.forEach(step => {
        stepsHtml += `
            <div class="step-card">
                <div class="step-num">Step ${step.step_number}</div>
                <div class="step-details">
                    <span class="step-op">${step.operation.replace('_', ' ').toUpperCase()}</span>
                    <span class="step-model">Model: ${step.model_name}</span>
                </div>
                <div class="step-meta">
                    <span>⚡ ${step.execution_time_seconds}s</span>
                    <span>${step.input_size} → ${step.output_size}</span>
                </div>
            </div>
        `;
    });
    stepsHtml += `</div>
        <p class="pipeline-total">Total Pipeline Execution Time: <strong>${totalTime.toFixed(2)} seconds</strong></p>
    `;

    stepsDiv.innerHTML = stepsHtml;
}

function renderResultComparison(imageId, restoreData) {
    const resultSection = document.getElementById('result-section');
    const origImgElem = document.getElementById('original-image');
    const restImgElem = document.getElementById('restored-image');
    const metricsPanel = document.getElementById('metrics-panel');
    const downloadBtn = document.getElementById('download-btn');

    resultSection.classList.remove('hidden');

    // Set Image URLs
    const origUrl = `/api/image/${imageId}`;
    const restoredUrl = `/api/image/restored_${imageId}`;

    origImgElem.src = origUrl;
    restImgElem.src = restoredUrl;

    const m = restoreData.metrics || {};
    metricsPanel.innerHTML = `
        <div class="quality-metrics-box">
            <div class="q-item">
                <span class="q-label">Sharpness Change</span>
                <span class="q-val">${m.sharpness_change_percent ?? 0}%</span>
            </div>
            <div class="q-item">
                <span class="q-label">Original Resolution</span>
                <span class="q-val">${restoreData.original_meta.width} x ${restoreData.original_meta.height}</span>
            </div>
            <div class="q-item">
                <span class="q-label">Restored Resolution</span>
                <span class="q-val">${restoreData.restored_meta.width} x ${restoreData.restored_meta.height}</span>
            </div>
        </div>
    `;

    // Download Button setup
    downloadBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = restoredUrl;
        a.download = `restored_${restoreData.original_meta.filename}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    resultSection.scrollIntoView({ behavior: 'smooth' });
}
