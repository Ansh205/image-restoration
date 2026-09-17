/**
 * AI Image Restoration — Frontend Logic
 * Phase 4: Combined Upload & Degradation Analysis
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

// --- Upload & Analyze Handling ---
uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Processing & Analyzing...';

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
        const analyzeResp = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_id: imageId }),
        });

        const analyzeData = await analyzeResp.json();

        // 3. Render Results on UI
        renderAnalysisReport(uploadData, analyzeData);

    } catch (err) {
        console.error('Error during upload & analysis:', err);
        alert('Error connecting to backend API: ' + err.message);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Restore Image';
    }
});

function renderAnalysisReport(uploadData, analyzeData) {
    const analysisSection = document.getElementById('analysis-section');
    const reportDiv = document.getElementById('degradation-report');

    analysisSection.classList.remove('hidden');

    const meta = uploadData.meta;
    const degradations = analyzeData.degradations || [];
    const metrics = analyzeData.raw_metrics || {};

    // Generate Badges for Detected Degradations
    let degradationsHtml = '';
    if (degradations.length === 0) {
        degradationsHtml = `<p style="color: var(--success); font-weight: 600;">✨ No major degradations detected (Image appears clean)</p>`;
    } else {
        degradationsHtml = '<div class="degradation-list">';
        degradations.forEach(deg => {
            const sevClass = deg.severity.toLowerCase(); // 'low', 'medium', 'high'
            degradationsHtml += `
                <div class="deg-card deg-${sevClass}">
                    <span class="deg-name">${deg.name.replace('_', ' ').toUpperCase()}</span>
                    <span class="deg-badge badge-${sevClass}">${deg.severity} (${(deg.score * 100).toFixed(0)}%)</span>
                </div>
            `;
        });
        degradationsHtml += '</div>';
    }

    // Format raw metrics table
    const metricsHtml = `
        <div class="metrics-grid">
            <div class="metric-box">
                <span class="metric-label">Laplacian Var (Blur)</span>
                <span class="metric-val">${metrics.laplacian_variance ?? 'N/A'}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">Noise Sigma (σ)</span>
                <span class="metric-val">${metrics.estimated_noise_sigma ?? 'N/A'}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">Mean Luminance</span>
                <span class="metric-val">${metrics.mean_luminance ?? 'N/A'}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">JPEG Block Score</span>
                <span class="metric-val">${metrics.jpeg_blocking_score ?? 'N/A'}</span>
            </div>
            <div class="metric-box">
                <span class="metric-label">Dimensions</span>
                <span class="metric-val">${meta.width} x ${meta.height} px</span>
            </div>
        </div>
    `;

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
