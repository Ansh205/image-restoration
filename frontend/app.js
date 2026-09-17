/**
 * AI Image Restoration — Frontend Logic
 * Phase 1: Upload handling skeleton
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

// --- Upload (placeholder — wired up in Phase 9) ---
uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Processing...';

    // TODO Phase 9: POST to /api/upload and poll /api/status/{job_id}
    console.log('Upload triggered for:', selectedFile.name);
    alert('Upload endpoint not yet implemented. Coming in Phase 9!');

    uploadBtn.disabled = false;
    uploadBtn.textContent = 'Restore Image';
});
