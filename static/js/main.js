const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadForm = document.getElementById('uploadForm');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');

dropArea.addEventListener('click', () => fileInput.click());
browseBtn.addEventListener('click', () => fileInput.click());

dropArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropArea.classList.add('drag-over');
});
dropArea.addEventListener('dragleave', (e) => {
  dropArea.classList.remove('drag-over');
});
dropArea.addEventListener('drop', (e) => {
  e.preventDefault();
  dropArea.classList.remove('drag-over');
  const dt = e.dataTransfer;
  if (dt.files && dt.files.length) {
    fileInput.files = dt.files;
  }
});

uploadForm.addEventListener('submit', (e) => {
  // show spinner and let form submit
  analyzeBtn.disabled = true;
  loading.classList.remove('hidden');
});
