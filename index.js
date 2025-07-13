const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const detectBtn = document.getElementById("detectBtn");
const output = document.getElementById("output");
const outputContainer = document.getElementById("outputContainer");
const emptyState = document.getElementById("emptyState");
const uploadArea = document.getElementById("uploadArea");
const previewContainer = document.getElementById("previewContainer");
const removeImage = document.getElementById("removeImage");
const statusIndicator = document.getElementById("statusIndicator");
const btnText = document.getElementById("btnText");
const loadingSpinner = document.getElementById("loadingSpinner");
const canvasResults = document.getElementById("canvasResults");
const jsonOutput = document.getElementById("jsonOutput");
const resultImage = document.getElementById("resultImage");
const resultCanvas = document.getElementById("resultCanvas");
const detectionList = document.getElementById("detectionList");
const toggleView = document.getElementById("toggleView");

let imageBitmap;
let isProcessing = false;
let showingJson = false;
let originalImageDimensions = { width: 0, height: 0 };

const worker = new Worker("worker.js");

// Color palette for different classes
const colors = [
  "#FF6B6B",
  "#4ECDC4",
  "#45B7D1",
  "#96CEB4",
  "#FFEAA7",
  "#DDA0DD",
  "#98D8C8",
  "#F7DC6F",
  "#BB8FCE",
  "#85C1E9",
  "#F8C471",
  "#82E0AA",
  "#F1948A",
  "#F4D03F",
];

function getColorForClass(classIndex) {
  return colors[classIndex % colors.length];
}

const mycanvas = document.getElementById("mycanvas")
const ctx = mycanvas.getContext("2d");

ctx.clearRect(0, 0, 400, 400);
ctx.lineWidth = 2

// Update status
function updateStatus(text, type = "ready") {
  const dot = statusIndicator.querySelector(".status-dot");
  const span = statusIndicator.querySelector("span");

  dot.className = "status-dot";
  dot.classList.add(`status-${type}`);
  span.textContent = text;
}

// Handle file upload
function handleFileUpload(file) {
  if (!file) return;

  if (file.size > 10 * 1024 * 1024) {
    alert("File size must be less than 10MB");
    return;
  }

  const url = URL.createObjectURL(file);
  preview.src = url;
  previewContainer.classList.remove("hidden");
  detectBtn.disabled = false;

  // Get original image dimensions
  const img = new Image();
  img.onload = function () {
    originalImageDimensions = { width: this.width, height: this.height };
  };
  img.src = url;

  createImageBitmap(file)
    .then((bitmap) => {
      imageBitmap = bitmap;
      updateStatus("Image loaded", "loaded");
    })
    .catch((err) => {
      console.error("Error creating image bitmap:", err);
      updateStatus("Error loading image", "error");
    });
}

// Draw bounding boxes on canvas
function drawBoundingBoxes(boundingBoxes) {
  if (!boundingBoxes || boundingBoxes.length === 0) {
    canvasResults.classList.add("hidden");
    return;
  }

  // Set result image
  resultImage.src = preview.src;

  resultImage.onload = function () {
    const imgWidth = resultImage.naturalWidth;
    const imgHeight = resultImage.naturalHeight;
    const displayWidth = resultImage.offsetWidth;
    const displayHeight = resultImage.offsetHeight;

    // Set canvas size to match displayed image
    resultCanvas.width = displayWidth;
    resultCanvas.height = displayHeight;

    const ctx = resultCanvas.getContext("2d");
    ctx.clearRect(0, 0, displayWidth, displayHeight);

    // Calculate scale factors
    const scaleX = displayWidth / imgWidth;
    const scaleY = displayHeight / imgHeight;

    // Draw bounding boxes
    boundingBoxes.forEach((box, index) => {
      const color = getColorForClass(box.classIndex);

      // Scale coordinates to display size
      const x = box.x1 * scaleX;
      const y = box.y1 * scaleY;
      const width = (box.x2 - box.x1) * scaleX;
      const height = (box.y2 - box.y1) * scaleY;

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      // Draw label background
      const label = `${box.className} (${(box.confidence * 100).toFixed(1)}%)`;
      ctx.font = "12px Arial";
      const textWidth = ctx.measureText(label).width;
      const textHeight = 16;

      ctx.fillStyle = color;
      ctx.fillRect(x, y - textHeight - 2, textWidth + 8, textHeight + 4);

      // Draw label text
      ctx.fillStyle = "white";
      ctx.fillText(label, x + 4, y - 4);
    });

    // Show results
    canvasResults.classList.remove("hidden");
    emptyState.classList.add("hidden");

    // Update detection list
    updateDetectionList(boundingBoxes);
  };
}

// Update detection list
function updateDetectionList(boundingBoxes) {
  detectionList.innerHTML = "";

  if (boundingBoxes.length === 0) {
    detectionList.innerHTML =
      '<div class="text-sm text-geist-muted">No detections found</div>';
    return;
  }

  boundingBoxes.forEach((box, index) => {
    const item = document.createElement("div");
    item.className = "detection-item";

    const color = getColorForClass(box.classIndex);

    item.innerHTML = `
      <div class="detection-color" style="background-color: ${color}"></div>
      <div class="flex-1">
        <div class="text-sm font-medium text-geist-primary">${
          box.className
        }</div>
        <div class="text-xs text-geist-secondary">
          Confidence: ${(box.confidence * 100).toFixed(1)}%
        </div>
      </div>
    `;

    detectionList.appendChild(item);
  });
}

// Toggle between canvas and JSON view
toggleView.addEventListener("click", () => {
  showingJson = !showingJson;

  if (showingJson) {
    canvasResults.classList.add("hidden");
    output.classList.remove("hidden");
    toggleView.textContent = "Show Visual";
  } else {
    canvasResults.classList.remove("hidden");
    output.classList.add("hidden");
    toggleView.textContent = "Show JSON";
  }
});

// Event listeners
uploadArea.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", (e) => {
  handleFileUpload(e.target.files[0]);
});

// Drag and drop
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    handleFileUpload(file);
  }
});

// Remove image
removeImage.addEventListener("click", () => {
  preview.src = "";
  previewContainer.classList.add("hidden");
  detectBtn.disabled = true;
  imageBitmap = null;
  imageInput.value = "";
  updateStatus("Ready", "ready");

  // Reset output
  output.classList.add("hidden");
  canvasResults.classList.add("hidden");
  jsonOutput.classList.add("hidden");
  emptyState.classList.remove("hidden");
  showingJson = false;
});

// Detect button
detectBtn.addEventListener("click", async () => {
  if (!imageBitmap || isProcessing) return;

  isProcessing = true;
  detectBtn.disabled = true;
  btnText.textContent = "Processing...";
  loadingSpinner.classList.remove("hidden");

  updateStatus("Processing...", "processing");

  try {
    const canvas = new OffscreenCanvas(640, 640);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(imageBitmap, 0, 0, 640, 640);
    const imageData = ctx.getImageData(0, 0, 640, 640);

    worker.postMessage({
      type: "runModel",
      imageData: {
        data: imageData.data,
        width: imageData.width,
        height: imageData.height,
      },
    });
  } catch (error) {
    console.error("Error processing image:", error);
    updateStatus("Processing failed", "error");
    resetButton();
  }
});

// Reset button state
function resetButton() {
  isProcessing = false;
  detectBtn.disabled = false;
  btnText.textContent = "Detect Masked Armed Bandits";
  loadingSpinner.classList.add("hidden");
}

// Worker messages
worker.onmessage = (e) => {
  resetButton();

  if (e.data.type === "result") {
    updateStatus("Detection complete", "complete");

    const boundingBoxes = e.data.boundingBoxes || [];

    // Show JSON output section
    jsonOutput.classList.remove("hidden");
    output.textContent = JSON.stringify(boundingBoxes, null, 2);

    // Draw bounding boxes
    drawBoundingBoxes(boundingBoxes);

    // Show visual by default
    showingJson = false;
    toggleView.textContent = "Show JSON";
  } else if (e.data.type === "error") {
    updateStatus("Detection failed", "error");
    emptyState.classList.add("hidden");
    jsonOutput.classList.remove("hidden");
    output.classList.remove("hidden");
    output.classList.add("fade-in");
    output.textContent = `Error: ${e.data.error}`;
  }
};

// Initialize
updateStatus("Ready", "ready");

const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const html = document.documentElement;

function setTheme(theme) {
  if (theme === "dark") {
    html.classList.add("dark");
    themeIcon.textContent = "☀️";
  } else {
    html.classList.remove("dark");
    themeIcon.textContent = "🌙";
  }
  localStorage.setItem("theme", theme);
}

function toggleTheme() {
  const isDark = html.classList.contains("dark");
  setTheme(isDark ? "light" : "dark");
}

themeToggle.addEventListener("click", toggleTheme);

// Apply saved theme on load
const savedTheme = localStorage.getItem("theme") || "light";
setTheme(savedTheme);
