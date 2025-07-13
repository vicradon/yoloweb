importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@dev/dist/';

let session = null;

async function loadModel() {
  if (!session) {
    session = await ort.InferenceSession.create('./yolov8n.onnx');
  }
}

function preprocess(imageData) {
  const { data, width, height } = imageData;
  const floatData = new Float32Array(3 * width * height);
  for (let i = 0; i < width * height; i++) {
    floatData[i] = data[i * 4] / 255.0;               // R
    floatData[i + width * height] = data[i * 4 + 1] / 255.0; // G
    floatData[i + 2 * width * height] = data[i * 4 + 2] / 255.0; // B
  }
  return new ort.Tensor('float32', floatData, [1, 3, height, width]);
}

function postprocess(output, originalWidth, originalHeight, confidenceThreshold = 0.5, iouThreshold = 0.45) {
  // Get the output tensor - typically the first output
  const outputTensor = Object.values(output)[0];
  const data = outputTensor.data;
  const dims = outputTensor.dims;
  
  // For YOLOv8: [1, 84, 8400] -> transpose to [8400, 84]
  const numDetections = dims[2];
  const numClasses = dims[1] - 4; // 84 - 4 = 80 classes
  
  const boxes = [];
  
  for (let i = 0; i < numDetections; i++) {
    // Extract box coordinates (center_x, center_y, width, height)
    const cx = data[i];
    const cy = data[numDetections + i];
    const w = data[2 * numDetections + i];
    const h = data[3 * numDetections + i];
    
    // Find the class with highest confidence
    let maxConfidence = 0;
    let maxClassIndex = 0;
    
    for (let j = 0; j < numClasses; j++) {
      const confidence = data[(4 + j) * numDetections + i];
      if (confidence > maxConfidence) {
        maxConfidence = confidence;
        maxClassIndex = j;
      }
    }
    
    // Filter by confidence threshold
    if (maxConfidence > confidenceThreshold) {
      // Convert from center format to corner format
      const x1 = (cx - w / 2) * originalWidth;
      const y1 = (cy - h / 2) * originalHeight;
      const x2 = (cx + w / 2) * originalWidth;
      const y2 = (cy + h / 2) * originalHeight;
      
      boxes.push({
        x1: Math.max(0, x1),
        y1: Math.max(0, y1),
        x2: Math.min(originalWidth, x2),
        y2: Math.min(originalHeight, y2),
        confidence: maxConfidence,
        classIndex: maxClassIndex,
        className: getClassName(maxClassIndex)
      });
    }
  }
  
  // Apply Non-Maximum Suppression
  return applyNMS(boxes, iouThreshold);
}

function applyNMS(boxes, iouThreshold) {
  // Sort by confidence (descending)
  boxes.sort((a, b) => b.confidence - a.confidence);
  
  const keep = [];
  
  while (boxes.length > 0) {
    const current = boxes.shift();
    keep.push(current);
    
    boxes = boxes.filter(box => {
      if (box.classIndex !== current.classIndex) {
        return true; // Different classes, keep
      }
      
      const iou = calculateIoU(current, box);
      return iou <= iouThreshold;
    });
  }
  
  return keep;
}

function calculateIoU(box1, box2) {
  const x1 = Math.max(box1.x1, box2.x1);
  const y1 = Math.max(box1.y1, box2.y1);
  const x2 = Math.min(box1.x2, box2.x2);
  const y2 = Math.min(box1.y2, box2.y2);
  
  if (x2 <= x1 || y2 <= y1) {
    return 0;
  }
  
  const intersection = (x2 - x1) * (y2 - y1);
  const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  const union = area1 + area2 - intersection;
  
  return intersection / union;
}

function getClassName(classIndex) {
  // COCO class names (80 classes)
  const classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
  ];
  
  return classes[classIndex] || `class_${classIndex}`;
}

onmessage = async (e) => {
  if (e.data.type === 'runModel') {
    try {
      await loadModel();
      const inputTensor = preprocess(e.data.imageData);

      const feeds = { images: inputTensor };
      const results = await session.run(feeds);

      // Post-process to get bounding boxes
      const boundingBoxes = postprocess(
        results, 
        e.data.imageData.width, 
        e.data.imageData.height,
        0.5, // confidence threshold
        0.45 // IoU threshold
      );

      console.log(boundingBoxes)

      postMessage({ 
        type: 'result', 
        boundingBoxes: boundingBoxes,
        rawOutput: results // Optional: include raw output for debugging
      });
    } catch (err) {
      postMessage({ type: 'error', error: err.toString() });
    }
  }
};