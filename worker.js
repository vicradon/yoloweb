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

onmessage = async (e) => {
  if (e.data.type === 'runModel') {
    try {
      await loadModel();
      const inputTensor = preprocess(e.data.imageData);

      // Update input name if different
      const feeds = { images: inputTensor };
      const results = await session.run(feeds);

      postMessage({ type: 'result', output: results });
    } catch (err) {
      postMessage({ type: 'error', error: err.toString() });
    }
  }
};
