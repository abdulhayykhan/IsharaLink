/**
 * IsharaLink — Frontend Application Logic (Muhammad Fasih)
 *
 * Implements:
 *   1. MediaPipe Holistic hand landmark tracking (left_hand/right_hand, conf 0.4)
 *   2. Real-time canvas overlay rendering
 *   3. 3-2-1 countdown & 2-second capture (~60 raw frames)
 *   4. Handedness / mirror check
 *   5. FastAPI /predict integration with unknown sign rejection
 *   6. Top-3 predictions and animated confidence meter
 *   7. Sentence builder with undo, clear, copy & SpeechSynthesis
 *   8. Supported vocabulary explorer (62 classes from GET /classes)
 *   9. Golden test sample runner & JSON debug exporter
 *  10. Results & confusion matrix gallery
 */

// --- Configuration & Constants ---------------------------------------------
const CONFIG = {
  defaultApiBaseUrl: 'http://localhost:8000',
  captureDurationMs: 2000,       // ~2 seconds recording window
  targetMinFrames: 30,           // minimum frames required
  targetMaxFrames: 60,           // standard clip size
  detectionConfidence: 0.4,      // MediaPipe detection confidence
  trackingConfidence: 0.4,       // MediaPipe tracking confidence
};

// 62 Supported PSL Classes Contract (Fallback if backend unreachable initially)
const FALLBACK_CLASSES = [
  "absolutely", "aircrash", "airplane", "all", "also", "arrival", "assalam-o-alaikum",
  "atm", "bald", "beach", "beak", "bear", "beard", "bed", "bench", "bicycle", "both",
  "bridge", "bring", "bulb", "cartoon", "chimpanzee", "color_pencils", "cow", "crow",
  "cupboard", "deer", "dog", "donttouch", "door", "elephant", "excuseme", "facelotion",
  "fan", "garden", "generator", "goodbye", "goodmorning", "have_a_good_day", "hello",
  "ihaveacomplaint", "left_hand", "lifejacket", "mine", "mobile_phone", "nailcutter",
  "nothing", "peacock", "policecar", "razor", "shampoo", "shower", "sunglasses",
  "thankyou", "tissue", "toothbrush", "toothpaste", "umbrella", "water", "we",
  "welldone", "you"
];

// Urdu translations dictionary for richer cultural feedback
const URDU_DICTIONARY = {
  "absolutely": "بالکل",
  "aircrash": "طیارے کا حادثہ",
  "airplane": "ہوائی جہاز",
  "all": "سب",
  "also": "بھی",
  "arrival": "آمد",
  "assalam-o-alaikum": "السلام علیکم",
  "atm": "اے ٹی ایم",
  "bald": "گنجا",
  "beach": "ساحل",
  "beak": "چونچ",
  "bear": "ریچھ",
  "beard": "داڑھی",
  "bed": "بستر",
  "bench": "بینچ",
  "bicycle": "سائیکل",
  "both": "دونوں",
  "bridge": "پل",
  "bring": "لانا",
  "bulb": "بلب",
  "cartoon": "کارٹون",
  "chimpanzee": "چمپینزی",
  "color_pencils": "رنگین پنسلیں",
  "cow": "گائے",
  "crow": "کوا",
  "cupboard": "الماری",
  "deer": "ہرن",
  "dog": "کتا",
  "donttouch": "ہاتھ مت لگائیں",
  "door": "دروازہ",
  "elephant": "ہاتھی",
  "excuseme": "معاف کیجئے",
  "facelotion": "فیس لوشن",
  "fan": "پنکھا",
  "garden": "باغ",
  "generator": "جنریٹر",
  "goodbye": "الوداع",
  "goodmorning": "صبح بخیر",
  "have_a_good_day": "آپ کا دن اچھا گزرے",
  "hello": "ہیلو",
  "ihaveacomplaint": "میری شکایت ہے",
  "left_hand": "بایاں ہاتھ",
  "lifejacket": "لائف جیکٹ",
  "mine": "میرا",
  "mobile_phone": "موبائل فون",
  "nailcutter": "ناخن تراش",
  "nothing": "کچھ نہیں",
  "peacock": "مور",
  "policecar": "پولیس کی گاڑی",
  "razor": "استرا",
  "shampoo": "شیمپو",
  "shower": "شاور",
  "sunglasses": "دھوپ کا چشمہ",
  "thankyou": "شکریہ",
  "tissue": "ٹشو",
  "toothbrush": "ٹوتھ برش",
  "toothpaste": "ٹوتھ پیسٹ",
  "umbrella": "چھتری",
  "water": "پانی",
  "we": "ہم",
  "welldone": "شاباش",
  "you": "آپ"
};

// Hand skeletal connections for 21 landmarks
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],// Ring
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20] // Pinky
];

// --- State Management ------------------------------------------------------
const state = {
  apiBaseUrl: localStorage.getItem('isharalink_api_url') || CONFIG.defaultApiBaseUrl,
  cameraActive: false,
  stream: null,
  holistic: null,
  cameraUtil: null,
  isRecording: false,
  isCountingDown: false,
  captureBuffer: [],
  captureStartTime: null,
  lastCapturedFrames: null,
  sentenceWords: [],
  classes: [...FALLBACK_CLASSES],
  fpsCounter: { frames: 0, lastTime: performance.now(), currentFps: 0 },
  mirrorCamera: true,
  swapHands: false,
  autoSpeak: false,
  goldenCasesCache: null,
};

// --- DOM Elements ----------------------------------------------------------
const DOM = {
  video: document.getElementById('webcam'),
  canvas: document.getElementById('overlay-canvas'),
  canvasCtx: null,

  // Status pills
  backendStatusPill: document.getElementById('backend-status-pill'),
  backendStatusText: document.getElementById('backend-status-text'),
  holisticStatusPill: document.getElementById('holistic-status-pill'),
  holisticStatusText: document.getElementById('holistic-status-text'),
  cameraFpsBadge: document.getElementById('camera-fps-badge'),
  handsDetectedBadge: document.getElementById('hands-detected-badge'),
  cameraPlaceholder: document.getElementById('camera-placeholder'),

  // Overlays
  countdownOverlay: document.getElementById('countdown-overlay'),
  countdownNumber: document.getElementById('countdown-number'),
  countdownLabel: document.getElementById('countdown-label'),
  captureProgressContainer: document.getElementById('capture-progress-container'),
  captureProgressFill: document.getElementById('capture-progress-fill'),
  captureFrameCount: document.getElementById('capture-frame-count'),

  // Buttons & Controls
  btnToggleCamera: document.getElementById('btn-toggle-camera'),
  btnStartCameraHero: document.getElementById('btn-start-camera-hero'),
  cameraBtnText: document.getElementById('camera-btn-text'),
  btnRecordSign: document.getElementById('btn-record-sign'),
  btnFlipMirror: document.getElementById('btn-flip-mirror'),

  // Settings
  btnToggleSettings: document.getElementById('btn-toggle-settings'),
  btnCloseSettings: document.getElementById('btn-close-settings'),
  settingsPanel: document.getElementById('settings-panel'),
  apiBaseUrlInput: document.getElementById('api-base-url'),
  btnSaveApiUrl: document.getElementById('btn-save-api-url'),
  chkAutoSpeak: document.getElementById('chk-auto-speak'),
  chkMirrorCamera: document.getElementById('chk-mirror-camera'),
  chkSwapHands: document.getElementById('chk-swap-hands'),

  // Result Card Elements
  inferenceTimeBadge: document.getElementById('inference-time-badge'),
  resultStateIdle: document.getElementById('result-state-idle'),
  resultStateLoading: document.getElementById('result-state-loading'),
  resultStateSuccess: document.getElementById('result-state-success'),
  resultStateUnknown: document.getElementById('result-state-unknown'),
  resultStateError: document.getElementById('result-state-error'),
  resultWord: document.getElementById('result-word'),
  resultWordUrdu: document.getElementById('result-word-urdu'),
  confidencePercentage: document.getElementById('confidence-percentage'),
  confidenceBarFill: document.getElementById('confidence-bar-fill'),
  top3List: document.getElementById('top3-list'),
  errorMessageText: document.getElementById('error-message-text'),
  btnRetryPredict: document.getElementById('btn-retry-predict'),

  // Sentence Builder Elements
  sentenceChips: document.getElementById('sentence-chips'),
  wordCountBadge: document.getElementById('word-count-badge'),
  btnSpeakSentence: document.getElementById('btn-speak-sentence'),
  btnUndoWord: document.getElementById('btn-undo-word'),
  btnClearSentence: document.getElementById('btn-clear-sentence'),
  btnCopySentence: document.getElementById('btn-copy-sentence'),

  // Debug & Golden Samples
  sampleSelect: document.getElementById('sample-select'),
  btnReplaySample: document.getElementById('btn-replay-sample'),
  btnExportJson: document.getElementById('btn-export-json'),
  fileInputJson: document.getElementById('file-input-json'),

  // Vocabulary & Gallery
  vocabGrid: document.getElementById('vocab-grid'),
  vocabSearch: document.getElementById('vocab-search'),
  vocabCountBadge: document.getElementById('vocab-count-badge'),
  galleryTabs: document.querySelectorAll('.gallery-tabs .tab-btn'),
  activeGalleryImg: document.getElementById('active-gallery-img'),
};

DOM.canvasCtx = DOM.canvas.getContext('2d');

// --- Initialization --------------------------------------------------------
async function initializeApp() {
  console.log('🚀 Initializing IsharaLink Frontend...');
  
  // Set initial settings inputs
  DOM.apiBaseUrlInput.value = state.apiBaseUrl;
  DOM.chkMirrorCamera.checked = state.mirrorCamera;
  DOM.chkSwapHands.checked = state.swapHands;
  DOM.chkAutoSpeak.checked = state.autoSpeak;
  applyMirrorClass();

  // Attach event listeners
  setupEventListeners();

  // Initialize MediaPipe Holistic Tracker
  await initMediaPipeHolistic();

  // Test backend connectivity & fetch classes
  await checkBackendHealth();
  await fetchSupportedClasses();

  // Render initial vocabulary list
  renderVocabularyGrid(state.classes);

  // Pre-load golden test samples
  loadGoldenSamples();
}

// --- MediaPipe Holistic Setup (Step 2) -------------------------------------
async function initMediaPipeHolistic() {
  setHolisticStatus('connecting', 'Loading Model...');

  try {
    if (typeof Holistic === 'undefined') {
      throw new Error('MediaPipe Holistic script could not be loaded from CDN.');
    }

    state.holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    state.holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: CONFIG.detectionConfidence,
      minTrackingConfidence: CONFIG.trackingConfidence,
      refineFaceLandmarks: false,
    });

    state.holistic.onResults(onHolisticResults);

    setHolisticStatus('ok', 'Tracker Ready (0.4 Conf)');
    console.log('✅ MediaPipe Holistic initialized successfully.');
  } catch (err) {
    console.error('❌ Failed to initialize MediaPipe Holistic:', err);
    setHolisticStatus('error', 'Tracker Error');
  }
}

function setHolisticStatus(type, message) {
  DOM.holisticStatusPill.className = `status-pill status-${type}`;
  DOM.holisticStatusText.textContent = `Tracker: ${message}`;
}

// --- Camera Management (Step 1) -------------------------------------------
async function startCamera() {
  if (state.cameraActive) return;

  try {
    DOM.btnToggleCamera.disabled = true;
    DOM.btnStartCameraHero.disabled = true;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user',
      },
      audio: false,
    });

    state.stream = stream;
    DOM.video.srcObject = stream;
    await DOM.video.play();

    state.cameraActive = true;
    DOM.cameraPlaceholder.classList.add('hidden');
    DOM.btnRecordSign.disabled = false;
    DOM.cameraBtnText.textContent = 'Stop Camera';
    DOM.btnToggleCamera.classList.remove('btn-secondary');
    DOM.btnToggleCamera.classList.add('btn-outline');

    // Sync canvas dimensions
    DOM.canvas.width = DOM.video.videoWidth || 640;
    DOM.canvas.height = DOM.video.videoHeight || 480;

    // Use MediaPipe Camera helper for continuous detection loop
    if (typeof Camera !== 'undefined') {
      state.cameraUtil = new Camera(DOM.video, {
        onFrame: async () => {
          if (state.cameraActive && state.holistic) {
            await state.holistic.send({ image: DOM.video });
            updateFps();
          }
        },
        width: 640,
        height: 480,
      });
      state.cameraUtil.start();
    } else {
      // Fallback requestAnimationFrame loop
      requestAnimationFrame(runFrameLoop);
    }

    console.log('📷 Camera started:', DOM.canvas.width, 'x', DOM.canvas.height);
  } catch (err) {
    console.error('❌ Camera error:', err);
    alert(`Could not start camera: ${err.message}. Please allow camera permissions.`);
  } finally {
    DOM.btnToggleCamera.disabled = false;
    DOM.btnStartCameraHero.disabled = false;
  }
}

async function runFrameLoop() {
  if (!state.cameraActive) return;
  if (state.holistic && DOM.video.readyState >= 2) {
    await state.holistic.send({ image: DOM.video });
    updateFps();
  }
  requestAnimationFrame(runFrameLoop);
}

function stopCamera() {
  if (!state.cameraActive) return;

  if (state.cameraUtil) {
    state.cameraUtil.stop();
    state.cameraUtil = null;
  }

  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }

  state.cameraActive = false;
  DOM.video.srcObject = null;
  DOM.cameraPlaceholder.classList.remove('hidden');
  DOM.btnRecordSign.disabled = true;
  DOM.cameraBtnText.textContent = 'Start Camera';
  DOM.btnToggleCamera.classList.remove('btn-outline');
  DOM.btnToggleCamera.classList.add('btn-secondary');
  DOM.cameraFpsBadge.textContent = '0 FPS';
  DOM.handsDetectedBadge.textContent = 'No Hands Detected';
  DOM.handsDetectedBadge.className = 'badge badge-neutral';

  // Clear canvas
  DOM.canvasCtx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);
}

function updateFps() {
  const now = performance.now();
  state.fpsCounter.frames++;
  if (now - state.fpsCounter.lastTime >= 1000) {
    state.fpsCounter.currentFps = Math.round((state.fpsCounter.frames * 1000) / (now - state.fpsCounter.lastTime));
    DOM.cameraFpsBadge.textContent = `${state.fpsCounter.currentFps} FPS`;
    state.fpsCounter.frames = 0;
    state.fpsCounter.lastTime = now;
  }
}

function applyMirrorClass() {
  if (state.mirrorCamera) {
    DOM.video.classList.add('mirrored');
    DOM.canvas.classList.add('mirrored');
  } else {
    DOM.video.classList.remove('mirrored');
    DOM.canvas.classList.remove('mirrored');
  }
}

// --- MediaPipe Holistic OnResults Callback (Step 2, 3, 4) ------------------
function onHolisticResults(results) {
  const ctx = DOM.canvasCtx;
  const width = DOM.canvas.width;
  const height = DOM.canvas.height;

  // Clear canvas
  ctx.save();
  ctx.clearRect(0, 0, width, height);

  const leftHand = results.leftHandLandmarks || null;
  const rightHand = results.rightHandLandmarks || null;

  // Update hands detection badge
  updateHandsBadge(leftHand, rightHand);

  // Draw hand landmarks and skeletal connections with palette styling
  if (leftHand) {
    drawHandSkeleton(ctx, leftHand, width, height, '#818865', '#BB8D3F');
  }
  if (rightHand) {
    drawHandSkeleton(ctx, rightHand, width, height, '#575B42', '#BB8D3F');
  }


  ctx.restore();

  // If in recording mode, buffer frame landmarks
  if (state.isRecording) {
    captureCurrentFrame(leftHand, rightHand);
  }
}

function updateHandsBadge(left, right) {
  if (left && right) {
    DOM.handsDetectedBadge.textContent = 'Both Hands Detected';
    DOM.handsDetectedBadge.className = 'badge badge-active';
  } else if (left) {
    DOM.handsDetectedBadge.textContent = 'Left Hand Detected';
    DOM.handsDetectedBadge.className = 'badge badge-accent';
  } else if (right) {
    DOM.handsDetectedBadge.textContent = 'Right Hand Detected';
    DOM.handsDetectedBadge.className = 'badge badge-info';
  } else {
    DOM.handsDetectedBadge.textContent = 'No Hands Detected';
    DOM.handsDetectedBadge.className = 'badge badge-neutral';
  }
}

function drawHandSkeleton(ctx, landmarks, width, height, strokeColor, fillColor) {
  // Draw skeletal connections
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const p1 = landmarks[startIdx];
    const p2 = landmarks[endIdx];
    if (p1 && p2) {
      ctx.beginPath();
      ctx.moveTo(p1.x * width, p1.y * height);
      ctx.lineTo(p2.x * width, p2.y * height);
      ctx.stroke();
    }
  }

  // Draw landmark points
  for (let i = 0; i < landmarks.length; i++) {
    const pt = landmarks[i];
    const px = pt.x * width;
    const py = pt.y * height;

    ctx.beginPath();
    ctx.arc(px, py, i === 0 || i === 9 ? 6 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

// --- Frame Capture & Landmark Formatting (Step 3 & 4) ----------------------
function captureCurrentFrame(leftLandmarks, rightLandmarks) {
  // Format hand points to exactly [[x, y, z] x 21] or null
  const formatPoints = (lmList) => {
    if (!lmList || lmList.length !== 21) return null;
    return lmList.map((pt) => [
      parseFloat(pt.x.toFixed(6)),
      parseFloat(pt.y.toFixed(6)),
      parseFloat(pt.z.toFixed(6)),
    ]);
  };

  let leftData = formatPoints(leftLandmarks);
  let rightData = formatPoints(rightLandmarks);

  // Handedness check / swap override (Step 4)
  if (state.swapHands) {
    const temp = leftData;
    leftData = rightData;
    rightData = temp;
  }

  // Record raw frame per contract
  state.captureBuffer.push({
    left: leftData,
    right: rightData,
  });

  // Update progress bar
  const count = state.captureBuffer.length;
  const elapsed = performance.now() - state.captureStartTime;
  const pct = Math.min(100, Math.max((count / CONFIG.targetMaxFrames) * 100, (elapsed / CONFIG.captureDurationMs) * 100));

  DOM.captureProgressFill.style.width = `${pct}%`;
  DOM.captureFrameCount.textContent = `${count} frames (${(elapsed / 1000).toFixed(1)}s)`;

  // If time elapsed or max frames reached, finalize recording
  if (elapsed >= CONFIG.captureDurationMs || count >= CONFIG.targetMaxFrames) {
    stopRecording();
  }
}

// --- Sign Recording Workflow (Step 3) ---------------------------------------
async function triggerRecordSequence() {
  if (!state.cameraActive || state.isRecording || state.isCountingDown) return;

  state.isCountingDown = true;
  DOM.btnRecordSign.disabled = true;
  DOM.countdownOverlay.classList.remove('hidden');

  // 3-2-1 Countdown
  const counts = [3, 2, 1];
  for (const num of counts) {
    DOM.countdownNumber.textContent = num;
    DOM.countdownLabel.textContent = 'Get Ready to Sign';
    await waitMs(900);
  }

  // Flash Sign Prompt
  DOM.countdownNumber.textContent = 'GO!';
  DOM.countdownLabel.textContent = 'Signing Window Active';
  await waitMs(400);

  DOM.countdownOverlay.classList.add('hidden');
  state.isCountingDown = false;

  // Start Capture
  startRecording();
}

function startRecording() {
  state.isRecording = true;
  state.captureBuffer = [];
  state.captureStartTime = performance.now();

  DOM.captureProgressContainer.classList.remove('hidden');
  DOM.captureProgressFill.style.width = '0%';
  DOM.captureFrameCount.textContent = '0 / 60 frames';
}

function stopRecording() {
  if (!state.isRecording) return;
  state.isRecording = false;

  DOM.captureProgressContainer.classList.add('hidden');
  DOM.btnRecordSign.disabled = false;

  const capturedCount = state.captureBuffer.length;
  console.log(`🎬 Captured ${capturedCount} raw frames`);

  if (capturedCount < CONFIG.targetMinFrames) {
    console.warn(`Captured clip too short (${capturedCount} frames, min ${CONFIG.targetMinFrames})`);
  }

  // Preserve captured frames for debugging/export (Step 9)
  state.lastCapturedFrames = [...state.captureBuffer];
  DOM.btnExportJson.disabled = false;

  // Send capture to backend API
  sendPredictionRequest(state.captureBuffer);
}

function waitMs(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// --- FastAPI /predict Integration (Step 5 & 6) -----------------------------
async function sendPredictionRequest(frames) {
  if (!frames || frames.length === 0) {
    showErrorState('No frames captured. Please ensure the camera is tracking your hands.');
    return;
  }

  showLoadingState();
  const startTime = performance.now();

  try {
    const url = `${state.apiBaseUrl.replace(/\/+$/, '')}/predict`;
    const payload = {
      frames: frames,
      client: 'web',
    };

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const elapsedMs = (performance.now() - startTime).toFixed(1);

    if (!response.ok) {
      let errDetail = `Server responded with ${response.status}`;
      try {
        const errJson = await response.json();
        errDetail = errJson.detail || JSON.stringify(errJson);
      } catch (e) {
        errDetail = await response.text();
      }
      throw new Error(errDetail);
    }

    const data = await response.json();
    const serverProcessMs = response.headers.get('X-Process-Time-Ms') || elapsedMs;

    console.log('🎯 Predict response:', data, `(${serverProcessMs}ms)`);
    DOM.inferenceTimeBadge.textContent = `${serverProcessMs}ms • ${frames.length} frames`;

    handlePredictionOutcome(data);
  } catch (err) {
    console.error('❌ Prediction request failed:', err);
    showErrorState(`API Error: ${err.message}. Ensure backend is running at ${state.apiBaseUrl}.`);
  }
}

// --- Result UI & Rejection Logic (Step 6) -----------------------------------
function handlePredictionOutcome(data) {
  const { label, confidence, top3, status } = data;

  // If status is "unknown", display "unknown sign, try again" instead of a word!
  if (status === 'unknown') {
    showUnknownState(
      `Model confidence (${(confidence * 100).toFixed(1)}%) was below the rejection threshold, or the gesture was not recognized.`
    );
    return;
  }

  // Successful Prediction
  showSuccessState(label, confidence, top3);

  // Append word to sentence builder
  appendWordToSentence(label);

  // Optional Text-to-Speech
  if (state.autoSpeak) {
    speakText(label);
  }
}

function showLoadingState() {
  DOM.resultStateIdle.classList.add('hidden');
  DOM.resultStateSuccess.classList.add('hidden');
  DOM.resultStateUnknown.classList.add('hidden');
  DOM.resultStateError.classList.add('hidden');
  DOM.resultStateLoading.classList.remove('hidden');
  DOM.inferenceTimeBadge.textContent = 'Processing...';
}

function showSuccessState(label, confidence, top3) {
  DOM.resultStateLoading.classList.add('hidden');
  DOM.resultStateUnknown.classList.add('hidden');
  DOM.resultStateError.classList.add('hidden');
  DOM.resultStateIdle.classList.add('hidden');
  DOM.resultStateSuccess.classList.remove('hidden');

  // Word & Urdu Translation
  DOM.resultWord.textContent = label.replace(/_/g, ' ');
  const urduText = URDU_DICTIONARY[label] || '';
  DOM.resultWordUrdu.textContent = urduText;

  // Confidence
  const pct = Math.min(100, Math.round(confidence * 1000) / 10);
  DOM.confidencePercentage.textContent = `${pct.toFixed(1)}%`;
  DOM.confidenceBarFill.style.width = `${pct}%`;

  // Top 3 Candidates
  DOM.top3List.innerHTML = '';
  if (top3 && top3.length > 0) {
    top3.forEach((item, idx) => {
      const top3Item = document.createElement('div');
      top3Item.className = 'top3-item';
      const probPct = (item.p * 100).toFixed(1);
      const urduSub = URDU_DICTIONARY[item.label] ? ` (${URDU_DICTIONARY[item.label]})` : '';

      top3Item.innerHTML = `
        <span class="top3-item-label">${idx + 1}. ${item.label.replace(/_/g, ' ')}${urduSub}</span>
        <span class="top3-item-prob">${probPct}%</span>
      `;
      DOM.top3List.appendChild(top3Item);
    });
  }
}

function showUnknownState(reason) {
  DOM.resultStateLoading.classList.add('hidden');
  DOM.resultStateSuccess.classList.add('hidden');
  DOM.resultStateError.classList.add('hidden');
  DOM.resultStateIdle.classList.add('hidden');
  DOM.resultStateUnknown.classList.remove('hidden');

  document.getElementById('unknown-reason').textContent = reason;
}

function showErrorState(msg) {
  DOM.resultStateLoading.classList.add('hidden');
  DOM.resultStateSuccess.classList.add('hidden');
  DOM.resultStateUnknown.classList.add('hidden');
  DOM.resultStateIdle.classList.add('hidden');
  DOM.resultStateError.classList.remove('hidden');

  DOM.errorMessageText.textContent = msg;
}

// --- Sentence Construction & Text-to-Speech (Step 7) -----------------------
function appendWordToSentence(word) {
  state.sentenceWords.push(word);
  renderSentenceChips();
}

function undoLastWord() {
  if (state.sentenceWords.length > 0) {
    state.sentenceWords.pop();
    renderSentenceChips();
  }
}

function clearSentence() {
  state.sentenceWords = [];
  renderSentenceChips();
}

function renderSentenceChips() {
  const container = DOM.sentenceChips;
  container.innerHTML = '';

  if (state.sentenceWords.length === 0) {
    container.innerHTML = '<span class="sentence-placeholder">Accepted signs will automatically build up here into a continuous phrase...</span>';
    DOM.wordCountBadge.textContent = '0 words';
    return;
  }

  DOM.wordCountBadge.textContent = `${state.sentenceWords.length} words`;

  state.sentenceWords.forEach((w, index) => {
    const chip = document.createElement('div');
    chip.className = 'word-chip';
    const urdu = URDU_DICTIONARY[w] ? ` (${URDU_DICTIONARY[w]})` : '';
    chip.innerHTML = `
      <span>${w.replace(/_/g, ' ')}${urdu}</span>
      <span class="word-chip-remove" title="Remove word" data-index="${index}">&times;</span>
    `;
    chip.querySelector('.word-chip-remove').addEventListener('click', (e) => {
      e.stopPropagation();
      state.sentenceWords.splice(index, 1);
      renderSentenceChips();
    });
    container.appendChild(chip);
  });
}

function copySentenceToClipboard() {
  if (state.sentenceWords.length === 0) return;
  const sentence = state.sentenceWords.map((w) => w.replace(/_/g, ' ')).join(' ');
  navigator.clipboard.writeText(sentence).then(() => {
    DOM.btnCopySentence.textContent = '✅ Copied!';
    setTimeout(() => {
      DOM.btnCopySentence.textContent = '📋 Copy';
    }, 1500);
  });
}

function speakText(text) {
  if (!('speechSynthesis' in window)) {
    console.warn('Speech synthesis not supported in this browser.');
    return;
  }

  window.speechSynthesis.cancel(); // Stop any pending speech
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.95;
  utterance.pitch = 1.0;

  // Try to find an Urdu or South Asian voice
  const voices = window.speechSynthesis.getVoices();
  const urduVoice = voices.find((v) => v.lang.startsWith('ur') || v.name.toLowerCase().includes('pakistan'));
  if (urduVoice) {
    utterance.voice = urduVoice;
  }

  window.speechSynthesis.speak(utterance);
}

function speakCurrentSentence() {
  if (state.sentenceWords.length === 0) {
    alert('The sentence box is empty. Record or select signs first.');
    return;
  }
  const phrase = state.sentenceWords.map((w) => w.replace(/_/g, ' ')).join(' ');
  speakText(phrase);
}

// --- Supported Vocabulary List (Step 8) ------------------------------------
async function fetchSupportedClasses() {
  try {
    const url = `${state.apiBaseUrl.replace(/\/+$/, '')}/classes`;
    const res = await fetch(url);
    if (res.ok) {
      const classesList = await res.json();
      if (Array.isArray(classesList) && classesList.length > 0) {
        state.classes = classesList;
        renderVocabularyGrid(state.classes);
        console.log(`📚 Loaded ${classesList.length} classes from backend.`);
      }
    }
  } catch (err) {
    console.warn('Could not fetch classes from backend; using 62 fallback classes.', err.message);
  }
}

function renderVocabularyGrid(classList) {
  DOM.vocabGrid.innerHTML = '';
  DOM.vocabCountBadge.textContent = `${classList.length} Words`;

  classList.forEach((word) => {
    const badge = document.createElement('div');
    badge.className = 'vocab-badge';
    const urdu = URDU_DICTIONARY[word] ? ` • ${URDU_DICTIONARY[word]}` : '';
    badge.textContent = `${word.replace(/_/g, ' ')}${urdu}`;
    badge.title = `PSL Sign: ${word}`;
    badge.addEventListener('click', () => {
      // Clicking a word in the vocabulary list adds it to the sentence
      appendWordToSentence(word);
    });
    DOM.vocabGrid.appendChild(badge);
  });
}

function filterVocabulary(query) {
  const q = query.trim().toLowerCase();
  const filtered = state.classes.filter((w) => {
    const urdu = URDU_DICTIONARY[w] || '';
    return w.toLowerCase().includes(q) || urdu.includes(q);
  });
  renderVocabularyGrid(filtered);
}

// --- Backend Health Check --------------------------------------------------
async function checkBackendHealth() {
  setBackendStatus('connecting', 'Connecting...');
  try {
    const url = `${state.apiBaseUrl.replace(/\/+$/, '')}/health`;
    const res = await fetch(url, { signal: AbortSignal.timeout(3500) });
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'ok' && data.model_loaded) {
        setBackendStatus('ok', `Online (${data.classes_count} classes)`);
        return true;
      }
    }
    setBackendStatus('error', 'Model Not Loaded');
    return false;
  } catch (err) {
    setBackendStatus('error', 'Offline (Start Backend)');
    return false;
  }
}

function setBackendStatus(type, message) {
  DOM.backendStatusPill.className = `status-pill status-${type}`;
  DOM.backendStatusText.textContent = `Backend: ${message}`;
}

// --- Developer & Golden Test Replay (Step 9) --------------------------------
async function fetchCasesData() {
  const candidateUrls = [
    '/tests/preprocess_cases.json',
    '../tests/preprocess_cases.json',
    `${state.apiBaseUrl.replace(/\/+$/, '')}/tests/preprocess_cases.json`
  ];
  for (const url of candidateUrls) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        const json = await res.json();
        if (json && json.cases) return json.cases;
      }
    } catch (e) {
      // try next candidate
    }
  }
  return null;
}

async function loadGoldenSamples() {
  const cases = await fetchCasesData();
  if (cases) {
    state.goldenCasesCache = cases;
    console.log(`🧪 Cached ${cases.length} golden test cases.`);
  } else {
    console.log('Note: preprocess_cases.json will be loaded on demand.');
  }
}

async function replayGoldenSample(sampleKey) {
  if (!sampleKey) return;

  console.log(`🧪 Replaying golden sample: ${sampleKey}`);

  let framesToTest = null;

  if (sampleKey === 'no_hands') {
    // Generate 60 empty frames
    framesToTest = Array.from({ length: 60 }, () => ({ left: null, right: null }));
  } else {
    if (!state.goldenCasesCache) {
      state.goldenCasesCache = await fetchCasesData();
    }
    if (state.goldenCasesCache) {
      const found = state.goldenCasesCache.find((c) => c.name === sampleKey);
      if (found && found.frames) {
        framesToTest = found.frames;
      }
    }
  }


  if (!framesToTest) {
    alert(`Golden sample "${sampleKey}" could not be loaded.`);
    return;
  }


  state.lastCapturedFrames = framesToTest;
  DOM.btnExportJson.disabled = false;
  sendPredictionRequest(framesToTest);
}

function exportCapturedJson() {
  if (!state.lastCapturedFrames || state.lastCapturedFrames.length === 0) {
    alert('No captured landmarks to export.');
    return;
  }

  const exportData = {
    client: 'isharalink_web',
    timestamp: new Date().toISOString(),
    frames_count: state.lastCapturedFrames.length,
    frames: state.lastCapturedFrames,
  };

  const jsonStr = JSON.stringify(exportData, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = `isharalink_capture_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function loadCustomJsonFile(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result);
      const frames = Array.isArray(data) ? data : data.frames;
      if (!frames || !Array.isArray(frames)) {
        throw new Error('Invalid JSON format: missing "frames" array.');
      }
      console.log(`📂 Loaded ${frames.length} frames from ${file.name}`);
      state.lastCapturedFrames = frames;
      DOM.btnExportJson.disabled = false;
      sendPredictionRequest(frames);
    } catch (err) {
      alert(`Could not parse JSON file: ${err.message}`);
    }
  };
  reader.readAsText(file);
}

// --- Event Listeners Setup -------------------------------------------------
function setupEventListeners() {
  // Camera toggle
  DOM.btnToggleCamera.addEventListener('click', () => {
    if (state.cameraActive) stopCamera();
    else startCamera();
  });
  DOM.btnStartCameraHero.addEventListener('click', startCamera);

  // Record Sign
  DOM.btnRecordSign.addEventListener('click', triggerRecordSequence);

  // Keyboard shortcut: Spacebar triggers recording
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.repeat && state.cameraActive && !state.isRecording && !state.isCountingDown) {
      // Don't trigger if user is typing in search or input fields
      if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      e.preventDefault();
      triggerRecordSequence();
    }
  });

  // Mirror view button
  DOM.btnFlipMirror.addEventListener('click', () => {
    state.mirrorCamera = !state.mirrorCamera;
    DOM.chkMirrorCamera.checked = state.mirrorCamera;
    applyMirrorClass();
  });

  // Sentence buttons
  DOM.btnSpeakSentence.addEventListener('click', speakCurrentSentence);
  DOM.btnUndoWord.addEventListener('click', undoLastWord);
  DOM.btnClearSentence.addEventListener('click', clearSentence);
  DOM.btnCopySentence.addEventListener('click', copySentenceToClipboard);

  // Retry button
  DOM.btnRetryPredict.addEventListener('click', () => {
    if (state.lastCapturedFrames) {
      sendPredictionRequest(state.lastCapturedFrames);
    } else {
      triggerRecordSequence();
    }
  });

  // Debug & Golden sample replay
  DOM.sampleSelect.addEventListener('change', (e) => {
    DOM.btnReplaySample.disabled = !e.target.value;
  });
  DOM.btnReplaySample.addEventListener('click', () => {
    replayGoldenSample(DOM.sampleSelect.value);
  });
  DOM.btnExportJson.addEventListener('click', exportCapturedJson);
  DOM.fileInputJson.addEventListener('change', loadCustomJsonFile);

  // Vocabulary search
  DOM.vocabSearch.addEventListener('input', (e) => {
    filterVocabulary(e.target.value);
  });

  // Settings Panel
  DOM.btnToggleSettings.addEventListener('click', () => {
    DOM.settingsPanel.classList.toggle('hidden');
  });
  DOM.btnCloseSettings.addEventListener('click', () => {
    DOM.settingsPanel.classList.add('hidden');
  });
  DOM.btnSaveApiUrl.addEventListener('click', () => {
    const val = DOM.apiBaseUrlInput.value.trim();
    if (val) {
      state.apiBaseUrl = val;
      localStorage.setItem('isharalink_api_url', val);
      checkBackendHealth();
      fetchSupportedClasses();
      DOM.settingsPanel.classList.add('hidden');
    }
  });
  DOM.chkMirrorCamera.addEventListener('change', (e) => {
    state.mirrorCamera = e.target.checked;
    applyMirrorClass();
  });
  DOM.chkSwapHands.addEventListener('change', (e) => {
    state.swapHands = e.target.checked;
  });
  DOM.chkAutoSpeak.addEventListener('change', (e) => {
    state.autoSpeak = e.target.checked;
  });

  // Research Gallery Tabs
  DOM.galleryTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      DOM.galleryTabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      const imgSrc = tab.getAttribute('data-img');
      if (imgSrc) {
        DOM.activeGalleryImg.src = imgSrc;
      }
    });
  });

  DOM.activeGalleryImg.addEventListener('error', () => {
    const current = DOM.activeGalleryImg.getAttribute('src') || '';
    if (current.startsWith('../')) {
      DOM.activeGalleryImg.src = current.replace(/^\.\.\//, '');
    } else if (!current.startsWith('http') && !current.startsWith('/')) {
      DOM.activeGalleryImg.src = '../' + current;
    }
  });



  // Periodic health check every 15s
  setInterval(checkBackendHealth, 15000);
}

// --- Start App -------------------------------------------------------------
window.addEventListener('DOMContentLoaded', initializeApp);
