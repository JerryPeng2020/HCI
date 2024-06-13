// Copyright 2023 The MediaPipe Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";
const demosSection = document.getElementById("demos");
let poseLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoHeight = "480px";
const videoWidth = "640px";
// Before we can use PoseLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `http://127.0.0.1:5500/dist/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        outputSegmentationMasks: true,
        smoothSegmentation: true,
        numPoses: 2

    });
    demosSection.classList.remove("invisible");
};
createPoseLandmarker();


/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
********************************************************************/
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);
// Check if webcam access is supported.
const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}
// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!poseLandmarker) {
        console.log("Wait! poseLandmaker not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "开始检测";
    }
    else {
        webcamRunning = true;
        enableWebcamButton.innerText = "停止检测";
    }
    // getUsermedia parameters.
    const constraints = {
        video: true
    };
    
    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}
let lastVideoTime = -1;


async function predictWebcam() {
    canvasElement.style.height = videoHeight;
    video.style.height = videoHeight;
    canvasElement.style.width = videoWidth;
    video.style.width = videoWidth;

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await poseLandmarker.setOptions({ runningMode: "VIDEO" });
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        poseLandmarker.detectForVideo(video, startTimeMs, async (result) => {
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            
            // Handle WebGL-rendered mask
            if (result.segmentationMasks && result.segmentationMasks.length > 0) {
                const mask = result.segmentationMasks[0];
                if (mask.canvas && mask.canvas.GLctxObject) {
                    const gl = mask.canvas.GLctxObject.GLctx;
                    const width = mask.canvas.width;
                    const height = mask.canvas.height;
                    const pixels = new Uint8Array(width * height * 4);
                    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

                    // Create ImageData and flip it vertically
                    const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);
                    flipImageDataVertically(imageData);

                    // console.log("Image Data after flip:", imageData);  // Debugging output
                    canvasCtx.putImageData(imageData, 0, 0);  // Draw Mask layer on canvas
                }
            }

            // draw pose marker, after render mask enable
            for (const landmark of result.landmarks) {
                drawingUtils.drawLandmarks(landmark, {
                    radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
                });
                drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
            }

            canvasCtx.restore();
        });
    }

    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}

///////////////////////////// Draw Mask  /////////////////////////////////////////////////////////////////
function flipImageDataVertically(imageData) {
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    const newColorMask = colorizeMask(imageData, 0, 255, 0, 128); // Change mask color to red with 50% opacity
    tempCtx.putImageData(newColorMask, 0, 0);
    // tempCtx.putImageData(imageData, 0, 0);

    const scale = -1; // Flip vertical
    tempCtx.scale(1, scale);
    tempCtx.drawImage(tempCanvas, 0, scale * imageData.height);

    return tempCtx.getImageData(0, 0, imageData.width, imageData.height);
}

function colorizeMask(imageData, red, green, blue, alpha) {
    const data = imageData.data; // Get a reference to the data array of ImageData
    for (let i = 0; i < data.length; i += 4) {
        if (data[i + 3] !== 0) { // Check if the pixel is not completely transparent
            data[i] = red;       // Set red channel
            data[i + 1] = green; // Set green channel
            data[i + 2] = blue;  // Set blue channel
            data[i + 3] = alpha; // Set alpha channel for transparency
        }
    }
    return imageData;
}
