from flask import Flask, render_template, request, Response
import io
import os
import cv2
import time
import pickle
import sklearn
import numpy as np
from gtts import gTTS
from collections import defaultdict

VOC_CLASSES = ['chair', 'diningtable', 'person', 'car', 'motorbike', 'bottle']

app = Flask(__name__)
cam = cv2.VideoCapture(0)
globaltts = ""

def load_model(modelName):
    if modelName == "ORB":
        with open("./models/classifier_orb.pkl", "rb") as f:
            model = pickle.load(f)
    elif modelName == "SIFT":
        with open("./models/classifier_sift.pkl", "rb") as f:
            model = pickle.load(f)
    return model

def preprocess_image(image_array):
    blurred_image = cv2.GaussianBlur(image_array, (5,5), 0)
    gray_image = cv2.cvtColor((blurred_image * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

    return edges

def recognize_labels_with_probabilities(img, classifier, modelName):
    orb = cv2.ORB_create()
    sift = cv2.SIFT_create()
    if modelName == "ORB":
        keypoints, descriptors = orb.detectAndCompute(img, None)
    elif modelName == "SIFT": 
        keypoints, descriptors = sift.detectAndCompute(img, None)

    if descriptors is not None and len(descriptors) > 0:
        if descriptors.shape[1] < 32:
            descriptors = np.pad(descriptors, ((0, 0), (0, 128 - descriptors.shape[1])), mode='constant')
        elif descriptors.shape[1] > 32:
            descriptors = descriptors[:, :32]
        
        feature_vector = np.mean(descriptors, axis=0).reshape(1, -1)

        try:
            probabilities = classifier.predict_proba(feature_vector)[0]
            return {VOC_CLASSES[i]: probabilities[i] for i in range(len(VOC_CLASSES))}
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {}
    else:
        print("No descriptors detected")
        return {}

def gen_frames(modelName):
    model = load_model(modelName)
    frame_count = 0
    accumulated_probabilities = defaultdict(list)
    curr_top_pred = ""

    while True:
        success, frame = cam.read()
        if not success:
            print("failed getting frames")
            break
        else:
            preproc = preprocess_image(frame)
            label_probabilities = recognize_labels_with_probabilities(preproc, model, modelName)
            for label, prob in label_probabilities.items():
                if prob > 0:
                    accumulated_probabilities[label].append(prob)

            frame_count += 1
            if frame_count >= 100:
                top_predictions = []
                for label in VOC_CLASSES:
                    if accumulated_probabilities[label]:
                        avg_prob = np.mean(accumulated_probabilities[label])
                        top_predictions.append((label, avg_prob))
                        print(f"{label}: {avg_prob:.4f}")
                    else:
                        print(f"{label}: No detections")
                top_predictions = sorted(top_predictions, key=lambda x: x[1], reverse=True)
                label, score = top_predictions[0]
                curr_top_pred = label
                globals()["globaltts"] = label

                frame_count = 0
                accumulated_probabilities = defaultdict(list)
            res = cv2.putText(frame, f"Prediction: {curr_top_pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', res)
            res = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + res + b'\r\n\r\n')

def check_audio():
    while True:
        if globaltts != "":
            tts = gTTS(globaltts, lang='en')
            audio_buffer = io.BytesIO()
            tts.save("./outputs/output.mp3")
            # tts.write_to_fp(audio_buffer)
            # print(f"Audio buffer size: {len(audio_buffer.getvalue())} bytes")
            # audio_buffer.seek(0)
            os.system("start ./outputs/output.mp3")
            yield (b'--frame\r\n'b'Content-Type: audio/mpeg\r\n\r\n' + audio_buffer.read() + b'\r\n\r\n')
        time.sleep(2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_result/<modelName>")
def get_result(modelName):
    return Response(gen_frames(modelName), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_audio")
def get_audio():
    return Response(check_audio(), mimetype='multipart/x-mixed-replace; boundary=frame')