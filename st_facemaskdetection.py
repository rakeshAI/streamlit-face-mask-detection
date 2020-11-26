import os
# CPUで動作するための設定
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tensorflowのINFOを非表示にするための設定
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import streamlit as st
from PIL import Image
st.set_page_config(layout="wide")


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7]*np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype('int')
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)

print("[INFO] loading face detector model")
prototxt = 'face_detector/deploy.prototxt'
model = 'face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
confidence_limit = 0.5
faceNet = cv2.dnn.readNetFromCaffe(prototxt, model)
print("[INFO] loading mask detector model")
maskNet = load_model("mask_detector.model")






st.title("Face mask detection")

uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
if uploaded_image is not None:
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
else :
    image = cv2.imread('image61.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# detect faces in the frame and determine if they are wearing a face mask or not
frame = image.copy()
(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
# loop over the detected face locations and their corresponding locations
for (box, pred) in zip(locs, preds):
    # unpack the bounding box and predictions
    (startX, startY, endX, endY) = box
    (mask, withoutMask) = pred

    # determine the class label and color we'll use to draw
    # the bounding box and text
    label = "Mask" if mask > withoutMask else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

    # include the probability in the label
    label = "{}:{:.2f}%".format(label, max(mask, withoutMask) * 100)

    # display the label and bounding box rectangle on the output frame
    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

col1, col2= st.beta_columns(2)
col1.header('Original image')
col1.image(image, width=480)
col2.header('Mask detection image')
col2.image(frame, width=480)
