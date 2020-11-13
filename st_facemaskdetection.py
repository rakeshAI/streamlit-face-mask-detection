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
st.markdown("# Face mask detection")



cap = cv2.VideoCapture(0)
col1, col2 = st.beta_columns(2)
col1 = col1.empty()
col2 = col2.empty()
#image_loc = st.empty()

while cap.isOpened:
    ret, img = cap.read()
    time.sleep(0.01)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #image_loc.image(img)
    col1.image(img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()