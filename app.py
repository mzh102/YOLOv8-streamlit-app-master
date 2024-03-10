#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     app.py
   @Author:        Luyao.zhang
   @Date:          2023/5/15
   @Description:
-------------------------------------------------
"""
from pathlib import Path
import streamlit as st
import config
from utils import load_detection_and_classification_model,infer_image_with_detection_and_classification

# setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Interactive Interface for YOLOv8")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection", "Classification"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
elif task_type == "Classification":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.CLASSIFICATION_MODEL_LIST
    )

else:
    st.error("Currently only 'Detection' and 'Classification' function is implemented")



model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model,model_classification = load_detection_and_classification_model(model_path,"D:\\ChromeCoreDownloads\\YOLOv8-streamlit-app-master\\YOLOv8-streamlit-app-master\\yolov8-streamlit-app-master\\weights\\classification\\mo2.h5")


except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")


# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)


source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    if task_type == "Detection":
        confidence = float(st.sidebar.slider("Select Model Confidence", 0, 100, 50)) / 100
        infer_image_with_detection_and_classification(confidence, model,model_classification)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    if task_type == "Detection":
        confidence = float(st.sidebar.slider("Select Model Confidence", 0, 100, 50)) / 100
        infer_image_with_detection_and_classification(confidence, model,model_classification)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    if task_type == "Detection":
        confidence = float(st.sidebar.slider("Select Model Confidence", 0, 100, 50)) / 100
        infer_image_with_detection_and_classification(confidence, model,model_classification)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")

