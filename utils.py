#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     utils.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description:
-------------------------------------------------
"""
from ultralytics import YOLO
import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import tempfile
import tensorflow as tf



def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: None
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )



def load_detection_and_classification_model(detection_model_path, classification_model_path):
    """
    Loads YOLO object detection and image classification models.

    Parameters:
        detection_model_path (str): The path to the YOLO model file for object detection.
        classification_model_path (str): The path to the image classification model file.

    Returns:
        A tuple containing the loaded YOLO object detection model and the image classification model.
    """
    detection_model = YOLO(detection_model_path)
    classification_model = tf.keras.models.load_model(classification_model_path)
    return detection_model, classification_model



def infer_image_with_detection_and_classification(conf, detection_model, classification_model):
    """
    Execute inference for uploaded image with object detection and image classification.

    :param conf: Confidence of YOLOv8 model
    :param detection_model: An instance of the `YOLOv8` class containing the YOLOv8 model for object detection.
    :param classification_model: The image classification model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp'),
        key="infer_uploaded_image"
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img is not None:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=uploaded_image,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img is not None:
        if st.button("Execution"):
            process_image_with_detection_and_classification(conf, detection_model, classification_model, uploaded_image, col2)


def process_image_with_detection_and_classification(conf, detection_model, classification_model, uploaded_image, col2):

    with st.spinner("Running..."):
        # Perform object detection
        detection_result = detection_model.predict(uploaded_image, conf=conf)
        boxes = detection_result[0].boxes
        print(boxes)
        detection_image = detection_result[0].plot()[:, :, ::-1]

        # Perform image classification
        img = uploaded_image.resize((224, 224))  # Resize the image for classification
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Preprocess the image
        img_array = np.repeat(np.expand_dims(img_array, axis=-1), 3, axis=-1)  # Add channel dimension and repeat for RGB

        classification_result = classification_model.predict(img_array)
        classification_label = "Benign" if classification_result[0][0] > classification_result[0][1] else "Malignant"
        
        # Display detection result
        with col2:
            st.image(detection_image, caption="Detected Image", use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.xywh)

                with st.expander("Classification Results"):
                    if classification_result[0][0] > classification_result[0][1]:
                        st.write("Prediction: Benign")
                    else:
                        st.write("Prediction: Malignant")

            except Exception as ex:
                st.write("No image is uploaded yet!")
                st.write(ex)


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

