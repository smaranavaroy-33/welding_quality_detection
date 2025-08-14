import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import torch
import os

st.title("Metal Welding Quality Detection")

# Loading the model 
model_path = os.getcwd() + "/model/yolov12n_ft.pt"
model = YOLO(model_path)

def non_max_suppression(boxes, scores, classes, iou_threshold=0.5):
    """
    Apply non-maximum suppression to eliminate overlapping bounding boxes
    Args:
        boxes: list of boxes in format [x1, y1, x2, y2]
        scores: confidence scores for each box
        classes: class labels for each box
        iou_threshold: threshold for considering boxes as overlapping
    Returns:
        indices of boxes to keep
    """
    # Convert to numpy arrays if they aren't already
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Get the coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate area of each box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by confidence score
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Find the intersection coordinates
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Calculate intersection area
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # Calculate IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU less than threshold
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

#st.markdown("Upload the image to find the defect.")
logo_path = os.getcwd() + "/cts_logo.png"
st.sidebar.image(logo_path, use_container_width=True)
# Use Streamlit's markdown to increase font size of the uploader label
# st.markdown(
#     "<h4 style='font-size: 28px; font-weight: 600;'>Upload image to determine the quality of the welding:</h4>",
#     unsafe_allow_html=True
# )

uploaded_file = st.file_uploader("Upload image to determine the quality of the welding:", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_file:
    for file in uploaded_file:
        image = Image.open(file).convert("RGB")
        
        results = model.predict(source=image, imgsz=640, conf=0.5)
        detected_boxes = results[0].boxes

        if detected_boxes.xyxy is not None and len(detected_boxes.xyxy) > 0:
            # Convert the PIL image to a NumPy array for OpenCV processing
            image_cv = np.array(image)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            
            # Get all boxes, scores and classes
            boxes = detected_boxes.xyxy.cpu().numpy()
            scores = detected_boxes.conf.cpu().numpy()
            classes = detected_boxes.cls.cpu().numpy()
            
            # Apply NMS
            keep_indices = non_max_suppression(boxes, scores, classes, iou_threshold=0.5)
            
            # Draw only the kept boxes
            for idx in keep_indices:
                box = boxes[idx]
                score = scores[idx]
                cls = classes[idx]
                
                # Extract coordinates and convert to integers
                x1, y1, x2, y2 = map(int, box)
                # Get the class name
                class_name = model.names[int(cls)]
                # Create the label string
                label = f"{class_name} {score:.2f}"
                
                # Set color based on class
                if class_name in ["Defect", "Bad Weld"]:
                    box_color = (0, 0, 255)   # Red in BGR
                    bg_color = (0, 0, 255)    # Red background for label
                else:
                    box_color = (0, 255, 0)   # Green in BGR
                    bg_color = (0, 255, 0)    # Green background for label
                
                # Draw the bounding box
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), box_color, 2)
                
                # Text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                text_color = (255, 255, 255)
                
                # Get text size and position
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_margin = 5
                text_y_baseline = y1 - text_margin
                
                if text_y_baseline - text_height < 0:
                    text_y_baseline = y1 + text_height + text_margin
                
                # Draw text background
                rect_padding = 3
                rect_x1 = x1 - rect_padding
                rect_y1 = text_y_baseline - text_height - rect_padding
                rect_x2 = x1 + text_width + rect_padding
                rect_y2 = text_y_baseline + baseline + rect_padding
                
                rect_x1 = max(0, rect_x1)
                rect_y1 = max(0, rect_y1)
                rect_x2 = min(image_cv.shape[1], rect_x2)
                rect_y2 = min(image_cv.shape[0], rect_y2)
                
                cv2.rectangle(image_cv, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
                
                # Draw text
                cv2.putText(image_cv, label, (x1, text_y_baseline),
                           font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
            # Convert back to RGB for Streamlit display
            img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Detected Defects", use_container_width=True)
        else:
            if len(results[0].boxes) > 0:
                st.warning(f"No defects detected above the confidence threshold (currently {results[0].conf.item():.2f}). There might be defects with lower confidence scores.")
            else:
                st.warning("No defects found in the image.")