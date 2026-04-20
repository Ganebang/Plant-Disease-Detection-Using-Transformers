import os
import sys
# Insert root path immediately so we don't break module resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
import json
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from src.model import PlantDocDETR

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="PlantDoc: Disease Detection", layout="wide")
st.title("🌿 PlantDoc: Transformer-based Crop Diagnosis")
st.markdown("Upload a leaf image to detect diseases using a Detection Transformer.")

# Load class mappings automatically
@st.cache_data
def load_categories():
    try:
        with open(os.path.join("data", "raw", "train", "_annotations.coco.json"), "r") as f:
            data = json.load(f)
            return {item["id"]: item["name"] for item in data.get("categories", [])}
    except Exception:
        return {}

CATEGORY_MAP = load_categories()

# 2. LOAD MODEL (Cached to save memory)
@st.cache_resource
def load_trained_model():
    # Model was dynamically trained on 29 classes based on ID discovery bounds from earlier
    model = PlantDocDETR(num_classes=29) 
    model.load_state_dict(torch.load("weights/model_20.pth", map_location="cpu"))
    model.eval()
    return model

model = load_trained_model()

# 3. IMAGE PREPROCESSING
def transform_image(image):
    transform = T.Compose([
        T.Resize((416, 416)), # MUST match the squash shape used precisely in training
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 4. SIDEBAR - UPLOAD & SETTINGS
st.sidebar.header("Settings & Sourcing")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
uploaded_file = st.sidebar.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Input Image")
        st.image(input_image, use_container_width=True)

    # 5. INFERENCE
    with st.spinner('Transformer is analyzing leaves...'):
        img_tensor = transform_image(input_image)
        with torch.no_grad():
            outputs = model(img_tensor)
        
        # Filter predictions (Confidence Threshold)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > confidence_threshold  # Dynamic threshold from sidebar
        
        # Rescale boxes to original image size
        boxes = outputs['pred_boxes'][0, keep]
        labels = probas[keep].argmax(-1)

    with col2:
        st.header("Model Predictions")
        # Draw boxes on image
        # Since we squashed it to 416x416 initially but we are drawing on the raw image, 
        # the normalized (0-1) coordinates actually scale back perfectly to the native width/height!
        draw = ImageDraw.Draw(input_image)
        w, h = input_image.size
        
        for box, label in zip(boxes, labels):
            # Convert normalized DETR (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
            cx, cy, bw, bh = box.tolist()
            xmin = (cx - bw/2) * w
            ymin = (cy - bh/2) * h
            xmax = (cx + bw/2) * w
            ymax = (cy + bh/2) * h
            
            label_id = label.item()
            class_name = CATEGORY_MAP.get(label_id, f"Class: {label_id}")
            
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=5)
            # Using basic default string rendering since PIL standard fonts are OS-reliant
            draw.text((xmin, ymin - 10), class_name, fill="red")

        st.image(input_image, use_container_width=True)
        st.success(f"Detected {len(labels)} disease areas!")