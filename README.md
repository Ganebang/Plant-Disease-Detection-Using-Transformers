# Plant Disease Detection Using Transformers 🌿

An end-to-end machine learning pipeline leveraging the **vision Transformer (DETR)** architecture to automatically detect and classify plant diseases from leaf imagery. This repository includes a robust training pipeline optimized for 6GB VRAM GPUs using Automatic Mixed Precision, and a visually rich **Streamlit Dashboard** for real-time inference.

## Features
- **Transformer-based Detection:** Uses the `facebookresearch/detr` (Detection Transformer) architecture fine-tuned on the PlantDoc datasets for robust object detection using Bipartite Matching.
- **Robust Training Pipeline:** Memory-efficient training pipeline structured for limited hardware using PyTorch AMP (Automatic Mixed Precision) and gradient accumulation limits.
- **Interactive Web UI:** A vibrant Streamlit application allowing users to upload leaf images, dynamically tweak confidence thresholds in real-time right from the UI, and view drawn bounding boxes.

## Repository Structure
- `app/main.py`: The Streamlit-based web dashboard and inference script.
- `src/train.py`: Full training lifecycle, loss calculation configuration, and model saving checkpointing.
- `src/engine.py`: Core logic routines for computing forward/backward passes during training, bounding box mAP mapping, and inference evaluations.
- `src/dataset.py`: PyTorch dataset implementations parsing COCO-formatted JSON annotations and handling relative squash coordinates.
- `src/model.py`: Architecture structure containing `PlantDocDETR` definitions.

## Installation & Usage

1. **Environment Setup:** Create a virtual environment and heavily recommended libraries. 
   ```bash
   pip install -r requirements.txt
   ```
2. **Download Weights:** The codebase relies on a populated `weights/` directory. Drop your trained `.pth` DETR checkpoints inside `weights/model_20.pth`.
3. **Launch the Dashboard:** Navigate inside your CLI and run the Streamlit application.
   ```bash
   python -m streamlit run app/main.py
   ```
   Modify the *Confidence Threshold* on the sidebar to adjust how stringent the app's Bipartite classification constraints are acting if the model was trained for limited epochs.
