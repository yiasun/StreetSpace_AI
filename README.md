# StreetSpace AI

A lightweight **spatial intelligence product demo** for street-level imagery.

The interface supports **中文 / English switching from the sidebar**, including tabs, controls, product notes, result tables and spatial statements.

The project turns one image into three interpretable capabilities:

1. **Scene semantics** — DETR object detection
2. **Relative spatial structure** — Depth Anything Small
3. **Visual-place similarity** — CLIP ViT-B/32 retrieval

The project is deliberately framed as an **MVP / product hypothesis test**, not as a world model. Its purpose is to demonstrate how mature perception models can be composed into a usable spatial-understanding experience, and to make model limitations visible.

## Why this is portfolio-relevant

The demo connects a background in Visual Place Recognition (VPR), multimodal learning and model evaluation with a product-oriented workflow:

**problem definition → technical option selection → runnable demo → interpretable output → evaluation plan → next-step roadmap**

This is particularly relevant to technical AI product roles in spatial intelligence, robotics, 3D content, world models and embodied AI.

## Features

### Spatial Understanding
Upload one street-level image and obtain:
- detected entities and confidence scores;
- image-region descriptions (left / center / right, upper / middle / lower);
- normalized relative-depth responses;
- structured, interpretable scene statements.

### Similar Places
Upload one query image and a gallery of 2–20 images. CLIP image embeddings rank visually similar places.

### Product Notes
The app explicitly documents:
- the product hypothesis;
- current MVP scope;
- known limitations;
- evaluation metrics;
- roadmap toward multi-view / 3D / world-model capabilities.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first run downloads model weights from Hugging Face, so it requires internet access and may take several minutes.

## Recommended deployment

### Hugging Face Spaces
1. Create a new **Streamlit** Space.
2. Upload `app.py`, `requirements.txt`, and `.streamlit/config.toml`.
3. Use a CPU instance for a basic demo or a GPU Space for faster inference.

### Streamlit Community Cloud
Push this folder to GitHub, then deploy the repository from Streamlit Community Cloud.

## Models

- `facebook/detr-resnet-50` — object detection
- `LiheYoung/depth-anything-small-hf` — monocular relative depth
- `openai/clip-vit-base-patch32` — image embeddings / visual retrieval

All three are loaded lazily so a user only downloads the model needed for the active feature.

## Important interpretation note

The depth map is treated as a **normalized relative model response**, not metric distance. This is intentional: the demo should expose uncertainty rather than claim geometric accuracy it does not have.

## Suggested next iteration

A stronger V2 could add:
- VLM-generated scene graphs;
- multi-view consistency;
- street-scene Recall@K benchmark;
- 3DGS / NeRF reconstruction;
- task-based spatial evaluation;
- latency / cost comparison across model options.
