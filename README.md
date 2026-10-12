# StreetSpace AI

Interactive spatial-intelligence demo for street-level imagery: scene entities, relative depth, and visual-place similarity.

**Live demo:** add your Streamlit URL here after making the app public.

The interface supports **中文 / English** switching from the sidebar.

## What the MVP does

1. **Scene semantics** — DETR object detection
2. **Relative spatial structure** — Depth Anything Small
3. **Visual-place similarity** — CLIP ViT-B/32 retrieval
4. **Interpretable output** — confidence, image-region descriptions, relative-depth response
5. **Product instrumentation** — inference time, input resolution, downloadable JSON/CSV results

The project is deliberately framed as a **product hypothesis test**, not as a world model. The goal is to test how far mature perception models can go in providing a useful first layer of spatial understanding before adding heavier multi-view / 3D / world-model components.

## Product logic

**problem definition → technical option selection → runnable MVP → interpretable output → evaluation plan → next-step roadmap**

This connects a background in Visual Place Recognition (VPR), multimodal learning and model evaluation with technical AI product work in spatial intelligence, robotics, 3D content and embodied AI.

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

The first inference downloads model weights from Hugging Face and can take several minutes.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create an app from the repository.
3. Use `app.py` as the entrypoint.
4. After deployment, open **App settings → Sharing** and choose **This app is public and searchable** if the link is going on a resume.
5. Add the final `*.streamlit.app` URL near the top of this README and to the resume.

If the app redirects a visitor to Streamlit sign-in, it is still private and should not yet be used as a resume link.

## Models

- `facebook/detr-resnet-50` — object detection
- `LiheYoung/depth-anything-small-hf` — monocular relative depth
- `openai/clip-vit-base-patch32` — image embeddings / visual retrieval

The detector and depth model are loaded when spatial analysis is requested. CLIP is loaded only when retrieval is requested. Streamlit resource caching reuses loaded models across reruns within the app process.

## Product choices

- **Relative depth, not metric distance:** the demo avoids claiming real-world scale without calibration.
- **Composable modules:** detection, depth and retrieval can be tested and replaced independently.
- **Interpretable output first:** the user sees entities, confidence and spatial cues instead of only a black-box answer.
- **MVP before 3D:** user value and evaluation criteria are tested before investing in multi-view reconstruction.

## Evaluation plan

- Detection quality on representative street scenes
- Cross-view stability of spatial descriptions
- Recall@K for visual-place retrieval
- End-to-end latency
- Human interpretability of outputs

## Suggested V2

- VLM-generated scene graphs
- Multi-view consistency
- A small labeled street-scene retrieval benchmark
- 3DGS / NeRF reconstruction experiments
- Task-driven spatial evaluation
- Latency / cost comparison across model options

## Data handling

This application code does not save uploaded images to disk or a database. Uploaded files are used for the current interactive session and model inference only.
