from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="StreetSpace AI | Spatial Intelligence Demo",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}
.hero {padding: 1.6rem 1.8rem; border: 1px solid rgba(49,51,63,.15); border-radius: 18px; margin-bottom: 1.2rem;}
.hero h1 {margin: 0 0 .35rem 0; font-size: 2.15rem;}
.hero p {margin: 0; color: rgba(49,51,63,.75); font-size: 1.03rem;}
.kicker {font-size: .83rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .65;}
.metric-card {border:1px solid rgba(49,51,63,.12); border-radius:14px; padding: 1rem 1.05rem; min-height: 120px;}
.small {font-size:.88rem; opacity:.74;}
.tag {display:inline-block; padding:.2rem .55rem; border:1px solid rgba(49,51,63,.15); border-radius:999px; margin:.15rem .25rem .15rem 0; font-size:.82rem;}
.flow {border:1px solid rgba(49,51,63,.12); border-radius:14px; padding:1rem; text-align:center; min-height:100px;}
.footer-note {font-size:.82rem; opacity:.65; text-align:center; margin-top:2rem;}
hr {margin: 1.7rem 0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

TEXT = {
    "en": {
        "language": "Language / 语言",
        "kicker": "Spatial Intelligence Product Demo",
        "hero": "From a single street-level image to scene semantics, relative spatial structure, and visual-place similarity.",
        "controls": "Demo controls",
        "threshold": "Detection confidence",
        "first_run": "The first run downloads open-source model weights. Later runs reuse the cached models.",
        "models": "Models",
        "model_detr": "DETR — object detection",
        "model_depth": "Depth Anything Small — relative depth",
        "model_clip": "CLIP ViT-B/32 — visual similarity",
        "hypothesis": "Product hypothesis",
        "hypothesis_text": "Can a lightweight combination of mature vision models provide a useful first layer of spatial understanding before moving to multi-view 3D or world-model systems?",
        "tab1": "1 · Spatial Understanding",
        "tab2": "2 · Similar Places",
        "tab3": "3 · Product Notes",
        "upload_one": "Upload one street-level image",
        "input_image": "Input image",
        "demo_tests": "What the demo tests",
        "scene_semantics": "Scene semantics",
        "scene_semantics_sub": "Which salient entities are present?",
        "image_structure": "Image-space structure",
        "image_structure_sub": "Where are the entities positioned?",
        "depth_response": "Relative depth response",
        "depth_response_sub": "What coarse depth pattern does a monocular model infer?",
        "analyze": "Analyze space",
        "running": "Running detection and depth estimation…",
        "load_fail": "Model loading/inference failed. This usually means the runtime cannot download Hugging Face weights yet. See README for deployment notes.",
        "outputs": "Model outputs",
        "entities": "Detected scene entities",
        "depth_caption": "Normalized relative-depth response",
        "structured_summary": "Structured scene summary",
        "object": "Object",
        "confidence": "Confidence",
        "relative_depth": "Relative depth score",
        "image_region": "Image region",
        "statements": "Interpretable spatial statements",
        "depth_note": "Depth is intentionally reported as a normalized relative response, not metric distance. The demo is designed to expose model capability and uncertainty rather than overclaim 3D understanding.",
        "visual_similarity": "Visual place similarity",
        "retrieval_intro": "Upload one query image and 2–20 gallery images. CLIP embeddings are used to rank visually similar places. This is a lightweight product-facing bridge from the VPR research background to a usable demo.",
        "query_image": "Query image",
        "gallery_images": "Gallery images",
        "gallery_limit": "Please keep the gallery to 20 images or fewer for this lightweight demo.",
        "query": "Query",
        "find_similar": "Find similar places",
        "encoding": "Encoding images with CLIP…",
        "rank": "Rank",
        "image": "Image",
        "cosine": "Cosine similarity",
        "top_matches": "Top matches",
        "clip_fail": "CLIP loading/inference failed. See README for model download notes.",
        "why": "Why this demo exists",
        "product_body": """This is not presented as a world model. It is a **product hypothesis test** for a narrower question:\n\n> Can mature perception models already create a useful, interpretable first layer of spatial understanding for street-level imagery?\n\nThe demo combines three capabilities that are individually mature but productively complementary:\n\n- **Object detection** → explicit scene entities\n- **Relative depth estimation** → coarse spatial structure\n- **Visual retrieval** → place similarity / representation matching\n\nThe key product decision is to expose **what the system knows, how it knows it, and where it is uncertain**, rather than returning a single opaque answer.""",
        "current_mvp": "Current MVP",
        "current_mvp_items": "- Single-frame input\n- Entity detection\n- Relative depth\n- Similar-place retrieval\n- Interpretable output",
        "limitations": "Known limitations",
        "limitations_items": "- No metric 3D geometry\n- No temporal memory\n- No multi-view consistency\n- Open-vocabulary coverage is limited\n- Retrieval is visual, not geospatial",
        "next_exp": "Next experiments",
        "next_exp_items": "- Multi-view input\n- VLM scene graph generation\n- 3DGS / NeRF reconstruction\n- Temporal consistency\n- Task-driven spatial evaluation",
        "eval_plan": "Evaluation plan",
        "eval_body": """1. **Perception quality** — detection precision on representative street scenes.\n2. **Spatial consistency** — whether image-region and relative-depth statements remain stable across viewpoint changes.\n3. **Retrieval usefulness** — Recall@K on a small labeled gallery.\n4. **Latency** — end-to-end response time on CPU/GPU.\n5. **Human interpretability** — whether users can understand why a result was produced.""",
        "positioning": "Product positioning",
        "tags": ["Spatial Intelligence", "Computer Vision", "VPR", "Multimodal-ready", "AI Product Demo", "Explainable Output"],
        "runtime": "Inference time",
        "object_count": "Detected objects",
        "input_resolution": "Input resolution",
        "download_report": "Download structured result (JSON)",
        "download_ranking": "Download retrieval ranking (CSV)",
        "architecture": "MVP architecture",
        "arch_input": "Street image",
        "arch_perception": "DETR\nScene entities",
        "arch_depth": "Depth Anything\nRelative structure",
        "arch_retrieval": "CLIP\nPlace similarity",
        "arch_output": "Interpretable output",
        "product_decisions": "Key product decisions",
        "product_decisions_body": "- **Composable over monolithic:** separate capabilities make failures easier to diagnose and evaluate.\n- **Relative depth, not metric distance:** avoid claiming geometric accuracy that the model cannot guarantee.\n- **Interpretable output first:** expose entities, confidence and spatial cues before adding heavier world-model components.\n- **MVP before 3D:** validate user value and evaluation criteria before investing in multi-view reconstruction.",
        "privacy": "Data handling",
        "privacy_body": "This demo code does not save uploaded images to disk or a database. Files are used only for the current interactive session and model inference.",
        "portfolio": "Portfolio demo by Ronggui Sun · MSc Computer Science, University of Auckland",
        "image_too_large": "The image was resized for inference to keep the demo responsive.",
        "none_detected": "No high-confidence objects were detected. Try lowering the threshold or using a clearer street-level image.",
        "detected": "Detected {n} salient objects. Dominant scene elements: {dominant}.",
        "depth_unavailable": "depth unavailable",
        "depth_high": "high relative-depth response",
        "depth_mid": "mid relative-depth response",
        "depth_low": "low relative-depth response",
        "relation": "{label}: {hpos}-{vpos} region, {depth_text}, detector confidence {score:.2f}",
        "left": "left", "center": "center", "right": "right",
        "upper": "upper", "middle": "middle", "lower": "lower",
    },
    "zh": {
        "language": "语言 / Language",
        "kicker": "空间智能产品 Demo",
        "hero": "从一张街景图像出发，提取场景语义、相对空间结构与视觉地点相似性。",
        "controls": "演示控制",
        "threshold": "目标检测置信度",
        "first_run": "首次运行会下载开源模型权重，后续运行将直接复用缓存模型。",
        "models": "模型",
        "model_detr": "DETR — 目标检测",
        "model_depth": "Depth Anything Small — 相对深度估计",
        "model_clip": "CLIP ViT-B/32 — 视觉相似度",
        "hypothesis": "产品假设",
        "hypothesis_text": "在进入多视角 3D 或世界模型系统之前，能否先通过成熟视觉模型的轻量组合，为街景提供一层有用、可解释的空间理解能力？",
        "tab1": "1 · 空间理解",
        "tab2": "2 · 相似地点",
        "tab3": "3 · 产品说明",
        "upload_one": "上传一张街景图片",
        "input_image": "输入图片",
        "demo_tests": "这个 Demo 在验证什么",
        "scene_semantics": "场景语义",
        "scene_semantics_sub": "画面中有哪些关键实体？",
        "image_structure": "图像空间结构",
        "image_structure_sub": "这些实体分别位于画面的什么位置？",
        "depth_response": "相对深度响应",
        "depth_response_sub": "单目深度模型推断出了怎样的粗粒度空间层次？",
        "analyze": "分析空间",
        "running": "正在执行目标检测与深度估计……",
        "load_fail": "模型加载或推理失败。通常是因为当前环境暂时无法下载 Hugging Face 模型权重，请查看 README 中的部署说明。",
        "outputs": "模型输出",
        "entities": "检测到的场景实体",
        "depth_caption": "归一化相对深度响应",
        "structured_summary": "结构化场景摘要",
        "object": "目标",
        "confidence": "置信度",
        "relative_depth": "相对深度得分",
        "image_region": "图像区域",
        "statements": "可解释的空间描述",
        "depth_note": "这里展示的是归一化后的相对深度响应，而不是实际米制距离。Demo 的目标是展示模型能力及不确定性，而不是夸大为完整的 3D 理解。",
        "visual_similarity": "视觉地点相似度",
        "retrieval_intro": "上传 1 张查询图片和 2–20 张图库图片。系统使用 CLIP 图像特征对视觉相似地点进行排序，用一个轻量、可交互的产品形式连接 VPR 研究背景与实际 Demo。",
        "query_image": "查询图片",
        "gallery_images": "图库图片",
        "gallery_limit": "为了保持 Demo 轻量，请将图库控制在 20 张图片以内。",
        "query": "查询图",
        "find_similar": "查找相似地点",
        "encoding": "正在使用 CLIP 编码图片……",
        "rank": "排名",
        "image": "图片",
        "cosine": "余弦相似度",
        "top_matches": "最相似结果",
        "clip_fail": "CLIP 加载或推理失败，请查看 README 中关于模型下载的说明。",
        "why": "为什么做这个 Demo",
        "product_body": """这个项目**不把自己包装成世界模型**，而是验证一个更窄、也更可落地的产品假设：\n\n> 成熟的视觉感知模型，是否已经能够为街景图像提供一层有用、可解释的空间理解能力？\n\nDemo 将三种已经较成熟、但可以互补的能力组合起来：\n\n- **目标检测** → 显式场景实体\n- **相对深度估计** → 粗粒度空间结构\n- **视觉检索** → 地点相似性 / 表征匹配\n\n核心产品选择不是只返回一个黑盒答案，而是尽量让用户看到：**系统知道什么、依据是什么、以及它在哪些地方存在不确定性。**""",
        "current_mvp": "当前 MVP",
        "current_mvp_items": "- 单帧图像输入\n- 场景实体检测\n- 相对深度估计\n- 相似地点检索\n- 可解释输出",
        "limitations": "已知局限",
        "limitations_items": "- 不提供真实尺度 3D 几何\n- 没有时序记忆\n- 没有多视角一致性\n- 目标类别覆盖有限\n- 检索依据是视觉相似性而非真实地理位置",
        "next_exp": "下一步实验",
        "next_exp_items": "- 多视角输入\n- VLM 场景图生成\n- 3DGS / NeRF 重建\n- 时序一致性\n- 面向具体任务的空间能力评估",
        "eval_plan": "评估方案",
        "eval_body": """1. **感知质量** —— 在代表性街景上的目标检测准确性。\n2. **空间一致性** —— 视角变化后，图像区域与相对深度描述是否保持稳定。\n3. **检索有效性** —— 在小规模标注图库上计算 Recall@K。\n4. **响应延迟** —— CPU / GPU 上的端到端响应时间。\n5. **可解释性** —— 用户能否理解系统为什么产生当前结果。""",
        "positioning": "产品定位",
        "tags": ["空间智能", "计算机视觉", "VPR", "多模态", "AI 产品 Demo", "可解释输出"],
        "runtime": "推理耗时",
        "object_count": "检测目标数",
        "input_resolution": "输入分辨率",
        "download_report": "下载结构化结果（JSON）",
        "download_ranking": "下载检索排序（CSV）",
        "architecture": "MVP 架构",
        "arch_input": "街景图像",
        "arch_perception": "DETR\n场景实体",
        "arch_depth": "Depth Anything\n相对空间结构",
        "arch_retrieval": "CLIP\n地点相似度",
        "arch_output": "可解释输出",
        "product_decisions": "关键产品取舍",
        "product_decisions_body": "- **模块组合优先于黑盒一体化：** 将能力拆开后，更容易定位失败原因并建立评价指标。\n- **只表达相对深度，不宣称真实距离：** 避免对模型几何能力作超出证据的承诺。\n- **先做可解释输出：** 在引入更重的世界模型能力前，先展示实体、置信度和空间线索。\n- **先验证 MVP，再进入 3D：** 先确认用户价值与评估方式，再投入多视角重建等更高成本方案。",
        "privacy": "数据处理说明",
        "privacy_body": "本 Demo 代码不会将上传图片写入磁盘或数据库；文件仅用于当前交互会话中的模型推理。",
        "portfolio": "作品 Demo · 孙嵘贵｜奥克兰大学 MSc Computer Science",
        "image_too_large": "为保证在线 Demo 响应速度，图片已自动缩放后再进行推理。",
        "none_detected": "未检测到高置信度目标。可以尝试降低置信度阈值，或使用更清晰的街景图片。",
        "detected": "检测到 {n} 个主要目标。场景中的主要元素：{dominant}。",
        "depth_unavailable": "无可用深度信息",
        "depth_high": "较高相对深度响应",
        "depth_mid": "中等相对深度响应",
        "depth_low": "较低相对深度响应",
        "relation": "{label}：位于{hpos}-{vpos}区域，{depth_text}，检测置信度 {score:.2f}",
        "left": "左侧", "center": "中部", "right": "右侧",
        "upper": "上部", "middle": "中部", "lower": "下部",
    },
}

COCO_ZH = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车", "airplane": "飞机",
    "bus": "公交车", "train": "火车", "truck": "卡车", "boat": "船", "traffic light": "交通信号灯",
    "fire hydrant": "消防栓", "stop sign": "停车标志", "parking meter": "停车计时器", "bench": "长椅",
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊", "cow": "牛",
    "backpack": "背包", "umbrella": "雨伞", "handbag": "手提包", "tie": "领带", "suitcase": "行李箱",
    "bottle": "瓶子", "chair": "椅子", "potted plant": "盆栽", "dining table": "餐桌", "tv": "电视",
    "laptop": "笔记本电脑", "cell phone": "手机", "book": "书", "clock": "时钟",
}


def t(lang: str, key: str):
    return TEXT[lang][key]


def localized_label(label: str, lang: str) -> str:
    return COCO_ZH.get(label, label) if lang == "zh" else label


def open_uploaded(uploaded_file, max_edge: int = 1600) -> tuple[Image.Image, bool]:
    """Open an uploaded image safely and downscale very large inputs for demo inference."""
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    resized = max(image.size) > max_edge
    if resized:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return image, resized


def upload_fingerprint(uploaded_file) -> str:
    return hashlib.sha1(uploaded_file.getvalue()).hexdigest()


@dataclass
class DetectedObject:
    label: str
    score: float
    box: Tuple[int, int, int, int]
    relative_depth: float | None = None


@st.cache_resource(show_spinner=False)
def load_detector():
    from transformers import pipeline
    return pipeline("object-detection", model="facebook/detr-resnet-50")


@st.cache_resource(show_spinner=False)
def load_depth_model():
    from transformers import pipeline
    return pipeline("depth-estimation", model="LiheYoung/depth-anything-small-hf")


@st.cache_resource(show_spinner=False)
def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor, torch


def draw_boxes(image: Image.Image, objects: List[DetectedObject], lang: str) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for obj in objects:
        x1, y1, x2, y2 = obj.box
        draw.rectangle((x1, y1, x2, y2), width=3)
        # Keep overlay labels in the detector's English vocabulary so the demo does not
        # depend on a CJK font being installed in the deployment runtime.
        text = f"{obj.label} {obj.score:.2f}"
        tw = max(76, len(text) * 7)
        draw.rectangle((x1, max(0, y1 - 20), x1 + tw, y1), fill="black")
        draw.text((x1 + 4, max(0, y1 - 18)), text, fill="white")
    return canvas


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    p2, p98 = np.percentile(depth, (2, 98))
    if p98 <= p2:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - p2) / (p98 - p2), 0, 1)


def depth_to_display(depth_norm: np.ndarray) -> Image.Image:
    arr = (depth_norm * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def box_position(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[str, str]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if cx < width / 3:
        horizontal = "left"
    elif cx > width * 2 / 3:
        horizontal = "right"
    else:
        horizontal = "center"

    if cy < height / 3:
        vertical = "upper"
    elif cy > height * 2 / 3:
        vertical = "lower"
    else:
        vertical = "middle"
    return horizontal, vertical


def object_depth_score(depth_norm: np.ndarray, box: Tuple[int, int, int, int]) -> float | None:
    h, w = depth_norm.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    region = depth_norm[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return float(np.median(region))


def summarize_scene(objects: List[DetectedObject], image_size: Tuple[int, int], lang: str) -> Tuple[str, List[str]]:
    width, height = image_size
    if not objects:
        return t(lang, "none_detected"), []

    labels = [localized_label(o.label, lang) for o in objects]
    counts = pd.Series(labels).value_counts().to_dict()
    dominant = "、".join(f"{k} ×{v}" if v > 1 else k for k, v in list(counts.items())[:6]) if lang == "zh" else ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in list(counts.items())[:6])
    summary = t(lang, "detected").format(n=len(objects), dominant=dominant)

    relations = []
    for obj in objects[:8]:
        hpos, vpos = box_position(obj.box, width, height)
        if obj.relative_depth is None:
            depth_text = t(lang, "depth_unavailable")
        elif obj.relative_depth >= 0.66:
            depth_text = t(lang, "depth_high")
        elif obj.relative_depth <= 0.33:
            depth_text = t(lang, "depth_low")
        else:
            depth_text = t(lang, "depth_mid")
        relations.append(
            t(lang, "relation").format(
                label=localized_label(obj.label, lang),
                hpos=t(lang, hpos),
                vpos=t(lang, vpos),
                depth_text=depth_text,
                score=obj.score,
            )
        )
    return summary, relations


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T


def clip_embeddings(images: List[Image.Image]) -> np.ndarray:
    model, processor, torch = load_clip()
    batch = processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_image_features(**batch)
    return features.cpu().numpy()


def analyze_image(image: Image.Image, threshold: float):
    detector = load_detector()
    depth_pipe = load_depth_model()

    raw_det = detector(image)
    objects: List[DetectedObject] = []
    for item in raw_det:
        if float(item["score"]) < threshold:
            continue
        b = item["box"]
        box = (int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"]))
        objects.append(DetectedObject(str(item["label"]), float(item["score"]), box))

    depth_result = depth_pipe(image)
    depth = depth_result.get("predicted_depth")
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    depth_norm = normalize_depth(depth)

    depth_img = Image.fromarray((depth_norm * 255).astype(np.uint8)).resize(image.size)
    depth_resized = np.asarray(depth_img).astype(np.float32) / 255.0
    for obj in objects:
        obj.relative_depth = object_depth_score(depth_resized, obj.box)

    return objects, depth_resized


# Language is selected before rendering the rest of the page so every UI block can update immediately.
with st.sidebar:
    language_name = st.selectbox("Language / 语言", ["中文", "English"], index=0)
lang = "zh" if language_name == "中文" else "en"

st.markdown(
    f"""
<div class="hero">
  <div class="kicker">{t(lang, 'kicker')}</div>
  <h1>StreetSpace AI</h1>
  <p>{t(lang, 'hero')}</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header(t(lang, "controls"))
    threshold = st.slider(t(lang, "threshold"), 0.30, 0.95, 0.70, 0.05)
    st.caption(t(lang, "first_run"))
    st.divider()
    st.markdown(f"**{t(lang, 'models')}**")
    st.markdown(f"- {t(lang, 'model_detr')}")
    st.markdown(f"- {t(lang, 'model_depth')}")
    st.markdown(f"- {t(lang, 'model_clip')}")
    st.divider()
    st.markdown(f"**{t(lang, 'hypothesis')}**")
    st.caption(t(lang, "hypothesis_text"))

input_tab, retrieval_tab, product_tab = st.tabs([
    t(lang, "tab1"),
    t(lang, "tab2"),
    t(lang, "tab3"),
])

with input_tab:
    uploaded = st.file_uploader(
        t(lang, "upload_one"),
        type=["jpg", "jpeg", "png", "webp"],
        key="query_image",
    )

    if uploaded:
        image, was_resized = open_uploaded(uploaded)
        current_image_key = upload_fingerprint(uploaded)
        if was_resized:
            st.caption(t(lang, "image_too_large"))
        c1, c2 = st.columns([1.15, 0.85])
        with c1:
            st.image(image, caption=t(lang, "input_image"), use_container_width=True)
        with c2:
            st.markdown(f"### {t(lang, 'demo_tests')}")
            st.markdown(
                f"""
                <div class="metric-card">
                <b>{t(lang, 'scene_semantics')}</b><br><span class="small">{t(lang, 'scene_semantics_sub')}</span><br><br>
                <b>{t(lang, 'image_structure')}</b><br><span class="small">{t(lang, 'image_structure_sub')}</span><br><br>
                <b>{t(lang, 'depth_response')}</b><br><span class="small">{t(lang, 'depth_response_sub')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button(t(lang, "analyze"), type="primary", use_container_width=True):
            try:
                with st.spinner(t(lang, "running")):
                    started = time.perf_counter()
                    objects, depth_norm = analyze_image(image, threshold)
                    elapsed = time.perf_counter() - started
                st.session_state["analysis"] = (objects, depth_norm, image, elapsed, current_image_key)
            except Exception as exc:
                st.error(t(lang, "load_fail"))
                st.exception(exc)

    if uploaded and "analysis" in st.session_state and st.session_state["analysis"][4] == current_image_key:
        objects, depth_norm, image, elapsed, _ = st.session_state["analysis"]
        st.divider()
        st.subheader(t(lang, "outputs"))
        m1, m2, m3 = st.columns(3)
        m1.metric(t(lang, "object_count"), len(objects))
        m2.metric(t(lang, "runtime"), f"{elapsed:.1f}s")
        m3.metric(t(lang, "input_resolution"), f"{image.size[0]} × {image.size[1]}")
        v1, v2 = st.columns(2)
        with v1:
            st.image(draw_boxes(image, objects, lang), caption=t(lang, "entities"), use_container_width=True)
        with v2:
            st.image(depth_to_display(depth_norm), caption=t(lang, "depth_caption"), use_container_width=True)

        summary, relations = summarize_scene(objects, image.size, lang)
        st.markdown(f"### {t(lang, 'structured_summary')}")
        st.write(summary)
        if relations:
            df = pd.DataFrame(
                [
                    {
                        t(lang, "object"): localized_label(o.label, lang),
                        t(lang, "confidence"): round(o.score, 3),
                        t(lang, "relative_depth"): None if o.relative_depth is None else round(o.relative_depth, 3),
                        t(lang, "image_region"): "-".join(t(lang, p) for p in box_position(o.box, *image.size)),
                    }
                    for o in objects
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown(f"### {t(lang, 'statements')}")
            for line in relations:
                st.markdown(f"- {line}")

        st.info(t(lang, "depth_note"))

        report = {
            "project": "StreetSpace AI",
            "image_size": {"width": image.size[0], "height": image.size[1]},
            "inference_seconds": round(elapsed, 3),
            "summary": summary,
            "objects": [
                {
                    "label": o.label,
                    "localized_label": localized_label(o.label, lang),
                    "confidence": round(o.score, 4),
                    "relative_depth_response": None if o.relative_depth is None else round(o.relative_depth, 4),
                    "box": list(o.box),
                }
                for o in objects
            ],
            "interpretation_note": t(lang, "depth_note"),
        }
        st.download_button(
            t(lang, "download_report"),
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="streetspace_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

with retrieval_tab:
    st.markdown(f"### {t(lang, 'visual_similarity')}")
    st.caption(t(lang, "retrieval_intro"))
    q = st.file_uploader(t(lang, "query_image"), type=["jpg", "jpeg", "png", "webp"], key="retrieval_query")
    gallery_files = st.file_uploader(
        t(lang, "gallery_images"),
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="gallery_images",
    )

    if q and gallery_files:
        if len(gallery_files) > 20:
            st.warning(t(lang, "gallery_limit"))
        else:
            query_img, _ = open_uploaded(q)
            gallery_imgs = [open_uploaded(f)[0] for f in gallery_files]
            names = [f.name for f in gallery_files]
            st.image(query_img, caption=t(lang, "query"), width=360)
            if st.button(t(lang, "find_similar"), use_container_width=True):
                try:
                    with st.spinner(t(lang, "encoding")):
                        started = time.perf_counter()
                        feats = clip_embeddings([query_img] + gallery_imgs)
                        scores = cosine_sim(feats[:1], feats[1:])[0]
                        retrieval_elapsed = time.perf_counter() - started
                    order = np.argsort(-scores)
                    ranked = pd.DataFrame(
                        {
                            t(lang, "rank"): np.arange(1, len(order) + 1),
                            t(lang, "image"): [names[i] for i in order],
                            t(lang, "cosine"): [round(float(scores[i]), 4) for i in order],
                        }
                    )
                    r1, r2 = st.columns(2)
                    r1.metric(t(lang, "runtime"), f"{retrieval_elapsed:.1f}s")
                    r2.metric(t(lang, "gallery_images"), len(gallery_imgs))
                    st.dataframe(ranked, use_container_width=True, hide_index=True)
                    st.download_button(
                        t(lang, "download_ranking"),
                        data=ranked.to_csv(index=False).encode("utf-8-sig"),
                        file_name="streetspace_retrieval.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.markdown(f"### {t(lang, 'top_matches')}")
                    topn = min(3, len(order))
                    cols = st.columns(topn)
                    for col, idx in zip(cols, order[:topn]):
                        with col:
                            st.image(gallery_imgs[idx], caption=f"{names[idx]} · {scores[idx]:.3f}", use_container_width=True)
                except Exception as exc:
                    st.error(t(lang, "clip_fail"))
                    st.exception(exc)

with product_tab:
    st.markdown(f"## {t(lang, 'why')}")
    st.markdown(t(lang, "product_body"))

    st.markdown(f"### {t(lang, 'architecture')}")
    a1, a2, a3, a4, a5 = st.columns(5)
    for col, label in zip(
        [a1, a2, a3, a4, a5],
        [t(lang, "arch_input"), t(lang, "arch_perception"), t(lang, "arch_depth"), t(lang, "arch_retrieval"), t(lang, "arch_output")],
    ):
        with col:
            st.markdown(f'<div class="flow"><b>{label.replace(chr(10), "<br>")}</b></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"#### {t(lang, 'current_mvp')}")
        st.markdown(t(lang, "current_mvp_items"))
    with c2:
        st.markdown(f"#### {t(lang, 'limitations')}")
        st.markdown(t(lang, "limitations_items"))
    with c3:
        st.markdown(f"#### {t(lang, 'next_exp')}")
        st.markdown(t(lang, "next_exp_items"))

    st.divider()
    st.markdown(f"### {t(lang, 'eval_plan')}")
    st.markdown(t(lang, "eval_body"))

    st.markdown(f"### {t(lang, 'product_decisions')}")
    st.markdown(t(lang, "product_decisions_body"))

    st.markdown(f"### {t(lang, 'privacy')}")
    st.caption(t(lang, "privacy_body"))

    st.markdown(f"### {t(lang, 'positioning')}")
    tags_html = "\n".join(f'<span class="tag">{tag}</span>' for tag in t(lang, "tags"))
    st.markdown(tags_html, unsafe_allow_html=True)
    st.markdown(f'<div class="footer-note">{t(lang, "portfolio")}</div>', unsafe_allow_html=True)
