## 简历项目写法（建议放在 ECG 项目前）

**StreetSpace 街景空间智能 Demo｜独立产品设计与开发**  
2026.10

- 面向真实街景空间理解场景，独立完成从问题定义、功能拆解、技术选型到可运行 MVP 的设计与实现。
- 组合 DETR、Depth Anything 与 CLIP，实现街景实体识别、图像空间位置描述、相对深度结构可视化及相似场景检索，将 VPR 研究经验转化为可交互产品 Demo。
- 将深度结果限定为“相对响应”而非真实距离，并在产品中显式呈现模型置信度与能力边界，避免对 3D 理解能力的过度承诺。
- 设计后续评估与演进路线，包括 Recall@K、跨视角一致性、响应延迟、多视角输入及 3DGS/NeRF 等方向，用实验结果驱动产品能力迭代。

**技术栈：** Python、Streamlit、PyTorch、Transformers、DETR、Depth Anything、CLIP
