# StreetSpace AI 上线前检查

## 1. GitHub

- [ ] `app.py` 已更新
- [ ] `requirements.txt` 在仓库根目录
- [ ] `README.md` 已更新
- [ ] `.streamlit/config.toml` 已提交
- [ ] 仓库中没有个人手机号、私人邮箱、API Key 或 token

## 2. Streamlit Community Cloud

- [ ] App 正常完成部署
- [ ] 打开 App settings → Sharing
- [ ] 用于简历时设为 **This app is public and searchable**
- [ ] 退出 Streamlit 账号或使用无痕窗口打开链接，确认不要求登录

## 3. 功能测试

### 空间理解
- [ ] 上传一张 JPG/PNG/WEBP 街景
- [ ] 点击“分析空间”
- [ ] 能显示目标检测结果
- [ ] 能显示相对深度图
- [ ] 能显示目标数量、推理耗时、输入分辨率
- [ ] 能下载 `streetspace_analysis.json`

### 相似地点
- [ ] 上传 1 张查询图
- [ ] 上传 2–5 张图库图片
- [ ] 点击“查找相似地点”
- [ ] 能显示排序和 Top 3
- [ ] 能下载 `streetspace_retrieval.csv`

### 中英文
- [ ] 中文界面正常
- [ ] English 界面正常
- [ ] 切换语言后已有分析结果仍能正确显示

## 4. 简历投递前

- [ ] README 顶部填入真实 Live Demo URL
- [ ] README / 简历填入 GitHub URL
- [ ] 用手机或无痕浏览器点击一次简历中的 Demo 链接
- [ ] 首次冷启动可接受；如果明显 OOM，再考虑换轻量目标检测模型
