# Chat_Discussion
多模型方案讨论系统，基于 Python + Streamlit 的轻量应用。支持多模型参与对话式讨论，并由总结模型输出可直接落地的实施方案。

## 特性
- 多模型参与：支持 OpenAI 与通义千问（可扩展），按轮次进行对话式协同
- 本地配置：模型配置持久化到 `providers.json`（name、kind、base_url、api_key、model）
- 侧边栏弹窗管理：新增模型、逐模型编辑与连通性测试
- 实时对话：运行过程中实时展示每轮发言
- 严格总结：输出工程可执行的最终方案（步骤、接口示例、代码片段、风险与验收清单）
- OpenAI 兼容：Qwen 使用 `compatible-mode` 接口，统一为 `messages` 顶层结构

## 快速开始
1. 安装依赖
   - `python -m pip install streamlit`
2. 设置环境变量（按需）
   - `OPENAI_API_KEY`、`DASHSCOPE_API_KEY`
3. 启动应用
   - `python -m streamlit run app.py`
4. 打开浏览器访问
   - `http://localhost:8501/`

## 配置与管理
- 配置文件：[providers.json](file:///c:/Users/杨大宇/Desktop/学科资料/Agent/Chat_Discussion/providers.json)
  - 字段：`name`、`kind`（openai|qwen）、`base_url`、`api_key`、`model`
- 侧边栏“模型管理”
  - 新增：弹窗表单写入 `providers.json`
  - 修改/测试：卡片列表 → 选中模型进入弹窗，保存或连通性测试

## 调用说明
- OpenAI：`/v1/chat/completions`，顶层 `messages`
- 通义千问：`/compatible-mode/v1/chat/completions`，顶层 `messages`
- Base URL 示例
  - OpenAI 官方：`https://api.openai.com`
  - OpenAI 兼容代理：`https://openkey.cloud/v1`（内部路径会自动规范化，避免重复 `/v1`）

## 使用流程
1. 在侧边栏选择参与模型与总结模型，设置轮次
2. 填写问题（主题可留空，自动生成）
3. 运行后实时查看多轮对话与最终方案

## 测试
- 通义千问连通性测试脚本：[test_qwen.py](file:///c:/Users/杨大宇/Desktop/学科资料/Agent/Chat_Discussion/test_qwen.py)
  - 运行：`python test_qwen.py`

## 重要文件
- 应用入口与UI：[app.py](file:///c:/Users/杨大宇/Desktop/学科资料/Agent/Chat_Discussion/app.py)
- 本地配置：[providers.json](file:///c:/Users/杨大宇/Desktop/学科资料/Agent/Chat_Discussion/providers.json)

## 许可
本项目用于学习与演示，可自由修改与扩展。
