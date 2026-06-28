# xiaohuangya

个人项目仓库。当前包含一个私人 RAG 后端，以及本地大模型推理环境的现状记录。

## 目录结构

```
xiaohuangya/
├── sihan-live/                  # 「仅本人可用」的私人 RAG 对话后端（FastAPI）
│   ├── backend/                 #   应用代码：config / auth / memory / llm_engine / main
│   ├── data/kb_seed/            #   示例知识库
│   ├── tests/                   #   pytest 用例
│   ├── config.yaml              #   运行配置（app / knowledge_base / llm）
│   └── README.md                #   子项目说明与启动方式
├── remote_model_inventory.txt   # 本地推理服务器上的大模型清单（快照）
├── remote_service_status.txt    # 本地推理服务进程/端口现状（快照）
├── .github/agents/              # GitHub 自定义 agent 模板（待填写）
└── LICENSE
```

## 子项目：sihan-live
一个轻量的检索增强对话（RAG）后端：导入文本知识库 → TF-IDF 检索 → 以人设风格回复，
可选接入本地 `llama-server`（OpenAI 兼容）。详见 [`sihan-live/README.md`](sihan-live/README.md)。

```bash
cd sihan-live
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8080
pytest -q
```

## 关于 `remote_*.txt`
这两份是本地推理环境（`llama-server` + `open-webui` + `LLaMA-Factory`，Qwen3.6-35B）的
现状快照，作为部署参考留存，并非由本仓库代码生成。
