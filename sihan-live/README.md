# sihan-live

「仅本人可用」的私人 RAG 后端：导入文本知识库、检索，并以人设风格做检索增强对话（RAG）。
可选接入本地 `llama-server` 等 OpenAI 兼容模型；未接入时自动降级为基于记忆的回复，服务永不硬失败。

## 功能
- `GET  /health`：存活探针（无需鉴权）。
- `POST /memory/ingest`：从本地目录/文件（可映射 NAS）导入文本知识。
- `POST /memory/search`：TF-IDF 检索历史知识（中文用字符 n-gram，无需分词器）。
- `POST /chat`：先检索记忆，再以人设风格回复（接入 LLM 时走真实模型）。
- `GET  /memory/stats` / `GET /memory/chunks`：统计与调试。
- 鉴权：所有业务接口需 Header `X-Owner-Id` + `X-API-Key`。

## 快速启动
1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 启动服务（默认 8080，避开本机 llama-server 的 8000）

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8080
```

3) 导入默认示例知识

```bash
curl -X POST 'http://127.0.0.1:8080/memory/ingest' \
  -H 'X-Owner-Id: nac' -H 'X-API-Key: local-dev-key' \
  -H 'Content-Type: application/json' \
  -d '{"path":"./data/kb_seed","recursive":true}'
```

4) 检索

```bash
curl -X POST 'http://127.0.0.1:8080/memory/search' \
  -H 'X-Owner-Id: nac' -H 'X-API-Key: local-dev-key' \
  -H 'Content-Type: application/json' \
  -d '{"query":"咖啡","top_k":3}'
```

5) 对话

```bash
curl -X POST 'http://127.0.0.1:8080/chat' \
  -H 'X-Owner-Id: nac' -H 'X-API-Key: local-dev-key' \
  -H 'Content-Type: application/json' \
  -d '{"message":"我们第一次见面是什么时候？","top_k":2}'
```

## 接入真实模型（llama-server / OpenAI 兼容）
在 `config.yaml` 的 `llm` 段把 `enabled` 设为 `true`，并填好 `base_url` / `model`：

```yaml
llm:
  enabled: true
  base_url: http://127.0.0.1:8000/v1   # 本机 llama-server
  model: qwen3.6-hauhau
  api_key: null
```

`/chat` 会把检索到的记忆作为系统提示喂给模型；模型不可达时自动回退到本地记忆回复。

## NAS 使用方式
- 将 NAS 路径挂载到本机后，把路径加入 `config.yaml` 的 `knowledge_base.allowed_roots`。
- 再通过 `/memory/ingest` 传入该挂载路径即可增量导入（重复导入幂等，相同内容不会重复入库）。

## 配置说明（`config.yaml`）
- `app.api_key` 为开发默认值，生产请用未入库的 `*.local.yaml` 或环境变量覆盖，切勿提交真实密钥。
- `knowledge_base.allowed_roots`：白名单根目录，`/memory/ingest` 只允许导入其下的路径。
- 运行时生成的 `data/knowledge_store.json` 已被 `.gitignore` 忽略。

## 测试

```bash
pytest -q
```
