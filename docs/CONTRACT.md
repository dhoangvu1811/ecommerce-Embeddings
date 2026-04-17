# Contract JSON — Chat RAG

## 1) clientEC → Commerce-Api

`POST /V1/ai-chat`

```json
{
  "message": "string (required)",
  "conversationId": "string (optional)",
  "productId": 123,
  "locale": "vi"
}
```

Response **200**:

```json
{
  "code": 200,
  "message": "OK",
  "data": {
    "reply": "string",
    "sources": [
      {
        "productId": 1,
        "title": "string",
        "url": "string",
        "score": 0.85
      }
    ]
  }
}
```

Lỗi chuẩn dự án: `{ "code": number, "message": string, "data": null }`.

---

## 2) Commerce-Api → n8n (webhook)

Body gửi tới `N8N_AI_CHAT_WEBHOOK_URL` (POST):

```json
{
  "conversationId": "uuid-or-empty",
  "message": "user text",
  "productId": null,
  "locale": "vi",
  "internalKey": "same as N8N_INTERNAL_KEY if set"
}
```

Header tùy chọn: `X-Internal-Key` (trùng `AI_CHAT_INTERNAL_SECRET`).

Response mong đợi từ n8n (JSON):

```json
{
  "reply": "assistant text",
  "sources": [
    { "productId": 1, "title": "Tên SP", "url": "https://...", "score": 0.82 }
  ]
}
```

---

## 3) n8n → ecommerce-Embeddings

### Embed (OpenAI-like batch)

`POST /v1/embed`

```json
{ "text": "một câu" }
```

hoặc:

```json
{ "inputs": ["a", "b"] }
```

Response:

```json
{
  "embeddings": [[0.1, ...]],
  "data": [{ "embedding": [...], "index": 0 }],
  "dimensions": 384
}
```

### Search kết hợp (tùy workflow)

`POST /v1/search`

```json
{
  "text": "câu hỏi",
  "limit": 8,
  "score_threshold": 0.25,
  "productId": null
}
```

Response:

```json
{
  "code": 200,
  "data": {
    "hits": [
      {
        "id": "...",
        "score": 0.55,
        "payload": {
          "product_id": 1,
          "name": "...",
          "url": "...",
          "text": "chunk used for RAG"
        }
      }
    ]
  }
}
```

### Reindex (nội bộ, không public)

`POST /v1/index/reindex` + `X-Reindex-Key` (khi `EMBEDDINGS_REINDEX_SECRET` bật)

```json
{ "productId": 12, "full_reset": false }
```

---

## 4) Biến env liên quan (tên gợi ý)

| Nơi | Biến |
|-----|------|
| Commerce-Api | `N8N_AI_CHAT_WEBHOOK_URL`, `AI_CHAT_INTERNAL_SECRET`, `EMBEDDINGS_SERVICE_URL`, `EMBEDDINGS_REINDEX_*` |
| n8n (Credentials / env) | URL embeddings, Qdrant (nếu tách bước), Ollama base `http://host.docker.internal:11434/v1` |
