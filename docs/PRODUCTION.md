# Production — Qdrant Cloud và vận hành

## Qdrant Cloud

1. Tạo cluster trên [Qdrant Cloud](https://cloud.qdrant.io/), lấy **URL** và **API key**.
2. Cập nhật `QDRANT_URL`, `QDRANT_API_KEY` trên host chạy **ecommerce-Embeddings** (không commit vào git).
3. Đảm bảo **cùng** `QDRANT_COLLECTION` và **cùng model embedding** (`EMBEDDING_BACKEND` + model / ProtonX) với môi trường đã index — nếu đổi model, **reindex toàn bộ** (`full_reset: true` một lần).

## Secrets

- `APP_ENV=production` — ép service chạy theo profile production.
- `EMBEDDING_BACKEND=protonx` (hoặc để trống để service tự resolve theo `APP_ENV`).
- `EMBEDDINGS_REINDEX_SECRET` — khớp Commerce-Api `EMBEDDINGS_REINDEX_SECRET` và header `X-Reindex-Key`.
- `PROTONX_API_KEY` — nếu dùng backend `protonx`.
- `N8N_AI_CHAT_WEBHOOK_URL` — URL production của n8n (HTTPS).
- `AI_CHAT_INTERNAL_SECRET` — chung giữa Commerce-Api và n8n (header `X-Internal-Key`).

## Backup

- Qdrant Cloud: bật snapshot theo tài liệu nhà cung cấp.
- Export workflow n8n định kỳ (file JSON trong repo hoặc registry).

## Giám sát

- Health: `GET http(s)://embeddings-host/health`, Commerce-Api `GET /V1/health`.
- Log lỗi webhook n8n và timeout LLM (Ollama chỉ dùng on-prem / VPN, không expose public).
