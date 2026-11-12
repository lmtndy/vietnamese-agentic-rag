# Ghi chú (tiếng Việt) — Agentic RAG cho tài liệu tiếng Việt

Chatbot hỏi–đáp trên tài liệu, **có trích dẫn nguồn**. Hướng dẫn cài đặt, kết quả đánh giá và cấu trúc thư mục: xem [README.md](README.md).

## Kiến trúc

```
                 OFFLINE (src/ingest.py)
  data/*.txt,*.pdf ─► cắt chunk ─► embedding (multilingual-e5) ─► Qdrant

                 ONLINE (src/rag.py, src/api.py)
  câu hỏi ─► embedding ─► Qdrant top-20 ─► cross-encoder rerank ─► top-4 ─► prompt ─► Gemini ─► trả lời + [Nguồn X]

                 AGENT (src/agent.py, LangGraph)
  route (cần tra tài liệu?) ─► retrieve ─► grade (ngữ cảnh có liên quan?) ─► rewrite câu hỏi + tra lại (giới hạn số lần) ─► generate
```

| Thành phần | Công nghệ | File |
|---|---|---|
| Embedding | `sentence-transformers` (`intfloat/multilingual-e5-small`) | `src/ingest.py`, `src/rag.py` |
| Vector database | Qdrant (Docker) | `src/ingest.py`, `src/rag.py` |
| Rerank | cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` | `src/rag.py` |
| Sinh câu trả lời | Google Gemini | `src/rag.py` |
| API | FastAPI (`POST /ask`, `/health`, Swagger `/docs`) | `src/api.py` |
| Agent | LangGraph `StateGraph` | `src/agent.py` |
| Đánh giá | UIT-ViQuAD 2.0, hit@k / MRR, MLflow | `evaluation/` |

## Từng bước
- **Chunking** (`src/chunking.py`): cắt đoạn ~500 ký tự, chồng lấn 100 ký tự để không đứt ý ở ranh giới.
- **Embedding**: mỗi chunk → vector; câu gần nghĩa ⇒ vector gần nhau. Model e5 cần tiền tố `query: ` / `passage: `.
- **Retrieve**: Qdrant trả 20 chunk gần nhất theo cosine (bi-encoder: nhanh, lọc rộng).
- **Rerank**: cross-encoder đọc từng cặp (câu hỏi, chunk) và chấm lại → giữ 4 chunk tốt nhất (chậm hơn nhưng chính xác hơn).
- **Augment + Generate**: ghép 4 chunk vào prompt, yêu cầu Gemini chỉ trả lời theo ngữ cảnh và trích `[Nguồn X]`.
- **Đánh giá**: trên 300 câu hỏi UIT-ViQuAD 2.0, rerank nâng hit@4 từ 0.813 lên 0.883 (chunk 500).
