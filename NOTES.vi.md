# RAG Chatbot tiếng Việt (Project học tập #1)

Chatbot hỏi–đáp trên tài liệu của bạn, **có trích dẫn nguồn**.
Gói gọn 3 kỹ năng Tier 1: **Embedding + Vector Search (RAG)** và nền tảng để gắn **FastAPI / LangGraph** sau này.

## Kiến trúc

```
                 GIAI ĐOẠN OFFLINE (chạy 1 lần, src/ingest.py)
  data/*.txt,*.pdf ─► cắt chunk ─► embedding ─► FAISS index (lưu ra đĩa)

                 GIAI ĐOẠN ONLINE (mỗi câu hỏi, src/chat.py)
  câu hỏi ─► embedding ─► tìm top-k chunk (FAISS) ─► ghép vào prompt ─► Gemini ─► trả lời + nguồn
```

| Thành phần | Công nghệ dùng | File |
|---|---|---|
| Embedding (CPU, đa ngôn ngữ) | `sentence-transformers` (multilingual-e5) | `src/ingest.py`, `src/rag.py` |
| Vector search | `faiss-cpu` | `src/ingest.py`, `src/rag.py` |
| Sinh câu trả lời | Google Gemini (free API) | `src/rag.py` |
| Đọc tài liệu | `pypdf` (+ .txt) | `src/ingest.py` |

## Cài đặt & chạy (làm theo đúng thứ tự)

### Bước 1 — Tạo môi trường ảo & cài thư viện
```bash
cd vietnamese-agentic-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Bước 2 — Lấy Gemini API key (miễn phí)
1. Vào https://aistudio.google.com/apikey → đăng nhập Google → **Create API key**.
2. Tạo file `.env` từ mẫu và dán key vào:
```bash
cp .env.example .env
# mở .env, thay your_api_key_here bằng key thật
```

### Bước 3 — Cho tài liệu vào thư mục `data/`
Đã có sẵn `rag_gioi_thieu.txt` để test. Bạn có thể thả thêm file `.txt` hoặc `.pdf` của mình vào.

### Bước 4 — Build index (đọc + embedding + lưu)
```bash
python -m src.ingest
```
> Lần đầu sẽ tải model embedding (~470MB). Các lần sau dùng cache, rất nhanh.

### Bước 5 — Chat với tài liệu
```bash
python -m src.chat
```
Thử hỏi: *"RAG là gì?"*, *"Chunking để làm gì?"*, *"Vector database gồm những loại nào?"*

## Hiểu từng bước (quan trọng khi phỏng vấn)
- **Chunking** (`ingest.py`): tài liệu dài bị cắt thành đoạn ~500 ký tự, chồng lấn 100 ký tự để không đứt ý.
- **Embedding**: mỗi chunk → vector; câu gần nghĩa ⇒ vector gần nhau.
- **FAISS `IndexFlatIP`**: vector được chuẩn hoá nên inner product = cosine similarity.
- **Retrieve** (`rag.py`): câu hỏi → vector → tìm top-k chunk gần nhất.
- **Augment + Generate**: ghép chunk vào prompt, buộc Gemini chỉ trả lời theo ngữ cảnh và trích `[Nguồn X]`.

## Bài tập mở rộng (lộ trình tiếp theo)
1. **FastAPI** (kỹ năng Tier 1 #2): bọc `RAGEngine.answer()` thành endpoint `POST /ask` → có Swagger UI.
2. **Qdrant** thay FAISS (Tier 1 #3): chạy Qdrant bằng Docker, học vector DB thật dùng trong production.
3. **LangGraph** (Tier 1 #4): biến thành agent — tự quyết định có cần tra tài liệu không, tự viết lại câu hỏi.
4. **Đánh giá RAG**: tạo bộ 20 câu hỏi/đáp án, đo độ chính xác retrieval (hit@k) và chất lượng trả lời.
5. **Reranking**: thêm bước xếp hạng lại kết quả bằng cross-encoder để tăng độ chính xác.

> Làm xong bước nào, commit lên GitHub và ghi vào CV: *"Built a Vietnamese RAG chatbot (sentence-transformers + FAISS + Gemini + FastAPI) with source citation."*
