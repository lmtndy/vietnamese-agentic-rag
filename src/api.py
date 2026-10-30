"""
MODULE 02 — FastAPI: biến RAG chatbot thành REST API có Swagger UI.

Chạy server:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
Mở Swagger UI (trang test API):
    http://<địa-chỉ-máy>:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src import config
from src.rag import RAGEngine

# Nơi giữ engine dùng chung — nạp MỘT LẦN lúc khởi động, không nạp lại mỗi request
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Chạy KHI SERVER KHỞI ĐỘNG ---
    print("Đang nạp RAG engine (model embedding + index)...")
    _state["rag"] = RAGEngine()
    print("✅ RAG engine sẵn sàng.")
    yield
    # --- Chạy KHI SERVER TẮT ---
    _state.clear()


app = FastAPI(
    title="RAG Chatbot API (tiếng Việt)",
    description="Hỏi đáp trên tài liệu, có trích nguồn. Module 02 — FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Định nghĩa SCHEMA input/output bằng Pydantic ----------
# Pydantic tự kiểm tra kiểu dữ liệu + tự sinh tài liệu trong Swagger.
class AskRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi về tài liệu", examples=["RAG là gì?"])
    top_k: int = Field(config.TOP_K, ge=1, le=10, description="Số đoạn tài liệu lấy ra")


class Source(BaseModel):
    source: str = Field(..., description="Tên file nguồn")
    score: float = Field(..., description="Độ liên quan (cosine, càng gần 1 càng tốt)")
    preview: str = Field(..., description="Trích đoạn đầu của chunk")


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------- Các ENDPOINT ----------
@app.get("/", summary="Thông tin cơ bản")
def root():
    return {"message": "RAG Chatbot API đang chạy. Mở /docs để dùng Swagger UI."}


@app.get("/health", summary="Kiểm tra tình trạng")
def health():
    return {"status": "ok", "model": config.GEMINI_MODEL}


@app.post("/ask", response_model=AskResponse, summary="Hỏi đáp trên tài liệu")
def ask(req: AskRequest):
    """Nhận câu hỏi → truy xuất tài liệu → sinh câu trả lời có trích nguồn."""
    rag: RAGEngine = _state["rag"]
    answer_text, contexts = rag.answer(req.question, top_k=req.top_k)
    sources = [
        Source(source=c["source"], score=round(c["score"], 3), preview=c["text"][:100])
        for c in contexts
    ]
    return AskResponse(answer=answer_text, sources=sources)
