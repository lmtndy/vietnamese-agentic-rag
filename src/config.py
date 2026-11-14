"""
Cấu hình tập trung cho toàn bộ project.
Mọi tham số (model, kích thước chunk, Qdrant, rerank...) đặt ở đây để dễ chỉnh.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Nạp các biến môi trường từ file .env (nếu có)
load_dotenv()

# --- Đường dẫn ---
ROOT = Path(__file__).resolve().parent.parent   # thư mục gốc project
DATA_DIR = ROOT / "data"                         # nơi để tài liệu nguồn (.txt/.pdf)

# --- Embedding (BI-ENCODER: mã hóa câu hỏi & chunk RIÊNG RẼ -> nhanh) ---
EMBED_MODEL = "intfloat/multilingual-e5-small"   # vector 384 chiều, đa ngôn ngữ
EMBED_DIM = 384                                  # số chiều vector của model trên

# --- Chunking (cắt tài liệu dài thành đoạn nhỏ) ---
CHUNK_SIZE = 500       # số ký tự mỗi đoạn
CHUNK_OVERLAP = 100    # số ký tự chồng lấn giữa 2 đoạn liền nhau

# --- Qdrant (VECTOR DATABASE chạy riêng, thay cho FAISS) ---
# Chạy thẳng trên máy: localhost. Trong docker-compose: biến môi trường QDRANT_URL=http://qdrant:6333
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "rag_chunks"

# --- Retrieve-then-rerank ---
TOP_N = 20       # Bước 1: lấy THÔ top-N chunk từ Qdrant (bi-encoder, nhanh)
TOP_K = 4        # Bước 2: rerank còn top-K chunk tốt nhất (cross-encoder, chính xác)
# RERANKER (CROSS-ENCODER: đọc (câu hỏi + chunk) CÙNG LÚC -> chính xác hơn, chậm hơn)
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # reranker đa ngôn ngữ (có tiếng Việt)

# --- Agent (Module 04 — LangGraph) ---
AGENT_MAX_RETRIES = 2   # số lần tối đa rewrite câu hỏi + tra lại

# --- Evaluation (Module 06) ---
EVAL_DIR = ROOT / "evaluation" / "data"          # bộ câu hỏi chuẩn (tự tải về, KHÔNG đưa lên git)
EVAL_DATASET = "taidng/UIT-ViQuAD2.0"            # hỏi-đáp tiếng Việt trên Wikipedia, có sẵn đáp án
EVAL_COLLECTION = "eval_viquad"                  # collection RIÊNG -> không đụng tài liệu thật
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

# --- Gemini (sinh câu trả lời) ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-3.8-flash"   # nhanh, nằm trong free tier
