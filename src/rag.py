"""
GIAI ĐOẠN ONLINE của RAG — Advanced (Module 03):
    câu hỏi
      -> RETRIEVE  : Qdrant lấy THÔ top-N chunk (bi-encoder, nhanh)
      -> RERANK    : cross-encoder chấm lại, giữ top-K tốt nhất (chính xác)
      -> AUGMENT   : ghép top-K vào prompt
      -> GENERATE  : Gemini sinh câu trả lời có trích nguồn.
"""
from google import genai
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from src import config


class RAGEngine:
    def __init__(self, require_llm: bool = True):
        # Bi-encoder: mã hóa câu hỏi thành vector để tìm trong Qdrant
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        # Cross-encoder: đọc (câu hỏi + chunk) cùng lúc để chấm điểm liên quan chính xác
        self.reranker = CrossEncoder(config.RERANK_MODEL)
        # Kết nối tới Qdrant (vector DB chạy riêng)
        self.qdrant = QdrantClient(url=config.QDRANT_URL)

        if not require_llm:     # evaluation chỉ đo retrieve + rerank -> không cần Gemini
            self.gemini = None
            return
        if not config.GOOGLE_API_KEY:
            raise RuntimeError(
                "Chưa có GOOGLE_API_KEY. Tạo file .env theo mẫu .env.example rồi dán key vào."
            )
        self.gemini = genai.Client(api_key=config.GOOGLE_API_KEY)

    # ---------- BƯỚC 1: RETRIEVE (lấy thô, nhanh) ----------
    def retrieve(self, query: str, top_n: int = None, collection: str = None):
        """Qdrant trả top-N chunk gần nghĩa nhất (bi-encoder + cosine)."""
        top_n = top_n or config.TOP_N
        q_vec = self.embedder.encode(
            [f"query: {query}"], normalize_embeddings=True
        )[0].tolist()
        hits = self.qdrant.query_points(
            collection_name=collection or config.QDRANT_COLLECTION,
            query=q_vec,
            limit=top_n,
            with_payload=True,
        ).points
        return [
            {"text": h.payload["text"], "source": h.payload["source"], "score": float(h.score)}
            for h in hits
        ]

    # ---------- BƯỚC 2: RERANK (lọc tinh, chính xác) ----------
    def rerank(self, query: str, candidates: list, top_k: int = None):
        """Cross-encoder chấm lại từng (câu hỏi, chunk), giữ top-K điểm cao nhất."""
        top_k = top_k or config.TOP_K
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)   # đọc cả cặp -> điểm liên quan
        for c, s in zip(candidates, scores):
            c["retrieve_score"] = c["score"]     # giữ lại điểm bi-encoder (cosine)
            c["score"] = float(s)                # điểm rerank (hiển thị)
        return sorted(candidates, key=lambda c: c["score"], reverse=True)[:top_k]

    # ---------- BƯỚC 3: AUGMENT ----------
    def build_prompt(self, query: str, contexts: list) -> str:
        """Ghép câu hỏi + ngữ cảnh đã rerank thành 1 prompt, buộc LLM trích nguồn."""
        context_block = "\n\n".join(
            f"[Nguồn {i + 1}: {c['source']}]\n{c['text']}"
            for i, c in enumerate(contexts)
        )
        return f"""Bạn là trợ lý hỏi-đáp. CHỈ trả lời dựa trên NGỮ CẢNH bên dưới.
Nếu ngữ cảnh không chứa thông tin, hãy nói: "Tôi không tìm thấy thông tin trong tài liệu."
Luôn trích dẫn nguồn dạng [Nguồn X] ngay sau ý liên quan.

### NGỮ CẢNH:
{context_block}

### CÂU HỎI:
{query}

### TRẢ LỜI (tiếng Việt, ngắn gọn, chính xác):"""

    # ---------- BƯỚC 4: GENERATE ----------
    def answer(self, query: str, top_n: int = None, top_k: int = None):
        candidates = self.retrieve(query, top_n)          # lấy thô top-N
        contexts = self.rerank(query, candidates, top_k)  # lọc tinh còn top-K
        prompt = self.build_prompt(query, contexts)
        resp = self.gemini.models.generate_content(
            model=config.GEMINI_MODEL, contents=prompt
        )
        return resp.text, contexts
