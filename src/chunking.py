"""
Cắt văn bản dài thành các đoạn (chunk) chồng lấn nhau.
Tách riêng file này (chỉ dùng thư viện chuẩn) để:
  - ingest.py và evaluation/ dùng CHUNG một cách cắt,
  - test được trên GitHub Actions mà không cần cài model nặng.
"""


def chunk_text(text: str, size: int, overlap: int):
    """Cắt văn bản dài thành các đoạn nhỏ chồng lấn nhau (giữ mạch ý ở ranh giới)."""
    if overlap >= size:
        raise ValueError("overlap phải nhỏ hơn size (nếu không vòng lặp không tiến lên được)")
    text = " ".join(text.split())  # gộp khoảng trắng/xuống dòng thừa
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]
