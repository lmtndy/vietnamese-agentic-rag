"""
GIAI ĐOẠN OFFLINE của RAG (chạy 1 lần mỗi khi tài liệu thay đổi):
    đọc tài liệu  ->  cắt thành chunk  ->  tạo embedding  ->  NẠP VÀO QDRANT.

Chạy:  python -m src.ingest
(Cần Qdrant đang chạy: docker run -d --name qdrant-rag -p 6333:6333 ... qdrant/qdrant)
"""
from pathlib import Path

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src import config


def load_documents(data_dir: Path):
    """Đọc toàn bộ file .txt và .pdf trong thư mục data/. Trả về list (tên_file, nội_dung)."""
    docs = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() == ".txt":
            docs.append((path.name, path.read_text(encoding="utf-8")))
        elif path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append((path.name, text))
    return docs


def chunk_text(text: str, size: int, overlap: int):
    """Cắt văn bản dài thành các đoạn nhỏ chồng lấn nhau (giữ mạch ý ở ranh giới)."""
    text = " ".join(text.split())  # gộp khoảng trắng/xuống dòng thừa
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def main():
    print(f"[1/4] Đọc tài liệu trong: {config.DATA_DIR}")
    docs = load_documents(config.DATA_DIR)
    if not docs:
        print("!! Không có tài liệu. Hãy bỏ file .txt/.pdf vào thư mục data/ rồi chạy lại.")
        return

    # Cắt chunk và ghi nhớ nguồn gốc (để trích dẫn sau này)
    chunks, metadatas = [], []
    for source, text in docs:
        for i, ch in enumerate(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            chunks.append(ch)
            metadatas.append({"source": source, "chunk_id": i})
    print(f"    -> {len(docs)} tài liệu  =>  {len(chunks)} đoạn (chunks)")

    print(f"[2/4] Tải model embedding: {config.EMBED_MODEL}")
    embedder = SentenceTransformer(config.EMBED_MODEL)

    print("[3/4] Tạo vector embedding cho từng đoạn ...")
    # Model họ e5 yêu cầu thêm tiền tố "passage: " cho tài liệu được lưu trữ
    inputs = [f"passage: {c}" for c in chunks]
    vectors = embedder.encode(inputs, normalize_embeddings=True, show_progress_bar=True)

    print(f"[4/4] Nạp vào Qdrant ({config.QDRANT_URL}, collection='{config.QDRANT_COLLECTION}') ...")
    client = QdrantClient(url=config.QDRANT_URL)
    # Tạo lại collection (xóa cũ nếu có) với không gian vector 384 chiều, đo bằng COSINE
    if client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
    )
    # Mỗi chunk là 1 "point": id + vector + payload (text & nguồn để lọc/trích dẫn)
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"text": chunks[i], "source": metadatas[i]["source"], "chunk_id": metadatas[i]["chunk_id"]},
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)

    count = client.count(config.QDRANT_COLLECTION).count
    print(f"\n✅ Xong! Qdrant collection '{config.QDRANT_COLLECTION}' có {count} points.")
    print("   Tiếp theo chạy:  python -m src.chat")


if __name__ == "__main__":
    main()
