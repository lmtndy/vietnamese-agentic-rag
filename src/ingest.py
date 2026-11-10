"""
GIAI ĐOẠN OFFLINE của RAG (chạy 1 lần mỗi khi tài liệu thay đổi):
    đọc tài liệu  ->  cắt thành chunk  ->  tạo embedding  ->  NẠP VÀO QDRANT.

Chạy:  python -m src.ingest
(Cần Qdrant đang chạy: docker run -d --name qdrant-rag -p 127.0.0.1:6333:6333 ... qdrant/qdrant)
"""
from pathlib import Path

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src import config
from src.chunking import chunk_text


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


def index_documents(client, embedder, docs, collection, chunk_size, chunk_overlap, batch_size=256):
    """Cắt chunk -> embedding -> tạo lại collection -> nạp vào Qdrant. Trả về số chunk đã nạp.
    docs: list (nguồn, nội_dung). Dùng chung cho ingest (tài liệu thật) và evaluation (bộ câu hỏi chuẩn)."""
    # Cắt chunk và ghi nhớ nguồn gốc (để trích dẫn sau này)
    chunks, metadatas = [], []
    for source, text in docs:
        for i, ch in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            chunks.append(ch)
            metadatas.append({"source": source, "chunk_id": i})

    # Model họ e5 yêu cầu thêm tiền tố "passage: " cho tài liệu được lưu trữ
    inputs = [f"passage: {c}" for c in chunks]
    vectors = embedder.encode(inputs, normalize_embeddings=True, show_progress_bar=True)

    # Tạo lại collection (xóa cũ nếu có) với không gian vector 384 chiều, đo bằng COSINE
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
    )
    # Mỗi chunk là 1 "point": id + vector + payload (text & nguồn để lọc/trích dẫn)
    points = [
        PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": chunks[i], **metadatas[i]})
        for i in range(len(chunks))
    ]
    for start in range(0, len(points), batch_size):   # nạp theo lô để mỗi request không quá lớn
        client.upsert(collection_name=collection, points=points[start:start + batch_size])
    return len(chunks)


def main():
    print(f"[1/3] Đọc tài liệu trong: {config.DATA_DIR}")
    docs = load_documents(config.DATA_DIR)
    if not docs:
        print("!! Không có tài liệu. Hãy bỏ file .txt/.pdf vào thư mục data/ rồi chạy lại.")
        return

    print(f"[2/3] Tải model embedding: {config.EMBED_MODEL}")
    embedder = SentenceTransformer(config.EMBED_MODEL)

    print(f"[3/3] Cắt chunk + embedding + nạp vào Qdrant ({config.QDRANT_URL}, collection='{config.QDRANT_COLLECTION}') ...")
    client = QdrantClient(url=config.QDRANT_URL)
    n_chunks = index_documents(client, embedder, docs, config.QDRANT_COLLECTION,
                               config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"    -> {len(docs)} tài liệu  =>  {n_chunks} đoạn (chunks)")

    count = client.count(config.QDRANT_COLLECTION).count
    print(f"\n✅ Xong! Qdrant collection '{config.QDRANT_COLLECTION}' có {count} points.")
    print("   Tiếp theo chạy:  python -m src.chat")


if __name__ == "__main__":
    main()
