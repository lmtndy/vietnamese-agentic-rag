"""
Giao diện dòng lệnh (CLI) để trò chuyện với tài liệu của bạn.
Chạy:  python -m src.chat
"""
from src.rag import RAGEngine


def main():
    print("Đang khởi tạo RAG engine (tải model embedding) ...")
    engine = RAGEngine()
    print("✅ Sẵn sàng! Đặt câu hỏi về tài liệu trong data/. Gõ 'exit' để thoát.\n")

    while True:
        try:
            query = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit", "thoat"}:
            break
        if not query:
            continue

        answer, contexts = engine.answer(query)
        print(f"\nBot: {answer}\n")
        print("--- Nguồn tham khảo ---")
        for i, c in enumerate(contexts):
            preview = c["text"][:80].replace("\n", " ")
            print(f"  [Nguồn {i + 1}] {c['source']} (score={c['score']:.3f}): {preview}...")
        print()

    print("Tạm biệt!")


if __name__ == "__main__":
    main()
