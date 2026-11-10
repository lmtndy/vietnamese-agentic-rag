"""
MODULE 06 — Bước 2: ĐO chất lượng RETRIEVAL rồi ghi kết quả vào MLflow.

So sánh 2 cách xếp hạng trên CÙNG bộ câu hỏi chuẩn:
  bi     : chỉ Qdrant (bi-encoder) — thứ tự top-N trả về
  rerank : Qdrant top-N -> cross-encoder chấm lại (đúng như RAGEngine dùng khi trả lời thật)
Chỉ số: hit@1/3/K/N, MRR (src/metrics.py) + thời gian mỗi bước (K = TOP_K, N = TOP_N trong config).
"Đúng" = chunk thuộc đúng đoạn văn của câu hỏi VÀ chứa đáp án.

Chạy:  python -m evaluation.run_retrieval_eval                    (chunk theo config: 500/100)
       python -m evaluation.run_retrieval_eval --chunk-size 200   (thử cấu hình khác)
Xem:   mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
"""
import argparse
import json
import time

import mlflow

from src import config
from src.ingest import index_documents
from src.metrics import summarize
from src.rag import RAGEngine


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_relevant(chunk, q):
    text = chunk["text"].lower()
    return chunk["source"] == q["doc_id"] and any(a.lower() in text for a in q["answers"])


def first_hit(relevance):
    """Hạng (1-based) của chunk đúng đầu tiên, None nếu không có."""
    return next((i + 1 for i, r in enumerate(relevance) if r), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--top-n", type=int, default=config.TOP_N)
    parser.add_argument("--limit", type=int, help="chỉ chạy N câu hỏi đầu (để thử nhanh)")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    # hit@TOP_K: LLM chỉ đọc top-K chunk -> chỉ số quan trọng nhất
    # hit@top_n: "trần" của rerank (rerank chỉ sắp lại top-N, không tìm thêm được)
    ks = tuple(sorted({1, 3, config.TOP_K, args.top_n}))

    corpus = load_jsonl(config.EVAL_DIR / "corpus.jsonl")
    questions = load_jsonl(config.EVAL_DIR / "questions.jsonl")[:args.limit]

    print("[1/3] Nạp model (bi-encoder + cross-encoder), không cần Gemini")
    engine = RAGEngine(require_llm=False)

    collection = f"{config.EVAL_COLLECTION}_c{args.chunk_size}_o{args.chunk_overlap}"
    print(f"[2/3] Index {len(corpus)} đoạn văn vào collection '{collection}'")
    t0 = time.perf_counter()
    n_chunks = index_documents(engine.qdrant, engine.embedder,
                               [(d["doc_id"], d["text"]) for d in corpus],
                               collection, args.chunk_size, args.chunk_overlap)
    index_seconds = time.perf_counter() - t0

    print(f"[3/3] Chạy {len(questions)} câu hỏi")
    bi_rel, rr_rel, rows = [], [], []
    t_retrieve = t_rerank = 0.0
    for q in questions:
        t0 = time.perf_counter()
        candidates = engine.retrieve(q["question"], top_n=args.top_n, collection=collection)
        t1 = time.perf_counter()
        # rerank ghi đè "score" trong dict -> đưa bản sao; giữ đủ top-N để tính MRR
        reranked = engine.rerank(q["question"], [dict(c) for c in candidates], top_k=args.top_n)
        t2 = time.perf_counter()
        t_retrieve += t1 - t0
        t_rerank += t2 - t1

        bi_rel.append([is_relevant(c, q) for c in candidates])
        rr_rel.append([is_relevant(c, q) for c in reranked])
        rows.append({"qid": q["qid"], "question": q["question"], "answers": q["answers"],
                     "bi_rank": first_hit(bi_rel[-1]), "rerank_rank": first_hit(rr_rel[-1]),
                     "rerank_top1": reranked[0]["text"] if reranked else ""})
    engine.qdrant.delete_collection(collection)     # dọn collection tạm, Qdrant chỉ giữ tài liệu thật

    bi, rr = summarize(bi_rel, ks), summarize(rr_rel, ks)
    n = len(questions)
    latency = {"latency_ms/retrieve": 1000 * t_retrieve / n, "latency_ms/rerank": 1000 * t_rerank / n}

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("rag-retrieval")
    with mlflow.start_run(run_name=args.run_name or f"chunk{args.chunk_size}-overlap{args.chunk_overlap}"):
        mlflow.log_params({
            "dataset": config.EVAL_DATASET, "n_questions": n, "n_docs": len(corpus),
            "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap, "n_chunks": n_chunks,
            "top_n": args.top_n, "embed_model": config.EMBED_MODEL, "rerank_model": config.RERANK_MODEL,
        })
        mlflow.log_metrics({**{f"bi/{k}": v for k, v in bi.items()},
                            **{f"rerank/{k}": v for k, v in rr.items()},
                            **latency, "index_seconds": index_seconds})
        mlflow.log_text(json.dumps(rows, ensure_ascii=False, indent=1), "per_question.json")

    print(f"\n=== {n} câu hỏi | {len(corpus)} đoạn văn -> {n_chunks} chunk "
          f"(size {args.chunk_size}, overlap {args.chunk_overlap}) ===")
    header = "".join(f"{f'hit@{k}':>9}" for k in ks)
    print(f"{'':<14}{header}{'MRR':>8}{'ms/câu':>9}")
    for name, m, ms in [("bi-encoder", bi, latency["latency_ms/retrieve"]),
                        ("+ rerank", rr, latency["latency_ms/retrieve"] + latency["latency_ms/rerank"])]:
        cells = "".join(f"{m[f'hit_at_{k}']:>9.3f}" for k in ks)
        print(f"{name:<14}{cells}{m['mrr']:>8.3f}{ms:>9.0f}")
    print(f"\n✅ Đã ghi vào MLflow (experiment 'rag-retrieval') -> {config.MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
