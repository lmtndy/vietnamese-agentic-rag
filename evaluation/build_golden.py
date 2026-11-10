"""
MODULE 06 — Bước 1: tạo BỘ CÂU HỎI CHUẨN (golden set) từ UIT-ViQuAD 2.0.

Mỗi câu hỏi biết sẵn đoạn văn (context) chứa đáp án + đáp án đúng
-> đo được hệ thống có TÌM ĐÚNG đoạn văn hay không, không cần người chấm.

Chạy:  python -m evaluation.build_golden                    (mặc định 300 câu hỏi)
       python -m evaluation.build_golden --n-questions 500
Ra:    evaluation/data/corpus.jsonl     (mọi đoạn văn = "kho tài liệu")
       evaluation/data/questions.jsonl  (câu hỏi + id đoạn văn đúng + đáp án)
"""
import argparse
import json
import random

from datasets import load_dataset

from src import config


def answer_texts(answers):
    """Lấy danh sách đáp án (dataset có thể lưu dạng dict-of-lists hoặc list-of-dicts)."""
    texts = answers["text"] if isinstance(answers, dict) else [a["text"] for a in answers]
    return [t for t in texts if t.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-questions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[1/3] Tải {config.EVAL_DATASET} (split validation) ...")
    ds = load_dataset(config.EVAL_DATASET, split="validation")

    print("[2/3] Gom đoạn văn thành kho tài liệu + lọc câu hỏi có đáp án")
    # Kho = MỌI đoạn văn trong split: đoạn không được chọn câu hỏi vẫn nằm trong kho để làm "nhiễu"
    doc_ids, corpus, candidates = {}, [], []
    for row in ds:
        ctx = row["context"]
        if ctx not in doc_ids:
            doc_ids[ctx] = f"doc{len(doc_ids)}"
            corpus.append({"doc_id": doc_ids[ctx], "title": row["title"], "text": ctx})
        answers = answer_texts(row["answers"])
        if not row["is_impossible"] and answers:          # bỏ câu hỏi "không có đáp án" của ViQuAD 2.0
            candidates.append({"qid": row["id"], "question": row["question"],
                               "doc_id": doc_ids[ctx], "answers": answers})

    # Mỗi đoạn văn tối đa 1 câu hỏi -> câu hỏi trải đều trên nhiều đoạn, không dồn vào đoạn dài
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    picked, used = [], set()
    for c in candidates:
        if c["doc_id"] not in used:
            picked.append(c)
            used.add(c["doc_id"])
        if len(picked) == args.n_questions:
            break

    print("[3/3] Ghi file")
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.EVAL_DIR / "corpus.jsonl", corpus)
    write_jsonl(config.EVAL_DIR / "questions.jsonl", picked)
    print(f"✅ Kho: {len(corpus)} đoạn văn | Câu hỏi chuẩn: {len(picked)} -> {config.EVAL_DIR}")
    print("   Ví dụ:", picked[0])


if __name__ == "__main__":
    main()
