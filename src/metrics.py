"""
Chỉ số đánh giá RETRIEVAL (Module 06).

Đầu vào chung: `relevance` = list True/False theo THỨ TỰ xếp hạng của 1 câu hỏi.
    VD: [False, True, False] -> chunk đúng nằm ở hạng 2.

  - hit@k : trong top-k có ít nhất 1 chunk đúng không? (1 hoặc 0)
  - MRR   : 1 / (hạng của chunk đúng ĐẦU TIÊN)  -> hạng 1 = 1.0, hạng 2 = 0.5, không có = 0
Trung bình trên toàn bộ câu hỏi = điểm của hệ thống.
"""


def hit_at_k(relevance: list, k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: list) -> float:
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def summarize(all_relevance: list, ks=(1, 3, 5)) -> dict:
    """Trung bình hit@k và MRR trên nhiều câu hỏi. Trả về dict tên-chỉ-số -> giá trị."""
    n = len(all_relevance)
    if n == 0:
        raise ValueError("Không có câu hỏi nào để đánh giá")
    out = {f"hit_at_{k}": sum(hit_at_k(r, k) for r in all_relevance) / n for k in ks}
    out["mrr"] = sum(reciprocal_rank(r) for r in all_relevance) / n
    return out
