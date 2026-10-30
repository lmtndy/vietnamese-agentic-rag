"""
MODULE 04 — Agentic RAG bằng LangGraph.

Khác pipeline cố định (Module 01-03): ở đây agent TỰ QUYẾT ĐỊNH các bước:
  route   : câu hỏi này có cần tra tài liệu không? (RETRIEVE / DIRECT)
  grade   : tài liệu tìm được có liên quan không? (YES / NO)
  rewrite : nếu NO -> viết lại câu hỏi rồi tra lại (lặp tối đa AGENT_MAX_RETRIES)

Sơ đồ:
  START -> route --(direct)--> answer_direct -> END
                \--(retrieve)-> retrieve -> grade --(yes)--> generate -> END
                                               \--(no)--> rewrite -> retrieve (loop)

Chạy thử:  python -m src.agent
"""
import time

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src import config
from src.rag import RAGEngine


# ----- STATE: "dict" chung chảy qua đồ thị -----
class AgentState(TypedDict, total=False):
    question: str      # câu hỏi gốc
    query: str         # câu hỏi hiện tại (có thể đã rewrite) dùng để tra
    documents: list    # các chunk tìm được (sau rerank)
    answer: str        # câu trả lời cuối
    route: str         # "retrieve" | "direct"
    relevant: bool     # tài liệu có liên quan không
    retries: int       # số lần đã rewrite


def _ask_gemini(engine: RAGEngine, prompt: str, tries: int = 2) -> str:
    """Gọi Gemini, thử lại nhẹ nếu lỗi tạm thời (503)."""
    last = None
    for _ in range(tries):
        try:
            resp = engine.gemini.models.generate_content(
                model=config.GEMINI_MODEL, contents=prompt
            )
            return (resp.text or "").strip()
        except Exception as e:      # noqa: BLE001
            last = e
            time.sleep(3)
    raise last


class RAGAgent:
    def __init__(self):
        self.engine = RAGEngine()          # tái dùng Module 03 (retrieve + rerank + gemini)
        self.app = self._build_graph()

    # ---------- CÁC NODE (mỗi node làm 1 việc, trả về phần cập nhật state) ----------
    def _route(self, state: AgentState):
        q = state["question"]
        prompt = (
            "Phân loại câu hỏi. Nếu CẦN tra tài liệu để trả lời (hỏi về nội dung/kiến thức) "
            "-> trả về đúng 1 từ: RETRIEVE. Nếu chỉ chào hỏi/xã giao/chung chung không cần "
            f"tài liệu -> trả về: DIRECT.\nCâu hỏi: {q}\nTrả lời (1 từ):"
        )
        route = "direct" if "DIRECT" in _ask_gemini(self.engine, prompt).upper() else "retrieve"
        print(f"  🔀 route = {route}")
        return {"route": route, "query": q, "retries": 0}

    def _answer_direct(self, state: AgentState):
        prompt = f"Trả lời ngắn gọn, thân thiện bằng tiếng Việt:\n{state['question']}"
        return {"answer": _ask_gemini(self.engine, prompt), "documents": []}

    def _retrieve(self, state: AgentState):
        q = state.get("query") or state["question"]
        docs = self.engine.rerank(q, self.engine.retrieve(q))   # bi-encoder (Qdrant) -> cross-encoder
        print(f"  🔎 retrieve: lấy {len(docs)} chunk (query='{q[:40]}...')")
        return {"documents": docs}

    def _grade(self, state: AgentState):
        joined = "\n---\n".join(d["text"][:300] for d in state["documents"][:3])
        prompt = (
            "Các đoạn tài liệu dưới đây có chứa thông tin để trả lời câu hỏi không? "
            f"Trả về đúng 1 từ: YES hoặc NO.\nCâu hỏi: {state['question']}\n"
            f"Tài liệu:\n{joined}\nTrả lời:"
        )
        relevant = "YES" in _ask_gemini(self.engine, prompt).upper()
        print(f"  ✅ grade: tài liệu liên quan = {relevant}")
        return {"relevant": relevant}

    def _rewrite(self, state: AgentState):
        prompt = (
            "Viết lại câu hỏi sau cho rõ ràng, giàu từ khóa hơn để tìm tài liệu tốt hơn "
            f"(giữ nguyên ý, tiếng Việt, chỉ trả về câu hỏi mới):\n{state['question']}"
        )
        new_q = _ask_gemini(self.engine, prompt)
        print(f"  ✍️  rewrite -> '{new_q[:50]}...'")
        return {"query": new_q, "retries": state.get("retries", 0) + 1}

    def _generate(self, state: AgentState):
        prompt = self.engine.build_prompt(state["question"], state["documents"])
        return {"answer": _ask_gemini(self.engine, prompt)}

    # ---------- CÁC HÀM RẼ NHÁNH (conditional edges) ----------
    @staticmethod
    def _decide_route(state: AgentState):
        return state["route"]                       # "retrieve" | "direct"

    @staticmethod
    def _decide_after_grade(state: AgentState):
        if state.get("relevant"):
            return "generate"
        if state.get("retries", 0) >= config.AGENT_MAX_RETRIES:
            return "generate"                       # hết lượt -> cứ trả lời với gì đang có
        return "rewrite"

    # ---------- DỰNG ĐỒ THỊ ----------
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("route", self._route)
        g.add_node("answer_direct", self._answer_direct)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("rewrite", self._rewrite)
        g.add_node("generate", self._generate)

        g.add_edge(START, "route")
        g.add_conditional_edges("route", self._decide_route,
                                {"retrieve": "retrieve", "direct": "answer_direct"})
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._decide_after_grade,
                                {"generate": "generate", "rewrite": "rewrite"})
        g.add_edge("rewrite", "retrieve")
        g.add_edge("answer_direct", END)
        g.add_edge("generate", END)
        return g.compile()

    def run(self, question: str):
        result = self.app.invoke({"question": question})
        return result.get("answer", ""), result.get("documents", [])


def main():
    print("Đang khởi tạo Agent (model + Qdrant)...")
    agent = RAGAgent()
    print("✅ Sẵn sàng! Gõ câu hỏi (exit để thoát).\n")
    while True:
        try:
            q = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "thoat"} or not q:
            if not q:
                continue
            break
        print("  — agent đang suy nghĩ —")
        answer, docs = agent.run(q)
        print(f"\nBot: {answer}")
        if docs:
            print(f"  (dựa trên {len(docs)} đoạn tài liệu)")
        print()
    print("Tạm biệt!")


if __name__ == "__main__":
    main()
