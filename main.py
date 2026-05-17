"""Entry point: chat loop, input check, orchestration.

Pipeline:
  user_query → input_check → classify_intent
    ├─ chitchat       → model + tools (time / weather / exchange rate)
    ├─ clarify        → hỏi lại
    ├─ out_of_scope   → báo scope
    └─ vn_history     → PDF RAG → (nếu fail) Wiki → (nếu fail) Model
"""

import json

from config import (
    openai_client,
    ensure_index,
    CHAT_MODEL,
    MAX_INPUT_LEN,
    FALLBACK_NO_INFO,
)
from history import ChatHistory
from intent import classify_intent
from rag_pdf import rag_pdf_query
from wiki_fallback import wiki_query
from model_fallback import model_query
from tools import get_tool_schemas, execute_tool


# ============================================================
# Input check
# ============================================================

def check_input(raw: str) -> tuple[bool, str, str]:
    """Trả về (ok, cleaned, error_msg)."""
    if raw is None:
        return False, "", "Input rỗng."
    cleaned = raw.strip()
    if not cleaned:
        return False, "", "Bạn chưa nhập câu hỏi."
    if len(cleaned) > MAX_INPUT_LEN:
        return False, "", f"Câu hỏi quá dài (>{MAX_INPUT_LEN} ký tự). Vui lòng rút gọn."
    return True, cleaned, ""


# ============================================================
# Chitchat với tool loop (auto tool calling, không if/else)
# ============================================================

CHITCHAT_SYSTEM_PROMPT = (
    "Bạn là trợ lý lịch sử Việt Nam thân thiện. "
    "Với câu hỏi không phải lịch sử, trả lời ngắn gọn, tự nhiên. "
    "Khi cần dữ liệu real-time (giờ hiện tại, thời tiết, tỷ giá), DÙNG TOOLS có sẵn — "
    "không bịa số liệu. Sau khi nhận tool result, tổng hợp thành câu trả lời tự nhiên. "
    "Nếu user hỏi về khả năng của bạn, nói bạn chuyên về lịch sử Việt Nam."
)


def handle_chitchat(query: str, history: ChatHistory, max_iter: int = 4) -> str:
    """Multi-iteration tool loop: LLM tự chọn tool tới khi có câu trả lời cuối."""
    messages = history.context(CHITCHAT_SYSTEM_PROMPT) + [
        {"role": "user", "content": query}
    ]
    tool_schemas = get_tool_schemas()

    for iteration in range(max_iter):
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
            max_completion_tokens=400,
            temperature=0.5,
        )
        response_message = response.choices[0].message

        if not response_message.tool_calls:
            return (response_message.content or "").strip()

        print(f"  [iter {iteration + 1}] LLM gọi {len(response_message.tool_calls)} tool:")
        messages.append(response_message)

        for tc in response_message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            print(f"    🔧 {name}({args})")
            result = execute_tool(name, args)
            print(f"    ✓ {result[:100]}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return "Đã đạt giới hạn tool iterations."


# ============================================================
# VN History cascade
# ============================================================

def handle_vn_history(
    query: str, index, category: str | None = None, verbose: bool = True, history=None
) -> str:
    """Cascade: PDF → Wiki → Model. category dùng để filter Pinecone ở step PDF.

    history (ChatHistory) được truyền xuống cả 3 tầng để model nhớ câu hỏi
    trước đó của user (giống handle_chitchat). Retrieval vẫn dùng query đã
    rewrite; grounding vẫn buộc fact từ CONTEXT theo system prompt mỗi tầng.
    """
    if verbose:
        cat_tag = f" (category={category})" if category else " (no filter)"
        print(f"→ Step 1: RAG từ PDF{cat_tag}...")
    pdf_ans, pdf_grounded, _ = rag_pdf_query(
        index, query, category=category, verbose=verbose, history=history
    )
    if pdf_grounded and FALLBACK_NO_INFO not in pdf_ans:
        return pdf_ans + "\n(Nguồn: tài liệu PDF)"

    if verbose:
        print("→ Step 2: Fallback Wikipedia...")
    wiki_ans, wiki_grounded = wiki_query(query, verbose=verbose, history=history)
    if wiki_grounded and FALLBACK_NO_INFO not in wiki_ans:
        return wiki_ans

    if verbose:
        print("→ Step 3: Fallback Model (guardrails)...")
    return model_query(query, verbose=verbose, history=history)


# ============================================================
# Single-turn orchestration (dùng chung cho CLI + Streamlit)
# ============================================================

def answer_query(query: str, index, history: ChatHistory, verbose: bool = True):
    """Xử lý 1 lượt: classify → route → trả (answer, intent_result).

    Không tự append vào history (để caller quyết định), trừ summarization
    nội bộ của ChatHistory khi caller gọi append.
    """
    intent_result = classify_intent(query, history.recent())
    if verbose:
        print(
            f"[Intent: {intent_result.intent} | "
            f"Category: {intent_result.category or '-'} | "
            f"Confidence: {intent_result.confidence:.2f} | "
            f"{intent_result.reasoning}]"
        )

    if intent_result.intent == "clarify":
        answer = (
            intent_result.clarifying_question
            or "Bạn có thể nói rõ hơn câu hỏi không?"
        )
    elif intent_result.intent == "out_of_scope":
        answer = (
            "Tôi chỉ chuyên về lịch sử Việt Nam, không thể trả lời câu hỏi này. "
            "Bạn thử hỏi về lịch sử Việt Nam nhé."
        )
    elif intent_result.intent == "chitchat":
        answer = handle_chitchat(query, history)
    else:  # vn_history
        rag_query = intent_result.rewritten_query or query
        if verbose and intent_result.rewritten_query:
            print(f"  ✏️  Rewrite cho RAG: '{rag_query}'")
        answer = handle_vn_history(
            rag_query,
            index,
            category=intent_result.category,
            verbose=verbose,
            history=history,
        )

    return answer, intent_result


# ============================================================
# Main loop
# ============================================================

def main():
    index = ensure_index()
    history = ChatHistory()

    print("\n" + "=" * 60)
    print("🇻🇳  RAG Lịch Sử Việt Nam (with real-time tools)")
    print("=" * 60)
    print("Gõ 'exit' để thoát.\n")

    while True:
        try:
            raw = input("👤 Bạn: ")
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if raw.strip().lower() == "exit":
            print("Tạm biệt!")
            break

        ok, query, err = check_input(raw)
        if not ok:
            print(f"⚠️  {err}\n")
            continue

        answer, intent_result = answer_query(query, index, history, verbose=True)
        tag = " (clarify)" if intent_result.intent == "clarify" else ""
        print(f"🤖 AI{tag}: {answer}\n")

        if intent_result.intent != "clarify":
            history.append("user", query)
            history.append("assistant", answer)


if __name__ == "__main__":
    main()
