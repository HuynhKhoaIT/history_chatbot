"""Classify intent: là câu hỏi lịch sử VN, chitchat, hay cần làm rõ.

Khi intent='vn_history' thì còn classify thêm category (thời kỳ) để filter Pinecone.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from config import openai_client, CHAT_MODEL, CATEGORIES


CategoryLiteral = Literal["co_dai", "phong_kien", "phap_thuoc", "hien_dai"]


DOMAIN_RULES = """
Hệ thống CHỈ chuyên về LỊCH SỬ VIỆT NAM:
- Các triều đại (Lý, Trần, Lê, Nguyễn...), vua chúa, nhân vật lịch sử VN
- Sự kiện lịch sử VN: chiến tranh, khởi nghĩa, hiệp định, ngày lễ
- Văn hóa - chính trị - xã hội VN trong các giai đoạn lịch sử

KHÔNG thuộc phạm vi:
- Lịch sử nước khác (Trung Quốc, Mỹ, Nhật...)
- Thời sự / tin tức hiện tại
- Các chủ đề không phải lịch sử (nấu ăn, công nghệ, thể thao...)
"""


def _categories_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())


class QueryIntent(BaseModel):
    intent: Literal["vn_history", "chitchat", "clarify", "out_of_scope"] = Field(
        description=(
            "vn_history: hỏi về lịch sử Việt Nam | "
            "chitchat: chào hỏi, hỏi giờ, câu hỏi chung không phải lịch sử | "
            "clarify: câu hỏi mơ hồ, cần làm rõ | "
            "out_of_scope: hỏi lịch sử nhưng KHÔNG phải VN (vd: lịch sử TQ)"
        )
    )
    category: Optional[CategoryLiteral] = Field(
        default=None,
        description=(
            "CHỈ điền khi intent='vn_history' VÀ chắc chắn câu hỏi thuộc 1 thời kỳ. "
            "Để None nếu mơ hồ / xuyên nhiều thời kỳ → khi đó sẽ search toàn bộ."
        ),
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="Câu hỏi làm rõ bằng tiếng Việt. CHỈ điền khi intent='clarify'.",
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description=(
            "CHỈ điền khi intent='vn_history' VÀ query gốc có đại từ/tham chiếu cần resolve "
            "từ HISTORY (vd: 'ông ấy', 'nó', 'lúc đó', 'trận đó'). "
            "Viết lại thành câu standalone đầy đủ entity. "
            "Vd: history nói về Trần Hưng Đạo, query 'Ông sinh năm nào?' "
            "→ 'Trần Hưng Đạo sinh năm nào?'. "
            "Để None nếu query đã standalone."
        ),
    )
    reasoning: str = Field(description="Giải thích ngắn (1 câu) lý do chọn intent + category")
    confidence: float = Field(ge=0.0, le=1.0, description="Độ tin cậy 0.0-1.0")


def classify_intent(query: str, history_msgs: list[dict]) -> QueryIntent:
    system_prompt = f"""Bạn là intent classifier cho hệ thống RAG về lịch sử Việt Nam.

{DOMAIN_RULES}

4 INTENTS:
1. vn_history: Câu hỏi rõ ràng về lịch sử Việt Nam → sẽ đi qua RAG/Wikipedia/Model.
2. chitchat: Chào hỏi, hỏi giờ, hỏi bản thân hệ thống, câu hỏi thông thường không phải lịch sử.
3. clarify: Câu hỏi mơ hồ (đại từ "nó", "cái đó" mà history không giải thích được), thiếu thông tin.
   PHẢI điền clarifying_question bằng tiếng Việt.
4. out_of_scope: Câu hỏi RÕ về lịch sử nhưng KHÔNG phải Việt Nam (vd: lịch sử Trung Quốc, Pháp).

CATEGORIES (chỉ áp dụng khi intent='vn_history'):
{_categories_block()}

QUY TẮC CATEGORY:
- Để None nếu câu hỏi xuyên nhiều thời kỳ ("tóm tắt lịch sử VN") hoặc không xác định được.
- Chỉ điền khi tự tin: vd "Trần Hưng Đạo đánh Nguyên" → phong_kien; "Điện Biên Phủ" → hien_dai.
- "Hai Bà Trưng", "An Dương Vương" → co_dai (Bắc thuộc + trước Bắc thuộc).
- "Phan Bội Châu", "Đông Dương Đại hội" → phap_thuoc.

QUY TẮC CHUNG:
- Dùng HISTORY để giải nghĩa đại từ tham chiếu. Nếu history rõ → KHÔNG cần clarify.
- confidence < 0.6 → ưu tiên 'clarify'.
- Câu hỏi liên quan VN nhưng KHÔNG phải lịch sử (vd: "Hà Nội có gì chơi?") → chitchat, KHÔNG phải vn_history.

QUY TẮC REWRITE (rewritten_query):
- Khi intent='vn_history' và query có đại từ/tham chiếu mà HISTORY giải được:
  điền rewritten_query là câu standalone (thay đại từ bằng entity từ history).
- Mục đích: dùng cho RAG retrieval, vì retrieval chỉ thấy query không thấy history.
- Query đã đầy đủ entity → để rewritten_query=None."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_msgs)
    messages.append({"role": "user", "content": query})

    response = openai_client.beta.chat.completions.parse(
        model=CHAT_MODEL,
        messages=messages,
        response_format=QueryIntent,
        temperature=0,
    )
    return response.choices[0].message.parsed
