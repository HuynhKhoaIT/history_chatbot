"""Verifier dùng chung cho RAG PDF và Wikipedia fallback."""

from pydantic import BaseModel, Field
from config import openai_client, CHAT_MODEL, FALLBACK_NO_INFO


class AnswerVerification(BaseModel):
    is_grounded: bool = Field(
        description="True nếu MỌI claim trong ANSWER đều được hỗ trợ bởi CONTEXT."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Các claim không có trong CONTEXT (trích nguyên văn).",
    )
    reasoning: str = Field(description="Giải thích ngắn 1-2 câu")


VERIFIER_PROMPT = f"""Bạn là verifier kiểm tra câu trả lời có "bịa" hay không.

NHIỆM VỤ: Xét từng claim trong ANSWER, kiểm tra có được hỗ trợ bởi CONTEXT không.

QUY TẮC:
- is_grounded=True CHỈ khi MỌI claim trong ANSWER có thể tìm được trong CONTEXT.
- Có claim bịa, suy luận, mở rộng, dùng kiến thức ngoài → is_grounded=False.
- Câu fallback "{FALLBACK_NO_INFO}" hoặc các biến thể tương đương → coi như is_grounded=True.
- Diễn đạt lại CONTEXT bằng từ khác → vẫn grounded.
- Thêm số liệu/tên/ngày tháng KHÔNG có trong CONTEXT → KHÔNG grounded."""


def verify_answer(query: str, answer: str, context_docs: list[dict]) -> AnswerVerification:
    context = "\n\n".join(
        [f"[Đoạn {i+1}]\n{doc['text']}" for i, doc in enumerate(context_docs)]
    )
    response = openai_client.beta.chat.completions.parse(
        model=CHAT_MODEL,
        temperature=0,
        response_format=AnswerVerification,
        messages=[
            {"role": "system", "content": VERIFIER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n---\n{context}\n---\n\n"
                    f"CÂU HỎI: {query}\n\n"
                    f"ANSWER cần kiểm tra:\n---\n{answer}\n---\n\n"
                    f"ANSWER có grounded với CONTEXT không?"
                ),
            },
        ],
    )
    return response.choices[0].message.parsed
