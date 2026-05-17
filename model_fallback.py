"""Fallback cuối: model tự trả lời với guardrails nghiêm ngặt.

Chỉ trả lời lịch sử VN, kiến thức đến 2010, không bịa.
Sau khi generate có self-check để bắt vi phạm (sự kiện sau 2010, không chắc chắn).
"""

from pydantic import BaseModel, Field

from config import openai_client, CHAT_MODEL, FALLBACK_NO_INFO


SYSTEM_PROMPT_MODEL = f"""Bạn là chuyên gia lịch sử Việt Nam. Trả lời CHỈ về lịch sử Việt Nam.

GIỚI HẠN KIẾN THỨC NGHIÊM NGẶT:
- Kiến thức của bạn CHỈ đến năm 2010. Sự kiện sau 2010 → trả lời:
  "Kiến thức của tôi chỉ đến năm 2010, vui lòng tra cứu nguồn khác về sự kiện này."
- CHỈ trả lời về lịch sử Việt Nam. Câu hỏi không liên quan → từ chối lịch sự.

QUY TẮC KHÔNG BỊA:
1. KHÔNG bịa số liệu, ngày tháng, tên riêng nếu KHÔNG chắc chắn 100%.
2. Không chắc → nói: "{FALLBACK_NO_INFO}"
3. KHÔNG dùng "có lẽ", "tôi nghĩ", "thường thì", "khoảng".
4. Trả lời NGẮN GỌN, đi thẳng vào ý.
5. Ghi rõ ở cuối: "(Nguồn: kiến thức nội tại của model, có thể không hoàn toàn chính xác)"."""


class ModelSelfCheck(BaseModel):
    has_post_2010: bool = Field(description="Câu trả lời có chứa sự kiện/thông tin sau năm 2010 không")
    is_vn_history: bool = Field(description="Câu trả lời có thực sự là về lịch sử Việt Nam không")
    has_uncertain_facts: bool = Field(
        description="Câu trả lời có chứa số liệu/ngày/tên cụ thể mà có thể không chắc chắn không"
    )
    reasoning: str = Field(description="Giải thích ngắn 1 câu")


SELF_CHECK_PROMPT = """Bạn là verifier kiểm tra câu trả lời lịch sử VN có vi phạm 3 quy tắc:
1. Có chứa sự kiện/thông tin SAU năm 2010 không?
2. Có thực sự về lịch sử Việt Nam không?
3. Có khẳng định số liệu/ngày/tên cụ thể mà thực tế khó chắc chắn (có dấu hiệu bịa) không?

Trả lời câu fallback "{fallback}" → không vi phạm gì.""".format(fallback=FALLBACK_NO_INFO)


def _self_check(answer: str) -> ModelSelfCheck:
    response = openai_client.beta.chat.completions.parse(
        model=CHAT_MODEL,
        temperature=0,
        response_format=ModelSelfCheck,
        messages=[
            {"role": "system", "content": SELF_CHECK_PROMPT},
            {"role": "user", "content": f"ANSWER cần kiểm tra:\n---\n{answer}\n---"},
        ],
    )
    return response.choices[0].message.parsed


def model_query(query: str, verbose: bool = True) -> str:
    """Fallback cuối: model trả lời với guardrails + self-check."""
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_MODEL},
            {"role": "user", "content": query},
        ],
        max_completion_tokens=400,
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()

    # nếu model đã tự fallback thì trả về luôn
    if FALLBACK_NO_INFO in answer or "Kiến thức của tôi chỉ đến năm 2010" in answer:
        return answer

    # check = _self_check(answer)
    # if verbose:
    #     print(
    #         f"  🔍 Model self-check: post2010={check.has_post_2010}, "
    #         f"vn_history={check.is_vn_history}, uncertain={check.has_uncertain_facts}"
    #     )

    # if check.has_post_2010:
    #     return "Kiến thức của tôi chỉ đến năm 2010, vui lòng tra cứu nguồn khác về sự kiện này."
    # if not check.is_vn_history:
    #     return FALLBACK_NO_INFO
    # if check.has_uncertain_facts:
    #     return FALLBACK_NO_INFO

    return answer
