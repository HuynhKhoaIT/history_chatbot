"""Fallback Wikipedia tiếng Việt qua langchain WikipediaRetriever.

Pipeline:
  query → extract_search_keyword (strip fluff) → WikipediaRetriever (top_k=6) → LLM answer
"""

import wikipedia
from pydantic import BaseModel, Field
from langchain_community.retrievers import WikipediaRetriever

from config import openai_client, CHAT_MODEL, FALLBACK_NO_INFO
from verifier import verify_answer


# Wikimedia bắt buộc User-Agent rõ ràng (tên app/version/contact),
# nếu không sẽ trả empty body → json.loads fail.
wikipedia.set_user_agent("rag-history-vn/1.0 (huynhkhoa.dev@gmail.com)")


_retriever = WikipediaRetriever(
    lang="vi",
    top_k_results=6,
    doc_content_chars_max=2000,
)


class WikiSearchQuery(BaseModel):
    keyword: str = Field(
        description=(
            "Cụm từ khóa chính để search Wikipedia, ĐÃ loại bỏ từ nghi vấn "
            "('là gì', 'ở đâu', 'khi nào', 'thời gian nào', 'diễn ra ra sao'...) "
            "và từ phụ. Giữ entity / tên riêng / sự kiện. "
            "Vd: 'Lăng Khải Định nằm ở đâu?' → 'Lăng Khải Định'. "
            "'Trận Điện Biên Phủ trên không diễn ra thời gian nào' → 'Điện Biên Phủ trên không'."
        )
    )


def extract_search_keyword(query: str) -> str:
    """LLM rút keyword chính để search Wikipedia hiệu quả hơn."""
    response = openai_client.beta.chat.completions.parse(
        model=CHAT_MODEL,
        temperature=0,
        response_format=WikiSearchQuery,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn rút keyword chính từ câu hỏi để search Wikipedia. "
                    "Bỏ từ nghi vấn và từ phụ, giữ tên riêng/sự kiện."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    kw = (response.choices[0].message.parsed.keyword or "").strip()
    return kw or query  # fallback nếu rút rỗng


SYSTEM_PROMPT_WIKI = f"""Bạn là trợ lý trả lời câu hỏi lịch sử Việt Nam DỰA HOÀN TOÀN trên các đoạn Wikipedia được cung cấp.

QUY TẮC:
1. CHỈ dùng thông tin trong WIKIPEDIA_CONTEXT. KHÔNG dùng kiến thức ngoài.
2. Nếu WIKIPEDIA_CONTEXT không liên quan / không đủ → nói NGUYÊN VĂN: "{FALLBACK_NO_INFO}"
3. KHÔNG bịa số liệu, ngày tháng, tên riêng.
4. Trả lời NGẮN GỌN. KHÔNG tự thêm tag nguồn — caller sẽ append sau khi verify."""


def search_wiki(search_term: str) -> list[dict]:
    """Search Wikipedia VN, trả về list dict cùng format với rag_pdf."""
    try:
        docs = _retriever.invoke(search_term)
    except Exception as e:
        print(f"  ⚠️  Wikipedia error: {e}")
        return []
    return [
        {
            "text": d.page_content,
            "title": d.metadata.get("title", "N/A"),
            "source": d.metadata.get("source", "wikipedia"),
        }
        for d in docs
    ]


def generate_from_wiki(query: str, wiki_docs: list[dict]) -> str:
    context = "\n\n".join(
        [f"[{d['title']}]\n{d['text']}" for d in wiki_docs]
    )
    user_prompt = (
        f"WIKIPEDIA_CONTEXT:\n---\n{context}\n---\n\n"
        f"CÂU HỎI: {query}\n\n"
        f"Trả lời CHỈ dựa trên WIKIPEDIA_CONTEXT. Nếu không có thông tin, "
        f'nói nguyên văn: "{FALLBACK_NO_INFO}"'
    )
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_WIKI},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=400,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def wiki_query(query: str, verbose: bool = True) -> tuple[str, bool]:
    """
    Trả về (answer, is_grounded).
    is_grounded=False khi không tìm thấy hoặc verifier bắt hallucination.
    """
    keyword = extract_search_keyword(query)
    if verbose:
        print(f"  🔑 Wiki keyword: '{keyword}'")
    wiki_docs = search_wiki(keyword)

    if verbose:
        print(f"  🌐 Wiki tìm thấy {len(wiki_docs)} trang")
        for d in wiki_docs[:2]:
            print(f"     - {d['title']}: {d['text'][:80]}...")

    if not wiki_docs:
        return FALLBACK_NO_INFO, False

    answer = generate_from_wiki(query, wiki_docs)

    if FALLBACK_NO_INFO in answer:
        return answer, False

    # verifier dùng cùng format dict
    verification = verify_answer(query, answer, wiki_docs)
    if verbose:
        print(f"  🔍 Wiki verify: grounded={verification.is_grounded} | {verification.reasoning}")

    if verification.is_grounded:
        answer = f"{answer}\n(Nguồn: Wikipedia)"
    return answer, verification.is_grounded
