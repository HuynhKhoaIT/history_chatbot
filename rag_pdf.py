"""RAG từ PDF: retrieve từ Pinecone → generate → verify."""

from config import openai_client, embed_text, CHAT_MODEL, TOP_K, MIN_RAG_SCORE, FALLBACK_NO_INFO
from verifier import verify_answer


SYSTEM_PROMPT_RAG = f"""Bạn là trợ lý trả lời câu hỏi lịch sử Việt Nam DỰA HOÀN TOÀN trên CONTEXT.

QUY TẮC NGHIÊM NGẶT:
1. CHỈ dùng thông tin trong CONTEXT. KHÔNG dùng kiến thức ngoài, không suy đoán.
2. CONTEXT không đủ trả lời → nói NGUYÊN VĂN: "{FALLBACK_NO_INFO}"
3. KHÔNG bịa số liệu, ngày tháng, tên riêng nếu không có trong CONTEXT.
4. KHÔNG dùng "có lẽ", "tôi nghĩ", "thường thì".
5. Trả lời NGẮN GỌN, đi thẳng vào ý."""


def retrieve(index, query: str, top_k: int = TOP_K, category: str | None = None) -> list[dict]:
    query_embedding = embed_text(query)
    query_kwargs = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
    }
    if category:
        query_kwargs["filter"] = {"category": {"$eq": category}}
    results = index.query(**query_kwargs)
    return [
        {
            "text": m["metadata"]["text"],
            "score": m["score"],
            "page": m["metadata"].get("page", -1),
            "category": m["metadata"].get("category"),
        }
        for m in results.matches
    ]


def generate_answer(query: str, context_docs: list[dict]) -> str:
    context = "\n\n".join(
        [f"[Đoạn {i+1}]\n{doc['text']}" for i, doc in enumerate(context_docs)]
    )
    user_prompt = (
        f"CONTEXT:\n---\n{context}\n---\n\n"
        f"CÂU HỎI: {query}\n\n"
        f"Trả lời CHỈ dựa trên CONTEXT. Nếu không có thông tin, "
        f'nói nguyên văn: "{FALLBACK_NO_INFO}"'
    )
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=400,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def rag_pdf_query(
    index, query: str, category: str | None = None, verbose: bool = True
) -> tuple[str, bool, float]:
    """
    Trả về (answer, is_grounded, top_score).
    Nếu category != None: filter Pinecone theo category. Khi không tìm thấy chunk nào,
    tự retry không filter (broaden search) trước khi báo fallback.
    """
    context_docs = retrieve(index, query, category=category)
    if category and not context_docs:
        if verbose:
            print(f"  ↩️  Không có chunk cho category={category}, retry không filter...")
        context_docs = retrieve(index, query, category=None)

    top_score = context_docs[0]["score"] if context_docs else 0.0

    if verbose:
        cat_tag = f" [filter={category}]" if category else ""
        print(f"  📚 PDF retrieved {len(context_docs)} chunks{cat_tag} (top score: {top_score:.4f})")
        for i, doc in enumerate(context_docs[:2]):
            print(f"     [{i+1}] {doc['score']:.4f} | {doc['text'][:80]}...")

    if not context_docs or top_score < MIN_RAG_SCORE:
        if verbose:
            print(f"  ⚠️  Score thấp (<{MIN_RAG_SCORE}) → cần fallback")
        return FALLBACK_NO_INFO, False, top_score

    answer = generate_answer(query, context_docs)

    # nếu generate trả về fallback message thì coi như không tìm thấy
    if FALLBACK_NO_INFO in answer:
        return answer, False, top_score

    verification = verify_answer(query, answer, context_docs)
    if verbose:
        print(f"  🔍 PDF verify: grounded={verification.is_grounded} | {verification.reasoning}")

    return answer, verification.is_grounded, top_score
