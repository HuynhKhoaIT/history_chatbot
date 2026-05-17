"""Load PDF lịch sử VN → classify từng page → chunk → upsert vào Pinecone.

Flow:
    data/*.pdf
      → load (PyPDFLoader, 1 doc / page)
      → classify_pages: LLM gán category cho từng page (batch 10 page/call)
      → chunk_documents: chunks inherit category từ page nó nằm
      → save_to_pinecone: upsert kèm metadata.category
"""

import os
import glob
from typing import Literal

from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    ensure_index,
    embed_batch,
    openai_client,
    CHAT_MODEL,
    INDEX_NAME,
    CATEGORIES,
)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
PAGE_SNIPPET_CHARS = 400      # số ký tự đầu mỗi page gửi cho classifier
CLASSIFY_BATCH_SIZE = 10      # số page mỗi LLM call


CategoryLiteral = Literal["co_dai", "phong_kien", "phap_thuoc", "hien_dai"]


# ============================================================
# Discover & load
# ============================================================

def discover_pdfs(data_dir: str = DATA_DIR) -> list[str]:
    """Trả về list path mọi PDF trong data_dir (recursive 1 cấp)."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    paths += sorted(glob.glob(os.path.join(data_dir, "*", "*.pdf")))
    return paths


def load_pdf(pdf_path: str) -> list:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    docs = PyPDFLoader(pdf_path).load()
    print(f"  ✅ Loaded {len(docs)} pages từ {pdf_path}")
    return docs


# ============================================================
# Classify pages → category
# ============================================================

class PageClassification(BaseModel):
    page_num: int = Field(description="Page index nguyên văn từ input.")
    category: CategoryLiteral = Field(
        description="Một trong 4 category dựa vào nội dung page snippet."
    )


class BatchPageClassification(BaseModel):
    classifications: list[PageClassification]


def _categories_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())


def _classify_batch(snippets: list[tuple[int, str]]) -> dict[int, str]:
    """1 LLM call cho 1 batch page → dict {page_num: category}."""
    pages_text = "\n\n".join(
        f"[page_num={p}]\n{snippet}" for p, snippet in snippets
    )
    system = (
        "Bạn là classifier phân loại page sách lịch sử Việt Nam vào 1 trong 4 thời kỳ.\n\n"
        f"CATEGORIES:\n{_categories_block()}\n\n"
        "QUY TẮC: Mỗi page đầu vào → đúng 1 category. "
        "Trả về MỌI page_num đã input. Dựa vào tên triều đại / mốc năm / sự kiện trong snippet."
    )
    response = openai_client.beta.chat.completions.parse(
        model=CHAT_MODEL,
        temperature=0,
        response_format=BatchPageClassification,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": pages_text},
        ],
    )
    parsed = response.choices[0].message.parsed
    return {c.page_num: c.category for c in parsed.classifications}


def classify_pages(docs: list, verbose: bool = True) -> dict[int, str]:
    """LLM gán category cho từng page. Batch để tiết kiệm call."""
    snippets = [
        (d.metadata.get("page", i), (d.page_content or "")[:PAGE_SNIPPET_CHARS])
        for i, d in enumerate(docs)
    ]
    # bỏ page rỗng
    snippets = [(p, s) for p, s in snippets if s.strip()]

    page_to_cat: dict[int, str] = {}
    for i in range(0, len(snippets), CLASSIFY_BATCH_SIZE):
        batch = snippets[i : i + CLASSIFY_BATCH_SIZE]
        result = _classify_batch(batch)
        page_to_cat.update(result)
        if verbose:
            cats = sorted(set(result.values()))
            print(
                f"  🏷️  Classified pages {batch[0][0]}–{batch[-1][0]} "
                f"({len(batch)} pages) → categories: {cats}"
            )

    if verbose:
        from collections import Counter
        dist = Counter(page_to_cat.values())
        print(f"  📊 Page distribution: {dict(dist)}")
    return page_to_cat


# ============================================================
# Chunk + tag
# ============================================================

def chunk_documents(
    docs: list,
    page_to_category: dict[int, str],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list:
    """Split + ghi metadata.category cho từng chunk theo page nó thuộc."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    skipped = 0
    for c in chunks:
        page = c.metadata.get("page", -1)
        cat = page_to_category.get(page)
        if cat is None:
            skipped += 1
            continue
        c.metadata["category"] = cat

    avg = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
    tagged = len(chunks) - skipped
    print(
        f"  ✅ Chunked thành {len(chunks)} chunks (avg {avg} ký tự, "
        f"tagged {tagged}/{len(chunks)})"
    )
    return chunks


# ============================================================
# Save to Pinecone
# ============================================================

def save_to_pinecone(chunks: list, source_pdf: str, index, id_offsets: dict[str, int]) -> int:
    """Upsert chunks. ID prefix theo category đã gán trong chunk.metadata.
    id_offsets: dict per-category counter (mutated)."""
    valid = [c for c in chunks if c.metadata.get("category")]
    if not valid:
        print("  ⚠️  Không có chunk nào có category, skip.")
        return 0

    ids: list[str] = []
    for c in valid:
        cat = c.metadata["category"]
        ids.append(f"vn-history-{cat}-{id_offsets[cat]}")
        id_offsets[cat] += 1

    texts = [c.page_content for c in valid]
    embeddings = embed_batch(texts)
    metadatas = [
        {
            "text": c.page_content,
            "page": c.metadata.get("page", -1),
            "source": c.metadata.get("source", source_pdf),
            "category": c.metadata["category"],
        }
        for c in valid
    ]
    BATCH = 100
    for i in range(0, len(ids), BATCH):
        index.upsert(
            vectors=list(zip(ids[i : i + BATCH], embeddings[i : i + BATCH], metadatas[i : i + BATCH]))
        )
    print(f"  ✅ Saved {len(valid)} chunks vào '{INDEX_NAME}'")
    return len(valid)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    index = ensure_index()
    pdfs = discover_pdfs()
    if not pdfs:
        print(f"⚠️  Không tìm thấy PDF nào trong {DATA_DIR}/")
        raise SystemExit(1)

    print(f"📂 Tìm thấy {len(pdfs)} file PDF:")
    for p in pdfs:
        print(f"   - {p}")

    id_offsets: dict[str, int] = {c: 0 for c in CATEGORIES}
    total = 0
    for pdf_path in pdfs:
        print(f"\n→ {pdf_path}")
        docs = load_pdf(pdf_path)
        page_to_cat = classify_pages(docs)
        chunks = chunk_documents(docs, page_to_cat)
        n = save_to_pinecone(chunks, pdf_path, index, id_offsets)
        total += n

    print(f"\n🎉 Done. Tổng cộng {total} chunks đã ingest.")
    print(f"   ID counter per category: {id_offsets}")
