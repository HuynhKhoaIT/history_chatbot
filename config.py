"""Config dùng chung: env, clients, constants."""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pinecone
INDEX_NAME = "vn-history"
EMBEDDING_DIM = 1024
EMBEDDING_MODEL = "text-embedding-3-small"

# Model
CHAT_MODEL = "gpt-5.4-nano"

# RAG params
TOP_K = 5
MIN_RAG_SCORE = 0.5

# Input
MAX_INPUT_LEN = 500

# History (token-based, theo style 10_multi_turn)
MAX_HISTORY_TOKENS = 4000        # vượt → auto-summarize phần cũ
HISTORY_KEEP_RECENT = 6          # số message gần nhất giữ raw
SUMMARIZE_TRIGGER_EXTRA = 4      # history > keep_recent + extra → ép summarize

# Fallback message
FALLBACK_NO_INFO = "Tôi không tìm thấy thông tin chính xác về câu hỏi này."

# Categories (theo thời kỳ lịch sử VN). Khớp với tên subfolder trong data/.
CATEGORIES: dict[str, str] = {
    "co_dai": "Cổ đại: từ Văn Lang - Âu Lạc đến trước thế kỷ 10 (Bắc thuộc).",
    "phong_kien": "Phong kiến độc lập: Ngô, Đinh, Tiền Lê, Lý, Trần, Hồ, Lê, Mạc, Trịnh-Nguyễn, Tây Sơn, Nguyễn (đến 1858).",
    "phap_thuoc": "Pháp thuộc: 1858 - 1945, kháng chiến chống Pháp đến CMT8.",
    "hien_dai": "Hiện đại: 1945 đến nay, kháng chiến chống Pháp/Mỹ, thống nhất, đổi mới.",
}


def ensure_index():
    """Tạo index nếu chưa có. Gọi 1 lần khi start."""
    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"✅ Created Pinecone index '{INDEX_NAME}'")
    else:
        print(f"✅ Index '{INDEX_NAME}' đã tồn tại")
    return pc.Index(INDEX_NAME)


def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIM,
    )
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIM,
    )
    return [item.embedding for item in response.data]
