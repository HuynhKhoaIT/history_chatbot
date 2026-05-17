"""Chat history với token counting + sliding window + auto-summarization.

Kỹ thuật mượn từ prompts/10_multi_turn.py:
- Đếm token bằng tiktoken
- Giữ N message gần đây
- Khi vượt MAX_HISTORY_TOKENS → tóm tắt phần cũ thành 1 đoạn, xóa raw

Khác với ChatBot ở 10_multi_turn: ở đây history KHÔNG gắn cứng 1 system prompt
vì mỗi route (chitchat / RAG / intent) dùng system prompt khác nhau. History chỉ
giữ user/assistant pairs + summary; caller build messages cuối cùng.
"""

import tiktoken

from config import (
    openai_client,
    CHAT_MODEL,
    MAX_HISTORY_TOKENS,
    HISTORY_KEEP_RECENT,
    SUMMARIZE_TRIGGER_EXTRA,
)


# tiktoken chưa chắc biết model id ta dùng → fallback cl100k_base
def _get_encoder():
    try:
        return tiktoken.encoding_for_model(CHAT_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


_ENCODER = _get_encoder()


def count_tokens(messages: list[dict]) -> int:
    """Đếm tokens xấp xỉ trong messages list."""
    total = 0
    for msg in messages:
        total += 4  # overhead mỗi message
        for v in msg.values():
            total += len(_ENCODER.encode(str(v)))
    return total


def summarize_messages(messages: list[dict]) -> str:
    """Tóm tắt một đoạn history thành 3-4 câu, giữ thông tin quan trọng."""
    text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        max_completion_tokens=300,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tóm tắt đoạn hội thoại thành 3-4 câu tiếng Việt, "
                    "giữ thông tin quan trọng (chủ đề, thực thể, câu hỏi đã hỏi, kết luận)."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return (response.choices[0].message.content or "").strip()


class ChatHistory:
    """In-memory chat history với auto-summarization theo token budget."""

    def __init__(
        self,
        max_tokens: int = MAX_HISTORY_TOKENS,
        keep_recent: int = HISTORY_KEEP_RECENT,
    ):
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.history: list[dict] = []
        self.summary: str = ""

    # --------- API public ---------

    def append(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self._maybe_summarize()

    def context(self, system_prompt: str | None = None) -> list[dict]:
        """Build messages cho LLM: [system] + [summary] + recent history."""
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        if self.summary:
            msgs.append(
                {
                    "role": "system",
                    "content": f"[Tóm tắt hội thoại trước]: {self.summary}",
                }
            )
        msgs.extend(self.history[-self.keep_recent:])
        return msgs

    def recent(self, last_n: int | None = None) -> list[dict]:
        """Lấy N message cuối, không kèm system prompt (dùng cho intent classifier)."""
        n = last_n if last_n is not None else self.keep_recent
        return self.history[-n:]

    # --------- Internal ---------

    def _maybe_summarize(self) -> None:
        """Khi vượt token budget HOẶC history quá dài → summarize phần cũ."""
        if len(self.history) <= self.keep_recent + SUMMARIZE_TRIGGER_EXTRA:
            # chưa đủ dài để cần summarize, chỉ check token
            if count_tokens(self.context()) <= self.max_tokens:
                return

        old = self.history[:-self.keep_recent]
        if not old:
            return

        new_summary = summarize_messages(old)
        if self.summary:
            self.summary = f"{self.summary} Tiếp theo: {new_summary}"
        else:
            self.summary = new_summary

        self.history = self.history[-self.keep_recent:]
        print(f"  [📝 Summarized {len(old)} messages cũ]")
