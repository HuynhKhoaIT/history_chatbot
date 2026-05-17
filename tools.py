"""@tool decorator: auto-build OpenAI tool schema từ type hints + docstring,
auto-register vào registry, auto-dispatch. Không cần if/else thủ công.

Cách dùng:
    @tool
    def get_weather(city: str) -> str:
        '''Lấy thời tiết hiện tại của một thành phố. city là tên thành phố.'''
        ...

Sau đó:
    get_tool_schemas()        → list[dict] để truyền vào OpenAI tools=
    execute_tool(name, args)  → tự gọi đúng function
"""

import inspect
import json
from datetime import datetime, timezone, timedelta

# VN không áp dụng DST → offset cố định UTC+7, đúng bất kể host UTC hay local
VN_TZ = timezone(timedelta(hours=7))
from typing import Callable, get_type_hints

try:
    import requests
except ImportError:
    requests = None


# ============================================================
# Registry + decorator
# ============================================================

_REGISTRY: dict[str, dict] = {}

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(func: Callable) -> Callable:
    """Auto-build schema và đăng ký vào registry."""
    name = func.__name__
    doc = inspect.getdoc(func) or ""
    description = doc.split("\n\n")[0].strip() or name

    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties: dict[str, dict] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        py_type = hints.get(pname, str)
        json_type = _PY_TO_JSON.get(py_type, "string")
        properties[pname] = {"type": json_type, "description": f"Tham số {pname}"}
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    _REGISTRY[name] = {"func": func, "schema": schema}
    return func


def get_tool_schemas() -> list[dict]:
    return [t["schema"] for t in _REGISTRY.values()]


def execute_tool(name: str, args: dict) -> str:
    entry = _REGISTRY.get(name)
    if not entry:
        return f"Tool '{name}' không tồn tại"
    try:
        result = entry["func"](**args)
        return str(result)
    except Exception as e:
        return f"Lỗi khi gọi {name}: {e}"


# ============================================================
# Tools thật (real-time data)
# ============================================================

@tool
def get_current_time() -> str:
    """Lấy giờ Việt Nam hiện tại (UTC+7) dạng HH:MM ngày DD/MM/YYYY. Dùng khi user hỏi giờ giấc, ngày tháng hiện tại."""
    return datetime.now(VN_TZ).strftime("%H:%M ngày %d/%m/%Y") + " (giờ VN)"


@tool
def get_weather(city: str) -> str:
    """Lấy thời tiết hiện tại của một thành phố (real-time). Tham số city là tên thành phố không dấu, vd: Hanoi, Saigon, Danang, Tokyo. Dùng khi user hỏi về thời tiết, nhiệt độ, mưa nắng."""
    if requests is None:
        return "Không có thư viện requests, không thể lấy thời tiết."
    try:
        r = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "%l: %C, %t, độ ẩm %h, gió %w", "lang": "vi"},
            timeout=8,
        )
        if r.status_code != 200:
            return f"Không lấy được thời tiết của {city} (status {r.status_code})"
        return r.text.strip()
    except Exception as e:
        return f"Lỗi lấy thời tiết: {e}"


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Lấy tỷ giá hối đoái real-time giữa 2 đồng tiền. Tham số dùng mã ISO 3 ký tự: USD, VND, EUR, JPY, CNY, GBP... Dùng khi user hỏi tỷ giá, 1 USD bằng bao nhiêu VND."""
    if requests is None:
        return "Không có thư viện requests, không thể lấy tỷ giá."
    try:
        frm = from_currency.upper().strip()
        to = to_currency.upper().strip()
        r = requests.get(f"https://open.er-api.com/v6/latest/{frm}", timeout=8)
        data = r.json()
        if data.get("result") != "success":
            return f"Không lấy được tỷ giá cho {frm}."
        rates = data.get("rates", {})
        if to not in rates:
            return f"Không tìm thấy tỷ giá {frm} → {to}."
        rate = rates[to]
        return f"1 {frm} = {rate:,.4f} {to} (cập nhật {data.get('time_last_update_utc', 'N/A')})"
    except Exception as e:
        return f"Lỗi lấy tỷ giá: {e}"
