import re
from typing import List, Tuple
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

STRUCTURE_PATTERNS = [
    r"(?m)^\s*(Điều\s+\d+[\.:]?)",
    r"(?m)^\s*(Chương\s+[IVXLC\d]+[\.:]?)",
    r"(?m)^\s*(Mục\s+\d+[\.:]?)",
    r"(?m)^\s*(Khoản\s+\d+[\.:]?)",
    r"(?m)^\s*(Điểm\s+[a-zA-Z0-9]+[\.:]?)",
    r"(?m)^\s*(\d+(?:\.\d+)*[\)\.])",
    r"(?m)^\s*([a-zA-Z][\)\.])",
]

LIST_PATTERNS = [
    (r"(?m)^\s*a\.", "<LIST_A>"),
    (r"(?m)^\s*b\.", "<LIST_B>"),
    (r"(?m)^\s*c\.", "<LIST_C>"),
    (r"(?m)^\s*\d+\.", "<LIST_NUM>"),
    (r"(?m)^\s*\d+\)", "<LIST_NUM_PAREN>"),
    (r"(?m)^\s*-\s+", "<LIST_DASH>"),
    (r"(?m)^\s*•\s+", "<LIST_BULLET>"),
]

# Tách và thêm các thẻ <table> để bảo vệ cấu trúc bảng khỏi bị chia cắt trong quá trình chunking.
def extract_and_protect_tables(text: str) -> Tuple[str, dict]:
    table_pattern = re.compile(r"(?:\|.*\|[\r\n]+)+")
    tables = {}

    def replace_table(match):
        table_id = f"<TABLE_{len(tables)}>"
        tables[table_id] = match.group(0)
        return f"\n{table_id}\n"

    protected_text = re.sub(table_pattern, replace_table, text)
    return protected_text, tables

# Bảo vệ các phần tử của danh sách khỏi bị chia cắt trong quá trình chunking
def protect_lists(text: str) -> Tuple[str, dict]:
    placeholders = {}
    protected = text

    for pattern, token in LIST_PATTERNS:
        matches = list(re.finditer(pattern, protected))
        for index, match in enumerate(matches):
            placeholder = f"{token}_{index}"
            placeholders[placeholder] = match.group(0)
            protected = protected.replace(match.group(0), placeholder, 1)

    return protected, placeholders

# Khôi phục các phần từ được bảo vệ về nội dung gốc bằng cách thay thế các placeholder 
def restore_placeholders(text: str, placeholders: dict) -> str:
    restored = text
    for placeholder, original in placeholders.items():
        restored = restored.replace(placeholder, original)
    return restored

# Tách văn bản dựa trên cấu trúc được xây dựng từ đầu 
def split_by_structure(text: str) -> List[str]:
    parts = [text]

    for pattern in STRUCTURE_PATTERNS:
        next_parts = []
        for part in parts:
            matches = list(re.finditer(pattern, part))
            if len(matches) <= 1:
                next_parts.append(part)
                continue

            last_pos = 0
            for idx, match in enumerate(matches):
                start = match.start()
                if idx > 0 and start > last_pos:
                    chunk = part[last_pos:start].strip()
                    if chunk:
                        next_parts.append(chunk)
                last_pos = start

            tail = part[last_pos:].strip()
            if tail:
                next_parts.append(tail)

        parts = next_parts or parts

    return [part for part in parts if part.strip()]

# Hàm chính thực hiện chunking thông minh 
def smart_chunking(docs: List) -> List:
    logger.info("Chunking theo cau truc + do dai...")
    length_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = []

    for doc in docs:
        text = doc.page_content or ""
        protected_text, tables = extract_and_protect_tables(text)
        protected_text, list_placeholders = protect_lists(protected_text)

        structure_parts = split_by_structure(protected_text)

        for part in structure_parts:
            if len(part) <= CHUNK_SIZE:
                sub_parts = [part]
            else:
                sub_parts = length_splitter.split_text(part)

            for chunk_text in sub_parts:
                restored = restore_placeholders(chunk_text, list_placeholders)
                restored = restore_placeholders(restored, tables)

                if not restored.strip():
                    continue

                new_doc = type(doc)(
                    page_content=restored.strip(),
                    metadata=doc.metadata.copy() if isinstance(doc.metadata, dict) else {},
                )
                chunks.append(new_doc)

    logger.info("Da tao %s chunks.", len(chunks))
    return chunks