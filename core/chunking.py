import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def extract_and_protect_tables(text: str) -> tuple[str, dict]:
    """Tìm và bọc các bảng Markdown để bảo vệ chúng khỏi việc bị cắt gãy."""
    # Pattern tìm bảng Markdown (các dòng bắt đầu và chứa ký tự | liên tiếp)
    table_pattern = re.compile(r'(?:\|.*\|[\r\n]+)+')
    tables = {}
    
    def replace_table(match):
        table_id = f"<TABLE_{len(tables)}>"
        tables[table_id] = match.group(0)
        return f"\n{table_id}\n"

    protected_text = re.sub(table_pattern, replace_table, text)
    return protected_text, tables

def smart_chunking(docs: List) -> List:
    print("Đang áp dụng Smart Chunking (Bảo toàn Bảng & Danh sách)...")
    legal_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\nĐiều ", "\nChương ", "\nMục ", "\nKhoản ",
            "\n\n", "\n", ". ", " ", ""
        ],
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = []
    for doc in docs:
        # 1. Bảo vệ List đang có
        protected_text = doc.page_content.replace('\na.', '<LIST_a>') \
                                         .replace('\nb.', '<LIST_b>') \
                                         .replace('\nc.', '<LIST_c>')
        
        # 2. Bảo vệ Table
        protected_text, tables = extract_and_protect_tables(protected_text)
        
        # 3. Tiến hành cắt
        doc_chunks = legal_splitter.split_text(protected_text)
        
        # 4. Phục hồi dữ liệu
        for chunk_text in doc_chunks:
            restored = chunk_text.replace('<LIST_a>', '\na.') \
                                 .replace('<LIST_b>', '\nb.') \
                                 .replace('<LIST_c>', '\nc.')
            
            for table_id, table_content in tables.items():
                if table_id in restored:
                    restored = restored.replace(table_id, table_content)
                    
            new_doc = type(doc)(
                page_content=restored,
                metadata=doc.metadata.copy()
            )
            chunks.append(new_doc)
            
    print(f" Đã tạo {len(chunks)} chunks thông minh (giữ nguyên cấu trúc bảng)")
    return chunks