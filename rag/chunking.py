from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import CHUNK_SIZE, CHUNK_OVERLAP

def smart_chunking(docs: List) -> List:
    print("Đang áp dụng Smart Chunking (Regex Lookahead)...")
    
    # Cấu hình Regex bắt cấu trúc phân cấp hành chính (Chương -> Điều -> Khoản)
    legal_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\nChương ",      
            "\nĐiều ",        
            "\nKhoản ",       
            "\n\n",           
            r"\n(?=\d+\.)",   
            r"\n(?=[a-z]\.)", 
            r"\n(?=-|\+)",    
            "\n", " ", ""
        ],
        length_function=len,
        is_separator_regex=True  
    )
    
    chunks = []
    for doc in docs:
        doc_chunks = legal_splitter.split_text(doc.page_content)
        
        for chunk_text in doc_chunks:
            new_doc = type(doc)(
                page_content=chunk_text,
                metadata=doc.metadata.copy()  
            )
            chunks.append(new_doc)
            
    print(f"Đã tạo {len(chunks)} chunks thông minh.")
    return chunks