import re

def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""
        
    # Nối các từ bị gãy ngang do xuống dòng 
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # \| và < > vào để bảo vệ khung Bảng Markdown và các Placeholder
    text = re.sub(r'[^\w\s\.,;:!?\-$$\"\'\À-ỹ\n\|<>]', ' ', text)
    
    # Chuẩn hóa khoảng trắng 
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r'\n +', '\n', text)
    
    # Giới hạn tối đa 2 dòng trống liên tiếp
    text = re.sub(r'\n{3,}', '\n\n', text) 
    
    # Sửa lỗi dư khoảng trắng trước dấu câu
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    return text.strip()