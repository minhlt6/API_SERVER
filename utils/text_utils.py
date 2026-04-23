import re

def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""
        
    # Nối các từ bị gãy ngang do xuống dòng 
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Loại bỏ các ký tự điều khiển không mong muốn, nhưng giữ lại các dấu câu thông thường
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Xóa các ký tự không nhìn thấy và các ký tự đặc biệt như zero-width space và BOM
    text = text.replace('\u200b', '').replace('\ufeff', '')
    
    # Chuẩn hóa khoảng trắng 
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    text = re.sub(r'\n +', '\n', text)
    
    # Giới hạn tối đa 2 dòng trống liên tiếp
    text = re.sub(r'\n{3,}', '\n\n', text) 
    
    # Sửa lỗi dư khoảng trắng trước dấu câu
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    return text.strip()