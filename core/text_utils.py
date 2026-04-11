import re

#Compile regex patterns một lần toàn cục - tránh recompile mỗi lần gọi
_HYPHENATED_WORD_PATTERN = re.compile(r'(\w+)-\s*\n\s*(\w+)')
_INVALID_CHARS_PATTERN = re.compile(r'[^\w\s\.,;:!?\-$$\"\'\À-ỹ\n\|<>]')
_MULTIPLE_SPACES_PATTERN = re.compile(r'[ \t]+')
_SPACE_BEFORE_NEWLINE_PATTERN = re.compile(r' +\n')
_SPACE_AFTER_NEWLINE_PATTERN = re.compile(r'\n +')
_MULTIPLE_NEWLINES_PATTERN = re.compile(r'\n{3,}')
_SPACE_BEFORE_PUNCTUATION_PATTERN = re.compile(r'\s+([.,;:!?])')


def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""
        
    # Nối các từ bị gãy ngang do xuống dòng 
    text = _HYPHENATED_WORD_PATTERN.sub(r'\1\2', text)
    
    # \| và < > vào để bảo vệ khung Bảng Markdown và các Placeholder
    text = _INVALID_CHARS_PATTERN.sub(' ', text)
    
    # Chuẩn hóa khoảng trắng 
    text = _MULTIPLE_SPACES_PATTERN.sub(' ', text)
    text = _SPACE_BEFORE_NEWLINE_PATTERN.sub('\n', text)
    text = _SPACE_AFTER_NEWLINE_PATTERN.sub('\n', text)
    
    # Giới hạn tối đa 2 dòng trống liên tiếp
    text = _MULTIPLE_NEWLINES_PATTERN.sub('\n\n', text)
    
    # Sửa lỗi dư khoảng trắng trước dấu câu
    text = _SPACE_BEFORE_PUNCTUATION_PATTERN.sub(r'\1', text)
    
    return text.strip()