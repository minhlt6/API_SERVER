import json
import re
from typing import Dict, Any
from .models import  llm
from .llm_utils import safe_invoke
def clean_json_string(text: str) -> str:
    """Hàm làm sạch chuỗi JSON từ phản hồi của LLM"""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        return ""

    json_str = text[start_idx : end_idx + 1]
    return json_str.strip()

def analyze_and_expand_query(question: str) -> Dict[str, Any]:
    print(" Phân tích & Mở rộng câu hỏi...")
    
    # Prompt được tối ưu để ép AI trả về JSON chuẩn
    prompt = f"""
    Bạn là bộ tìm kiếm thông tin thông minh cho hệ thống hỏi đáp về "Quy chế đào tạo của Trường Đại học Thủy Lợi".
    Nhiệm vụ: Phân tích câu hỏi "{question}" và trả về JSON.
    

     QUY TẮC PHÂN LOẠI CỰC KỲ NGHIÊM NGẶT:

    1. **CHỐNG ẢO GIÁC (ANTI-HALLUCINATION) - ƯU TIÊN SỐ 1:**
        - Đọc kỹ `CÂU HỎI CỦA SINH VIÊN` và `TÀI LIỆU THAM KHẢO`.
        - Nếu `CÂU HỎI` là câu hỏi cá nhân, trêu đùa (VD: "bạn biết tôi là ai không", "ăn cơm chưa") -> BỎ QUA TÀI LIỆU, trả lời ngay: "Xin lỗi, tôi chỉ hỗ trợ giải đáp thông tin về quy chế đào tạo."
        - Nếu `TÀI LIỆU THAM KHẢO` chứa nội dung KHÔNG LIÊN QUAN CHÚT NÀO đến câu hỏi (VD: Hỏi về 'điểm rèn luyện' nhưng tài liệu lại nói về 'học phí') -> TUYỆT ĐỐI KHÔNG tóm tắt tài liệu. Trả lời ngay: "Rất tiếc, hệ thống không tìm thấy thông tin phù hợp trong quy chế để trả lời câu hỏi của bạn."
    
    2. LOẠI "normal" (Xã giao):
       - CHỈ DÀNH CHO: "Xin chào", "Hi", "Hello", "Cảm ơn", "Tạm biệt", "Bạn tên là gì?", "Bạn ai tạo ra".
       - HÀNH ĐỘNG: Trả về câu trả lời ngắn gọn, thân thiện.
       - Expanded queries: Rỗng [].

    3. LOẠI "simple" / "comparative" / "sequential" / "temporal" / "verification" / "exception" (Tìm kiếm tài liệu):
       - Dành cho TẤT CẢ các câu hỏi khác, kể cả câu hỏi ngắn hay viết tắt.
       - Ví dụ: "Quy chế thi", "mất mạng thì sao", "bị đình chỉ", "tính điểm thế nào", "sinh viên làm gì".
       - BẮT BUỘC đặt "answer": null (để hệ thống đi tìm trong tài liệu).
       - Expanded queries: Tạo 2-3 biến thể từ khóa để tìm kiếm tốt hơn.

    OUTPUT JSON FORMAT:
    {{
        "question_type": "normal" | "simple" | "comparative" | "sequential" | "temporal" | "verification" | "exception",
        "answer": "Nội dung trả lời (chỉ nếu là normal) hoặc null (nếu là câu hỏi thi cử)",
        "expanded_queries": ["câu gốc", "biến thể 1", "biến thể 2"]
    }}
    
    CHỈ TRẢ VỀ JSON DUY NHẤT. KHÔNG GIẢI THÍCH THÊM.
    """

    try:
        response = safe_invoke(llm, prompt, timeout=15, retries=1)
        content = response.content if hasattr(response, 'content') else str(response)
        
        cleaned_json = clean_json_string(content)
        if not cleaned_json:
            raise ValueError("Empty JSON content")

        try:
            result = json.loads(cleaned_json)
        except json.JSONDecodeError:
            fixed_str = cleaned_json.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
            result = json.loads(fixed_str)

        # Logic an toàn: Nếu AI lỡ trả lời câu hỏi chuyên môn trong field "answer", ta xóa nó đi để ép hệ thống tìm docs
        q_type = result.get("question_type", "simple")
        ans = result.get("answer", None)
        
        if q_type == "normal" and not ans:
            ans = "Chào bạn 👋 Mình hỗ trợ tra cứu quy chế đào tạo."

        if q_type != "normal":
            ans = None
        
        # Đảm bảo danh sách truy vấn
        queries = result.get("expanded_queries", [])
        if not isinstance(queries, list): queries = []
        if not queries: queries = [question]
        if question not in queries: queries.insert(0, question)

        final_result = {
            "question_type": q_type,
            "answer": ans,
            "expanded_queries": queries
        }
        
        print(f"Phân loại: {final_result['question_type']} | Queries: {len(final_result['expanded_queries'])}")
        return final_result

    except Exception as e:
        print(f" Lỗi phân tích ({e}). Mặc định chuyển sang tìm kiếm.")
        return {
            "question_type": "simple",
            "answer": None,
            "expanded_queries": [question]
        }