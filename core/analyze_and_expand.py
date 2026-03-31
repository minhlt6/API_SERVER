import json
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

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
    
    # Import cục bộ để tránh lỗi vòng lặp import (circular import) với qa_pipeline
    from .qa_pipeline import api_manager
    
    # Prompt được tối ưu để ép AI trả về JSON chuẩn
    prompt = f"""
    Bạn là bộ tìm kiếm thông tin thông minh cho hệ thống hỏi đáp về "Quy chế đào tạo của Trường Đại học Thủy Lợi".
    Nhiệm vụ: Phân tích câu hỏi "{question}" và trả về JSON.
    

     QUY TẮC PHÂN LOẠI CỰC KỲ NGHIÊM NGẶT:

    1. **CHỐNG ẢO GIÁC (ANTI-HALLUCINATION) - ƯU TIÊN SỐ 1:**
        - Đọc kỹ `CÂU HỎI CỦA SINH VIÊN` và `TÀI LIỆU THAM KHẢO`.
        - Nếu câu hỏi nhắc đến TÊN CÁC TRƯỜNG ĐẠI HỌC KHÁC (VD: Bách Khoa, NEU, Kinh tế...) hoặc các chủ đề hoàn toàn nằm ngoài môi trường đại học (VD: nấu ăn, thời tiết, lịch sử thế giới, code lập trình):
        => BẮT BUỘC đặt "question_type": "normal", "answer": "Xin lỗi, tôi là trợ lý AI chuyên trách của Trường Đại học Thủy Lợi. Tôi chỉ hỗ trợ giải đáp các quy chế và thông tin liên quan đến sinh viên Thủy Lợi."
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
    4. KỸ NĂNG MỞ RỘNG TỪ KHÓA TỔNG QUÁT (BẮT BUỘC):
       - Bạn phải đóng vai một chuyên viên phòng Đào tạo. Nhiệm vụ của bạn là "dịch" ngôn ngữ đời thường/viết tắt của sinh viên sang các THUẬT NGỮ HÀNH CHÍNH, PHÁP LÝ chính thức thường xuất hiện trong các văn bản quy phạm.
       - LUÔN LUÔN tạo ra các biến thể tìm kiếm theo 3 hướng sau để đảm bảo không bỏ sót tài liệu:
         + Hướng 1 (Hành chính hóa): Chuyển đổi các động từ/danh từ thông tục sang từ ngữ học vụ trang trọng. (Ví dụ: "đuổi học" -> "buộc thôi học"; "trượt môn" -> "học lại, điểm F"; "xin nghỉ" -> "tạm ngừng học tập").
         + Hướng 2 (Từ khóa bao trùm): Tìm chủ đề lớn chứa vấn đề đó. (Ví dụ: Hỏi về "điểm rèn luyện" -> Thêm từ khóa "Đánh giá kết quả rèn luyện").
         + Hướng 3 (Định nghĩa): Thêm các tiền tố để tìm chính xác định nghĩa. (Ví dụ: "Học bổng là gì", "Các loại học bổng", "Quy định về...").
       - Trả về danh sách gồm câu gốc và các biến thể này.

    OUTPUT JSON FORMAT:
    {{
        "question_type": "normal" | "simple" | "comparative" | "sequential" | "temporal" | "verification" | "exception",
        "answer": "Nội dung trả lời (chỉ nếu là normal) hoặc null (nếu là câu hỏi thi cử)",
        "expanded_queries": ["câu gốc", "biến thể 1", "biến thể 2"]
    }}
    
    CHỈ TRẢ VỀ JSON DUY NHẤT. KHÔNG GIẢI THÍCH THÊM.
    """

    try:
        content = ""
        # Gọi API Groq trực tiếp và áp dụng xoay tua Key
        for _ in range(max(1, len(api_manager.groq_keys))):
            try:
                client = api_manager.get_groq_client()
                if not client:
                    break
                    
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                content = response.choices[0].message.content
                break  # Gọi API thành công thì thoát vòng lặp
            except Exception as e:
                logger.error(f"Lỗi Groq khi expand query, đang xoay key: {e}")
                api_manager.rotate_groq()

        if not content:
            raise ValueError("Không nhận được phản hồi hợp lệ từ API")

        cleaned_json = clean_json_string(content)
        if not cleaned_json:
            cleaned_json = content # Thử parse trực tiếp nếu clean trả về rỗng

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