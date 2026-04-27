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
    
    from core.ai_provider import api_manager  
    
    # Prompt được tối ưu để ép AI trả về JSON chuẩn
    prompt = f"""
    Bạn là bộ tìm kiếm thông tin thông minh cho hệ thống hỏi đáp về "Quy chế đào tạo của Trường Đại học Thủy Lợi".
    Nhiệm vụ: Phân tích câu hỏi "{question}" và trả về JSON.
    

     QUY TẮC PHÂN LOẠI CỰC KỲ NGHIÊM NGẶT:
    BẠN CHỈ CẦN ĐƯA RA CÂU HỎI TƯƠNG TỰ CHỨ KHÔNG CẦN TRẢ LỜI CÂU HỎI 
    1. **CHỐNG ẢO GIÁC (ANTI-HALLUCINATION) - ƯU TIÊN SỐ 1:**
        - Đọc kỹ `CÂU HỎI CỦA SINH VIÊN` và `TÀI LIỆU THAM KHẢO`.
        - Nếu `CÂU HỎI` là câu hỏi cá nhân, trêu đùa (VD: "bạn biết tôi là ai không", "ăn cơm chưa") -> BỎ QUA TÀI LIỆU, trả lời ngay: "Xin lỗi, tôi chỉ hỗ trợ giải đáp thông tin về quy chế đào tạo."
        - Nếu `TÀI LIỆU THAM KHẢO` chứa nội dung KHÔNG LIÊN QUAN CHÚT NÀO đến câu hỏi (VD: Hỏi về 'điểm rèn luyện' nhưng tài liệu lại nói về 'học phí') -> TUYỆT ĐỐI KHÔNG tóm tắt tài liệu. Trả lời ngay: "Rất tiếc, hệ thống không tìm thấy thông tin phù hợp trong quy chế để trả lời câu hỏi của bạn."
    
    2. Loại "outlier" (Ngoài lề):
       - Dành cho các câu hỏi hoàn toàn không liên quan đến QUY CHẾ ĐÀO TẠO CỦA TRƯỜNG ĐẠI HỌC THỦY LỢI ( VD : các câu hỏi về trường khác, chủ đề khác, hoặc câu hỏi cá nhân, trêu đùa).
       - HÀNH ĐỘNG: Trả về câu trả lời từ chối lịch sự, KHÔNG ĐƯỢC phép trả lời theo kiểu "Tôi không biết" hoặc "Tôi không thể trả lời". PHẢI TRẢ LỜI CỤ THỂ rằng bạn chỉ hỗ trợ về quy chế đào tạo của Thủy Lợi.
       - Expanded queries: Rỗng [].
    3. LOẠI "normal" (Xã giao):
       - CHỈ DÀNH CHO: "Xin chào", "Hi", "Hello", "Cảm ơn", "Tạm biệt", "Bạn tên là gì?", "Bạn ai tạo ra".
       - HÀNH ĐỘNG: Trả về câu trả lời ngắn gọn, thân thiện.
       - Expanded queries: Rỗng [].

    4. LOẠI "simple" / "comparative" / "sequential" / "temporal" / "verification" / "exception" (Tìm kiếm tài liệu):
       - Dành cho TẤT CẢ các câu hỏi khác, kể cả câu hỏi ngắn hay viết tắt.
       - Ví dụ: "Quy chế thi", "mất mạng thì sao", "bị đình chỉ", "tính điểm thế nào", "sinh viên làm gì".
       - BẮT BUỘC đặt "answer": null (để hệ thống đi tìm trong tài liệu).
       - Expanded queries: Tạo 2-3 biến thể từ khóa để tìm kiếm tốt hơn.
    5. KỸ NĂNG MỞ RỘNG TỪ KHÓA TỔNG QUÁT (BẮT BUỘC):
       - Trả về danh sách CHÍNH XÁC 3 CÂU bao gồm: 1 câu gốc đã tối ưu + 2 biến thể theo từ khóa học vụ. KHÔNG ĐƯỢC sinh quá 3 câu.
       - Bạn phải đóng vai một chuyên viên phòng Đào tạo. Nhiệm vụ của bạn là "dịch" ngôn ngữ đời thường/viết tắt của sinh viên sang các THUẬT NGỮ HÀNH CHÍNH, PHÁP LÝ chính thức thường xuất hiện trong các văn bản quy phạm.
       - LUÔN LUÔN tạo ra các biến thể tìm kiếm theo 3 hướng sau để đảm bảo không bỏ sót tài liệu:
         + Hướng 1 (Hành chính hóa): Chuyển đổi các động từ/danh từ thông tục sang từ ngữ học vụ trang trọng. (Ví dụ: "đuổi học" -> "buộc thôi học"; "trượt môn" -> "học lại, điểm F"; "xin nghỉ" -> "tạm ngừng học tập").
         + Hướng 2 (Từ khóa bao trùm): Tìm chủ đề lớn chứa vấn đề đó. (Ví dụ: Hỏi về "điểm rèn luyện" -> Thêm từ khóa "Đánh giá kết quả rèn luyện").
         + Hướng 3 (Định nghĩa): Thêm các tiền tố để tìm chính xác định nghĩa. (Ví dụ: "Học bổng là gì", "Các loại học bổng", "Quy định về...").
       - Trả về danh sách gồm câu gốc và các biến thể này.

    OUTPUT JSON FORMAT:
    {{
        "question_type": "outlier"|"normal" | "simple" | "comparative" | "sequential" | "temporal" | "verification" | "exception",
        "answer": "Nội dung trả lời (chỉ nếu là normal hoặc outlier) hoặc null (nếu là câu hỏi thi cử)",
        "expanded_queries": ["Câu số 1", "Câu số 2", "Câu số 3"]
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
        question_lower = question.lower()
        
        academic_keywords = [
            "tín chỉ", "học bổng", "bảo lưu", "thi", "điểm", "học vụ", 
            "buộc thôi học", "đình chỉ", "nghỉ học", "học phí", "rèn luyện", 
            "cảnh báo", "tốt nghiệp", "học lại", "học cải thiện", "đồ án",
            "chuyên đề", "chuẩn đầu ra", "học kỳ", "học phần"
        ]
        school_keywords = ["tlu", "thủy lợi", "thuy loi", "truong minh", "trường mình"]
        
        has_academic_keyword = any(kw in question_lower for kw in academic_keywords)
        has_school_keyword = any(skw in question_lower for skw in school_keywords)
        
        if q_type == "normal":
            # Normal: Chỉ cần có từ khóa học vụ là bẻ lái (hoặc câu quá dài)
            if has_academic_keyword or len(question.split()) > 10:
                logger.info(" Phát hiện từ khóa học vụ. Ép về simple.")
                q_type = "simple"
                ans = None
                
        elif q_type == "outlier":
            if has_academic_keyword and has_school_keyword:
                logger.info(" Có đủ từ khóa học vụ và tên trường. Cứu vớt ép về simple.")
                q_type = "simple"
                ans = None
        if q_type == "normal" and not ans:
            ans = "Chào bạn 👋 Mình hỗ trợ tra cứu quy chế đào tạo."
        if q_type == "outlier" and not ans:
            ans = "Xin lỗi, tôi là trợ lý AI chuyên trách của Trường Đại học Thủy Lợi. Tôi chỉ hỗ trợ giải đáp các quy chế và thông tin liên quan đến sinh viên Thủy Lợi."

        if q_type != "normal" and q_type != "outlier":
            ans = None
        
        # Đảm bảo danh sách truy vấn
        queries = result.get("expanded_queries", [])
        if not isinstance(queries, list): queries = []
        if not queries: queries = [question]
        if question not in queries: queries.insert(0, question)

        queries = queries[:3]

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