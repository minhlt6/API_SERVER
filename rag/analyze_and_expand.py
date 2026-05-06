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
    
    # Prompt được tối ưu để ép AI trả về JSON chuẩn và xử lý nhiễu giao tiếp
    prompt = f"""
    Bạn là bộ tìm kiếm thông tin thông minh cho hệ thống hỏi đáp về "Quy chế đào tạo của Trường Đại học Thủy Lợi".
    Nhiệm vụ: Phân tích câu hỏi "{question}" và trả về JSON.
    
    QUY TẮC PHÂN LOẠI CỰC KỲ NGHIÊM NGẶT:
    BẠN CHỈ CẦN ĐƯA RA CÂU HỎI TƯƠNG TỰ CHỨ KHÔNG CẦN TRẢ LỜI CÂU HỎI 
    1. **CHỐNG ẢO GIÁC (ANTI-HALLUCINATION) - ƯU TIÊN SỐ 1:**
        - ĐỌC KỸ CÂU HỎI, NẾU phát hiện nhắc đến các trường đại học KHÁC:
        (VD: Học bổng đại học bách khoa -> Outlier vì nhắc đến Bách Khoa; Bạn biết tôi là ai -> Outlier vì câu hỏi cá nhân;...)
        - HOẶC các chủ đề ngoài quy chế (code, nấu ăn, lịch sử...)
        => BẮT BUỘC: "question_type": "outlier"
   
         DANH SÁCH TRƯỜNG KHÁC (KHÔNG PHẢI THỦY LỢI):
        - "bách khoa", "bach khoa", "hust"
        - "neu", "kinh tế"
        - "ngoại thương",  "ftu"
        - "sư phạm",  "hpu"
        - "nông lâm", 
        - "công nghệ",  "uit"
        - "huflit", "rmit", "fpt", "văn hiến",  "mở"
    2. Loại "outlier" (Ngoài lề):
       - DÀNH CHO: Câu hỏi nhắc đến tên trường đại học khác, chủ đề không liên quan môi trường đại học, câu hỏi cá nhân.
       - HÀNH ĐỘNG: Trả về câu trả lời từ chối lịch sự trong trường "answer".
       - Expanded queries: Rỗng [].

    3. Loại "normal" (Giao tiếp thuần túy):
       - CHỈ DÀNH CHO: "Xin chào", "Cảm ơn", "Tạm biệt", "Bạn tên là gì?", "Bạn ai tạo ra".
       - LƯU Ý KHI LỌC NHIỄU: NẾU câu vừa có chào hỏi vừa có câu hỏi học vụ (VD: "Chào bạn, cho mình hỏi về học bổng") -> KHÔNG được xếp vào "normal". Phải xếp vào loại tìm kiếm phía dưới và cắt bỏ phần chào hỏi.
       - HÀNH ĐỘNG: Trả về câu trả lời ngắn gọn, thân thiện trong "answer". 
       - Expanded queries: Rỗng [].

    4. CÁC LOẠI TÌM KIẾM TÀI LIỆU ("simple" / "comparative" / "sequential" / "temporal" / "verification" / "exception"):
       - Dành cho TẤT CẢ các câu có ý định hỏi về quy chế, bất kể CÓ HAY KHÔNG kèm lời chào/cảm ơn.
       - "simple": Câu hỏi đơn giản, hỏi về 1 khái niệm/quy định.
       - "comparative": Cần so sánh giữa 2 hay nhiều thứ.
       - "sequential": Câu hỏi về quy trình, các bước, thủ tục.
       - "temporal": Câu hỏi về thời gian, thời hạn.
       - "verification": Câu hỏi đúng/sai, xác minh thông tin "Có được không?".
       - "exception": Câu hỏi về ngoại lệ, trường hợp đặc biệt.
       - BẮT BUỘC đặt "answer": null.

    5. KỸ NĂNG MỞ RỘNG TỪ KHÓA TỔNG QUÁT (BẮT BUỘC):
       - Trả về danh sách CHÍNH XÁC 3 CÂU bao gồm: 1 câu gốc đã tối ưu + 2 biến thể theo từ khóa học vụ. KHÔNG ĐƯỢC sinh quá 3 câu.
       - Bạn phải đóng vai một chuyên viên phòng Đào tạo. Nhiệm vụ của bạn là "dịch" ngôn ngữ đời thường/viết tắt của sinh viên sang các THUẬT NGỮ HÀNH CHÍNH, PHÁP LÝ chính thức thường xuất hiện trong các văn bản quy phạm.
       - LUÔN LUÔN tạo ra các biến thể tìm kiếm theo 3 hướng sau để đảm bảo không bỏ sót tài liệu:
         + Hướng 1 (Hành chính hóa): Chuyển đổi các động từ/danh từ thông tục sang từ ngữ học vụ trang trọng. (Ví dụ: "đuổi học" -> "buộc thôi học"; "trượt môn" -> "học lại, điểm F"; "xin nghỉ" -> "tạm ngừng học tập").
         + Hướng 2 (Từ khóa bao trùm): Tìm chủ đề lớn chứa vấn đề đó. (Ví dụ: Hỏi về "điểm rèn luyện" -> Thêm từ khóa "Đánh giá kết quả rèn luyện").
         + Hướng 3 (Định nghĩa): Thêm các tiền tố để tìm chính xác định nghĩa. (Ví dụ: "Học bổng là gì", "Các loại học bổng", "Quy định về...").
       - Trả về danh sách chính xác 3 câu tìm kiếm đã được tối ưu hóa. Câu gốc phải nằm trong danh sách nếu nó phù hợp, nếu không hãy tạo biến thể gần nhất theo hướng hành chính hóa.

    [VÍ DỤ MẪU - FEW SHOT EXAMPLES]
    Input: "Chào bot nhé bạn ăn cơm chưa"
    Output: {{"question_type": "normal", "answer": "Chào bạn 👋 Mình là trợ lý hỏi đáp quy chế đào tạo, bạn cần hỗ trợ gì ạ?", "expanded_queries": []}}

    Input: "Dạ cho em hỏi Bách khoa xét học bổng thế nào ạ?"
    Output: {{"question_type": "outlier", "answer": "Xin lỗi, tôi chỉ hỗ trợ thông tin liên quan đến Trường Đại học Thủy Lợi, không thể trả lời về Bách Khoa.", "expanded_queries": []}}

    Input: "Chào bạn, cho mình hỏi kỳ này sinh viên đăng ký rút học phần muộn nhất là khi nào vậy, cảm ơn bạn."
    Output: {{"question_type": "temporal", "answer": null, "expanded_queries": ["Thời hạn đăng ký rút học phần mùa muộn nhất", "Quy định thời gian hủy học phần đăng ký", "Lịch trình xin rút bớt môn học"]}}

    Input: "Bị điểm F thì có bị đuổi học không"
    Output: {{"question_type": "verification", "answer": null, "expanded_queries": ["Điểm F học phần có bị buộc thôi học không", "Quy định xử lý sinh viên nhận điểm F", "Điều kiện buộc thôi học kết quả học tập"]}}

    Input: "Học bổng khá với giỏi khác nhau nhiều không ạ"
    Output: {{"question_type": "comparative", "answer": null, "expanded_queries": ["So sánh học bổng khuyến khích học tập loại khá và giỏi", "Tiêu chuẩn xét học bổng khá và giỏi", "Mức tiền học bổng khá giỏi"]}}

    Input: "Em bị cảnh báo học vụ lần 1 thì phải làm giấy tờ gì không"
    Output: {{"question_type": "sequential", "answer": null, "expanded_queries": ["Quy trình thủ tục xử lý sinh viên cảnh báo học vụ lần 1", "Xử lý kết quả học tập cảnh báo học vụ", "Sinh viên cần làm gì khi bị cảnh báo kết quả học tập"]}}

    Input: "Đang bảo lưu mà có giấy gọi nhập ngũ thì sao?"
    Output: {{"question_type": "exception", "answer": null, "expanded_queries": ["Ngoại lệ gọi nhập ngũ khi đang tạm ngừng học tập", "Quy định bảo lưu kết quả học tập đi nghĩa vụ quân sự", "Trường hợp đặc biệt tạm ngừng học tập"]}}

    Input: "Mất thẻ sinh viên"
    Output: {{"question_type": "simple", "answer": null, "expanded_queries": ["Quy định cấp lại thẻ sinh viên bị mất", "Thủ tục xin cấp lại thẻ sinh viên", "Xử lý trường hợp làm mất thẻ sinh viên"]}}

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
        question_lower = question.lower()
        other_schools = [
            "bách khoa", "bach khoa", "hust",
            "neu", "kinh tế",
            "ngoại thương",  "ftu",
            "sư phạm",  "hpu",
            "nông lâm", 
            "công nghệ",  "uit",
            "huflit", "rmit", "fpt", "văn hiến",  "mở"
        ]

        has_other_school = any(school in question_lower for school in other_schools)

        if has_other_school:
            logger.info(f"⚠️ DETECTED trường khác trong câu hỏi! Force → outlier")
            q_type = "outlier"
            ans = "Xin lỗi, tôi là trợ lý AI chuyên trách của Trường Đại học Thủy Lợi. Tôi chỉ hỗ trợ giải đáp các quy chế và thông tin liên quan đến sinh viên Thủy Lợi."
            queries = []
            
            final_result = {
                "question_type": q_type,
                "answer": ans,
                "expanded_queries": queries
            }
            print(f"✓ Phân loại: {final_result['question_type']} (Safety Check)")
            return final_result
        # Logic an toàn: Nếu AI lỡ trả lời câu hỏi chuyên môn trong field "answer", ta xóa nó đi để ép hệ thống tìm docs
        q_type = result.get("question_type", "simple")
        ans = result.get("answer", None)
        
        academic_keywords = [
            "tín chỉ", "học bổng", "bảo lưu", "thi", "điểm", "học vụ", 
            "buộc thôi học", "đình chỉ", "nghỉ học", "học phí", "rèn luyện", 
            "cảnh báo", "tốt nghiệp", "học lại", "học cải thiện", "đồ án",
            "chuyên đề", "chuẩn đầu ra", "học kỳ", "học phần"
        ]
        school_keywords = ["tlu", "thủy lợi", "thuy loi", "truong minh", "trường mình", "trường "]
        
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