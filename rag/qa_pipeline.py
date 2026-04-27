from typing import List, Generator
import os, re, hashlib
import logging 
from google import genai
import json 
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from core.ai_provider import api_manager
from core.config import TOP_K_RESULTS, FINAL_TOP_K
from .rerank import advanced_rerank
from .prompting import create_advanced_prompt
from .analyze_and_expand import analyze_and_expand_query

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000
MAX_DOC_CHARS = 1800 
MAX_OUT_CHARS = 3000

# Quản lý API Keys cho Groq và Gemini với xoay tua tự động khi gặp lỗi hoặc hết hạn


def sanitize_for_prompt(text: str) -> str:
    """Lọc bỏ prompt injection và PII """
    text = re.sub(r"(?i)(ignore previous instructions|system prompt|developer message|jailbreak)", "[FILTERED_INJECTION]", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\b(0\d{9}|\+84\d{9,10})\b", "[PHONE]", text)
    text = re.sub(r"\b\d{8,12}\b", "[ID]", text)  
    return text.strip()

def generate_standalone_query(message: str, history: List) -> str:
    """Tái tạo câu hỏi từ lịch sử """
    if not history:
        return message
        
    print("Đang phân tích ngữ cảnh từ lịch sử trò chuyện...")
    
    standardized_history = []
    for h in history:
        if isinstance(h, dict) and 'role' in h and 'content' in h:
            standardized_history.append({"role": h['role'], "content": h['content']})
        elif hasattr(h, 'role') and hasattr(h, 'content'):
            standardized_history.append({"role": h.role, "content": h.content})
        elif isinstance(h, (list, tuple)) and len(h) >= 2:
            if h[0]: standardized_history.append({"role": "user", "content": h[0]})
            if h[1]: standardized_history.append({"role": "assistant", "content": h[1]})
    
    recent_history = standardized_history[-4:] if len(standardized_history) > 4 else standardized_history

    history_parts=[]
    for msg in recent_history:
        role_name= "User" if msg["role"] == "user" else "AI"
        history_parts.append(f"{role_name}: {msg['content']}")
    history_str = "\n".join(history_parts)

    prompt = f"""Bạn là một chuyên gia ngôn ngữ. Nhiệm vụ: Đọc Lịch sử trò chuyện và Câu hỏi hiện tại để tạo ra một CÂU HỎI ĐỘC LẬP.
    
    QUY TẮC PHÂN TÍCH NGỮ CẢNH:
    - Nếu câu hỏi hiện tại bị thiếu chủ đề (nói tắt, dùng đại từ thay thế), hãy lấy chủ đề từ lịch sử đắp vào. 
    - Nếu câu hỏi hiện tại ĐÃ CÓ ĐỦ chủ đề hoặc LÀ CHỦ ĐỀ MỚI HOÀN TOÀN, phải GIỮ NGUYÊN.

    VÍ DỤ 1 (Cần ghép ngữ cảnh - vì câu hỏi nói tắt):
    Lịch sử: 
    User: Điều kiện nhận học bổng khuyến khích học tập là gì?
    AI: Sinh viên cần đạt điểm trung bình từ 3.2 trở lên...
    Câu hỏi hiện tại: Vậy điểm rèn luyện thì yêu cầu bao nhiêu?
    -> JSON trả về: {{"standalone_query": "Điều kiện về điểm rèn luyện để nhận học bổng khuyến khích học tập là bao nhiêu?"}}

    VÍ DỤ 2 (Chủ đề mới, tự đứng độc lập, tuyệt đối không mượn ngữ cảnh cũ):
    Lịch sử:
    User: Sinh viên vắng thi cuối kỳ không có lý do thì bị điểm mấy?
    AI: Theo quy định, sinh viên vắng thi không phép sẽ nhận điểm 0...
    Câu hỏi hiện tại: Điều kiện để được đăng ký học vượt là gì?
    -> JSON trả về: {{"standalone_query": "Điều kiện để sinh viên được đăng ký học vượt là gì?"}}

    VÍ DỤ 3 (Chuyển chủ đề hoàn toàn - KHÔNG được nhầm lẫn):
    Lịch sử:
    User: Quy định về thời hạn đóng học phí ra sao?
    AI: Học phí phải được đóng trong 4 tuần đầu...
    Câu hỏi hiện tại: Có những môn giáo dục thể chất nào?
    -> JSON trả về: {{"standalone_query": "Có những môn giáo dục thể chất nào?"}}

    BẮT BUỘC TRẢ VỀ ĐỊNH DẠNG JSON DUY NHẤT:
    {{
        "standalone_query": "Câu hỏi sau khi đã được tái tạo"
    }}

    Lịch sử thực tế:
    {history_str}
    
    Câu hỏi hiện tại: {message}"""
    
    # Gọi API ép trả về JSON
    for _ in range(max(1, len(api_manager.groq_keys))):
        try:
            client = api_manager.get_groq_client()
            if not client:
                break
                
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"} 
            )
            
            # Phân tích chuỗi JSON trả về
            result_str = response.choices[0].message.content.strip()
            result_json = json.loads(result_str)
            
            standalone_q = result_json.get("standalone_query", message).strip()
            logger.info(f" CÂU HỎI ĐÃ TÁI TẠO (JSON): {standalone_q}")
            return standalone_q
            
        except Exception as e:
            logger.error(f"Lỗi tái tạo câu hỏi (JSON): {e}")
            api_manager.rotate_groq()
            continue
            
    return message

def ask_ai_improved(message: str, history: List, hybrid_retriever, cohort_key: str | None = None) -> Generator[str, None, None]:
    full_response = ""
    for delta in ask_ai_stream_delta(message, history, hybrid_retriever, cohort_key=cohort_key):
        full_response += delta
        if len(full_response) > MAX_OUT_CHARS:
            yield full_response[:MAX_OUT_CHARS] + "\n\n[Đã cắt bớt nội dung dài]"
            return
        yield full_response

def ask_ai_stream_delta(message: str, history: List, hybrid_retriever, cohort_key: str | None = None) -> Generator[str, None, None]:
    if not message.strip():
        yield " Bạn chưa nhập câu hỏi."
        return

    if message.strip().lower() in {"hello", "hi", "xin chào", "chào"}:
        yield "Chào bạn 👋 Mình hỗ trợ tra cứu quy chế đào tạo. Bạn cần hỏi điều gì?"
        return

    logger.info(f" CÂU HỎI GỐC: {message}")
    question = generate_standalone_query(message, history)

    processed_data = analyze_and_expand_query(question)
    q_type = processed_data.get("question_type")

    # Chặn ngay lập tức nếu là xã giao hoặc lạc đề
    if q_type in ["normal", "outlier"]:
        ans = processed_data.get("answer") or "Mình là trợ lý hỗ trợ quy chế đào tạo Trường Đại học Thủy Lợi. Bạn cần hỏi gì về quy chế?"
        yield ans
        return

    question_type = processed_data['question_type']
    queries = processed_data['expanded_queries']
    logger.info(f"Các truy vấn tìm kiếm: {queries}")

    all_docs: List = []
    seen = set()
    seen_lock = Lock()
    if cohort_key:
        logger.info(f"Sử dụng cohort_key: {cohort_key}")
    
    def search_query(query: str):
        current_alpha = 0.4 if "CNTT" in query.upper() else 0.5
        return hybrid_retriever.search(
            query,
            k=TOP_K_RESULTS,
            alpha=current_alpha,
            cohort_key=cohort_key,
        )
    
    with ThreadPoolExecutor(max_workers=min(3, len(queries))) as executor:
        futures = {executor.submit(search_query, q): q for q in queries}
        for future in futures:
            try:
                docs = future.result(timeout=30)
                for doc in docs:
                    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
                    with seen_lock:
                        if content_hash not in seen:
                            all_docs.append(doc)
                            seen.add(content_hash)
            except Exception as e:
                logger.error(f"Lỗi khi search query '{futures[future]}': {e}")

    logger.info(f"Tìm thấy tổng {len(all_docs)} documents.")
    if not all_docs:
        yield "Không tìm thấy thông tin liên quan trong tài liệu."
        return

    final_docs = advanced_rerank(question, all_docs, top_k=FINAL_TOP_K)

    context_parts = []
    context_docs = []  
    total_chars = 0
    
    for doc in final_docs:
        page = doc.metadata.get('page_number', 'N/A')
        file_name = doc.metadata.get('source_file') or doc.metadata.get('source')
        source = f"[{os.path.basename(file_name)} | Trang {page}]" if file_name else f"[Trang {page}]"
        block = f"{source}\n{doc.page_content}"
        
        #  Dùng continue để nhét tối đa các chunk ngắn thay vì break làm đứt gánh
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            continue
            
        total_chars += len(block)
        context_parts.append(block)
        context_docs.append({
            'source': file_name or "Không rõ",
            'page': page
        })
    
    context = "\n\n---\n\n".join(context_parts)
    topic_hint = processed_data.get('topic') or processed_data.get('root_question') or question

    prompt = create_advanced_prompt(question, context, question_type, topic_hint)

    logger.info("Đang tạo câu trả lời cuối cùng ...")
    generated_text = ""
    success = False
    for _ in range(len(api_manager.groq_keys) if api_manager.groq_keys else 1):
        try:
            client = api_manager.get_groq_client()
            if not client:
                break
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    generated_text += token
                    yield token
            success = True
            break
        except Exception as e:
            if "429" in str(e): 
                api_manager.rotate_groq()
                continue
            logger.error(f"Lỗi Groq: {e}")
            break
    
    if not success:
        logger.warning("Chuyển sang Gemini ...")
        for _ in range(len(api_manager.gemini_keys) if api_manager.gemini_keys else 1):
            try:
                genai.configure(api_key=api_manager.get_gemini_key())
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        generated_text += chunk.text
                        yield chunk.text
                success = True
                break
            except Exception as e:
                api_manager.rotate_gemini()
                logger.error(f"Lỗi Gemini: {e}")
    
    if not success:
        yield "Đã xảy ra lỗi hệ thống hoặc quá tải. Vui lòng thử lại sau giây lát!"
        return
    is_refusal = "Xin lỗi, tôi là trợ lý" in generated_text or "không thể tư vấn" in generated_text
    if context_docs and not is_refusal:
        yield "\n\n---\n\n"
        yield "## 📚 Tài liệu tham khảo\n\n"
        seen_sources = set()
        for i, doc_info in enumerate(context_docs, 1):
            source_key = f"{doc_info['source']}_{doc_info['page']}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                yield f"- **{doc_info['source']}** (Trang {doc_info['page']})\n"