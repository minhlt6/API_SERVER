from typing import List, Generator
import os, re, hashlib
import logging 
import groq
import google.generativeai as genai
# Giữ nguyên các import của Minh
from .models import llm
from .config import TOP_K_RESULTS, FINAL_TOP_K
from .rerank import advanced_rerank
from .prompting import create_advanced_prompt
from .retriever import HybridRetriever
from .analyze_and_expand import analyze_and_expand_query
from .llm_utils import safe_invoke, safe_stream

logger = logging.getLogger(__name__)

# Giữ nguyên các hằng số
MAX_CONTEXT_CHARS = 12000
MAX_DOC_CHARS = 1800 
MAX_OUT_CHARS = 3000

# Quản lý API Keys cho Groq và Gemini với xoay tua tự động khi gặp lỗi hoặc hết hạn
class AIProviderManager:
    def __init__(self):
        # Lấy danh sách keys 
        self.groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
        self.gemini_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
        self.groq_idx = 0
        self.gemini_idx = 0

    def get_groq_client(self):
        if not self.groq_keys: return None
        return groq.Groq(api_key=self.groq_keys[self.groq_idx])

    def rotate_groq(self):
        if len(self.groq_keys) > 1:
            self.groq_idx = (self.groq_idx + 1) % len(self.groq_keys)
            logger.info(f"🔄 Đã xoay sang Groq Key thứ {self.groq_idx + 1}")

    def get_gemini_key(self):
        if not self.gemini_keys: return None
        return self.gemini_keys[self.gemini_idx]

    def rotate_gemini(self):
        if len(self.gemini_keys) > 1:
            self.gemini_idx = (self.gemini_idx + 1) % len(self.gemini_keys)
            logger.info(f"🔄 Đã xoay sang Gemini Key dự phòng")

api_manager = AIProviderManager()

def sanitize_for_prompt(text: str) -> str:
    """Lọc bỏ prompt injection và PII - Giữ nguyên của Minh"""
    text = re.sub(r"(?i)(ignore previous instructions|system prompt|developer message|jailbreak)", "[FILTERED_INJECTION]", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\b(0\d{9}|\+84\d{9,10})\b", "[PHONE]", text)
    text = re.sub(r"\b\d{8,12}\b", "[ID]", text)  
    return text.strip()

def generate_standalone_query(message: str, history: List) -> str:
    """Tái tạo câu hỏi từ lịch sử - Giữ nguyên logic xử lý history phức tạp của Minh"""
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

    prompt = f"""Dựa vào lịch sử hội thoại, hãy viết lại câu hỏi hiện tại thành một câu hỏi độc lập, trọn vẹn ý nghĩa.
     QUY TẮC QUAN TRỌNG (BẮT BUỘC TUÂN THỦ): 
    1. BẮT BUỘC THÊM CHỦ ĐỀ TỪ LỊCH SỬ: Nếu câu hỏi hiện tại là câu hỏi nối tiếp, hỏi cộc lốc hoặc thiếu chủ đề (Ví dụ: "điều 5 là gì?", "vậy còn điểm F thì sao?", "áp dụng cho đối tượng nào?"), bạn PHẢI lấy TÊN VĂN BẢN hoặc CHỦ ĐỀ đang được nói đến ở AI ngay trước đó ghép vào câu hỏi.
       - Ví dụ lịch sử đang nói về Giáo dục thể chất. Câu hỏi: "điều 5 là gì?" -> Câu độc lập: "Điều 5 trong quy định môn học Giáo dục thể chất là gì?".
    2. GIỮ NGUYÊN VẸN các thuật ngữ chuyên ngành, tên ngành, từ viết tắt.
    3. Nếu câu hỏi hiện tại đang chuyển sang chủ đề hoàn toàn mới (có chứa từ khóa của chủ đề mới), hãy bỏ qua lịch sử và giữ nguyên câu hỏi hiện tại.
    Lịch sử:
    {history_str}
    
    Câu hỏi hiện tại: {message}
    Câu hỏi độc lập:"""
    
    # Sử dụng xoay tua cho bước tái tạo câu hỏi
    for _ in range(max(1, len(api_manager.groq_keys))):
        try:
            client = api_manager.get_groq_client()
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant", # Dùng bản 8B cho nhanh và tiết kiệm
                messages=[{"role": "user", "content": prompt}]
            )
            standalone_q = response.choices[0].message.content.strip()
            logger.info(f" Câu hỏi đã tái tạo: {standalone_q}")
            return standalone_q
        except Exception:
            api_manager.rotate_groq()
            continue
    return message

def ask_ai_improved(message: str, history: List, hybrid_retriever) -> Generator[str, None, None]:
    """Giữ nguyên hàm cumulative stream của Minh"""
    full_response = ""
    for delta in ask_ai_stream_delta(message, history, hybrid_retriever):
        full_response += delta
        if len(full_response) > MAX_OUT_CHARS:
            yield full_response[:MAX_OUT_CHARS] + "\n\n[Đã cắt bớt nội dung dài]"
            return
        yield full_response

def ask_ai_stream_delta(message: str, history: List, hybrid_retriever) -> Generator[str, None, None]:
    """Hàm chính xử lý RAG - Giữ nguyên 100% flow của Minh"""
    if not message.strip():
        yield " Bạn chưa nhập câu hỏi."
        return

    if message.strip().lower() in {"hello", "hi", "xin chào", "chào"}:
        yield "Chào bạn 👋 Mình hỗ trợ tra cứu quy chế đào tạo. Bạn cần hỏi điều gì?"
        return

    logger.info(f" CÂU HỎI GỐC: {message}")
    question = generate_standalone_query(message, history)
    processed_data = analyze_and_expand_query(question)

    if processed_data.get("question_type") == "normal":
        ans = processed_data.get("answer") or "Chào bạn 👋 Mình hỗ trợ tra cứu quy chế đào tạo."
        yield ans
        return

    question_type = processed_data['question_type']
    queries = processed_data['expanded_queries']
    logger.info(f"Các truy vấn tìm kiếm: {queries}")

    all_docs: List = []
    seen = set()
    for query in queries:
        # GIỮ NGUYÊN logic alpha ngành CNTT của Minh
        current_alpha = 0.4 if "CNTT" in query.upper() else 0.5
        docs = hybrid_retriever.search(query, k=TOP_K_RESULTS, alpha=current_alpha)
        for doc in docs:
            content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
            if content_hash not in seen:
                all_docs.append(doc)
                seen.add(content_hash)

    logger.info(f"Tìm thấy tổng {len(all_docs)} documents.")
    if not all_docs:
        yield "Không tìm thấy thông tin liên quan trong tài liệu."
        return

    final_docs = advanced_rerank(question, all_docs, top_k=FINAL_TOP_K)

    context_parts = []
    total_chars = 0
    for doc in final_docs:
        page = doc.metadata.get('page_number', 'N/A')
        file_name = doc.metadata.get('source_file') or doc.metadata.get('source')
        source = f"[{os.path.basename(file_name)} | Trang {page}]" if file_name else f"[Trang {page}]"
        block = f"{source}\n{doc.page_content}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        total_chars += len(block)
        context_parts.append(block)
    
    context = "\n\n---\n\n".join(context_parts)
    topic_hint = processed_data.get('topic') or processed_data.get('root_question') or question
    prompt = create_advanced_prompt(question, context, question_type, topic_hint)

    logger.info("Đang tạo câu trả lời cuối cùng ...")
    
    success = False
    # Thử với Groq 
    for _ in range(len(api_manager.groq_keys)):
        try:
            client = api_manager.get_groq_client()
            stream = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token
            success = True
            break
        except Exception as e:
            if "429" in str(e): # Lỗi Rate Limit
                api_manager.rotate_groq()
                continue
            logger.error(f"Lỗi Groq: {e}")
            break
            
    # Dự phòng sang Gemini (nếu Groq lỗi hoặc hết key)
    if not success:
        logger.warning("Chuyển sang Gemini ...")
        for _ in range(max(1, len(api_manager.gemini_keys))):
            try:
                genai.configure(api_key=api_manager.get_current_gemini_key())
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                success = True
                break
            except Exception as e:
                api_manager.rotate_gemini()
                logger.error(f"Lỗi Gemini: {e}")

    if not success:
        yield "Đã xảy ra lỗi hệ thống hoặc quá tải. Vui lòng thử lại sau giây lát!"