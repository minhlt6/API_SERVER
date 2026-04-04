from typing import List, Generator
import os, re, hashlib
import logging 
import groq
import google.generativeai as genai
import json 

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
# [YEAR-AWARE CHANGE] Pattern nhan dien nam hoc trong cau hoi.
ACADEMIC_YEAR_PATTERN = re.compile(r"\b(20\d{2})\s*[-_/]\s*(20\d{2})\b")
SINGLE_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

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
            logger.info(f" Đã xoay sang Groq Key thứ {self.groq_idx + 1}")

    def get_gemini_key(self):
        if not self.gemini_keys: return None
        return self.gemini_keys[self.gemini_idx]

    def rotate_gemini(self):
        if len(self.gemini_keys) > 1:
            self.gemini_idx = (self.gemini_idx + 1) % len(self.gemini_keys)
            logger.info(f"Đã xoay sang Gemini Key dự phòng")

api_manager = AIProviderManager()


def normalize_academic_year(start_year: str, end_year: str) -> str:
    return f"{int(start_year):04d}-{int(end_year):04d}"


# [YEAR-AWARE CHANGE] Trich xuat nam yeu cau tu cau hoi.
def detect_requested_year(text: str) -> tuple[str, set]:
    """Phat hien nam hoc duoc nhac den trong cau hoi."""
    requested_range = ""
    mentioned_years = set()

    for start_year, end_year in ACADEMIC_YEAR_PATTERN.findall(text or ""):
        requested_range = normalize_academic_year(start_year, end_year)
        mentioned_years.add(start_year)
        mentioned_years.add(end_year)

    for year in SINGLE_YEAR_PATTERN.findall(text or ""):
        mentioned_years.add(year)

    return requested_range, mentioned_years


def infer_doc_academic_year(doc) -> str:
    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
    existing_year = metadata.get("academic_year")
    if existing_year:
        return existing_year

    source_text = " ".join(
        str(x) for x in [
            metadata.get("source_relpath"),
            metadata.get("source"),
            metadata.get("source_file"),
        ]
        if x
    )
    match = ACADEMIC_YEAR_PATTERN.search(source_text)
    if match:
        year = normalize_academic_year(match.group(1), match.group(2))
        metadata["academic_year"] = year
        doc.metadata = metadata
        return year

    metadata["academic_year"] = "ALL"
    doc.metadata = metadata
    return "ALL"


# [YEAR-AWARE CHANGE] Loc tai lieu theo metadata nam hoc.
def filter_docs_by_year(docs: List, requested_range: str, mentioned_years: set) -> List:
    if not requested_range and not mentioned_years:
        return docs

    filtered_docs = []
    for doc in docs:
        doc_year = infer_doc_academic_year(doc)
        if doc_year == "ALL":
            filtered_docs.append(doc)
            continue

        if requested_range and doc_year == requested_range:
            filtered_docs.append(doc)
            continue

        doc_year_tokens = set(SINGLE_YEAR_PATTERN.findall(doc_year))
        if doc_year_tokens.intersection(mentioned_years):
            filtered_docs.append(doc)

    return filtered_docs

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

def ask_ai_improved(message: str, history: List, hybrid_retriever) -> Generator[str, None, None]:
    full_response = ""
    for delta in ask_ai_stream_delta(message, history, hybrid_retriever):
        full_response += delta
        if len(full_response) > MAX_OUT_CHARS:
            yield full_response[:MAX_OUT_CHARS] + "\n\n[Đã cắt bớt nội dung dài]"
            return
        yield full_response

def ask_ai_stream_delta(message: str, history: List, hybrid_retriever) -> Generator[str, None, None]:
    if not message.strip():
        yield " Bạn chưa nhập câu hỏi."
        return

    if message.strip().lower() in {"hello", "hi", "xin chào", "chào"}:
        yield "Chào bạn 👋 Mình hỗ trợ tra cứu quy chế đào tạo. Bạn cần hỏi điều gì?"
        return

    logger.info(f" CÂU HỎI GỐC: {message}")
    question = generate_standalone_query(message, history)
    # [YEAR-AWARE CHANGE] Xac dinh pham vi nam ma nguoi dung yeu cau.
    requested_year_range, mentioned_years = detect_requested_year(f"{message}\n{question}")
    if requested_year_range:
        logger.info(f"Lọc theo năm học yêu cầu: {requested_year_range}")
    elif mentioned_years:
        logger.info(f"Lọc theo năm được nhắc tới: {sorted(mentioned_years)}")

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
        #Giữ nguyên logic alpha ngành CNTT của Minh
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

    # [YEAR-AWARE CHANGE] Loc tap docs theo nam truoc khi rerank.
    year_filtered_docs = filter_docs_by_year(all_docs, requested_year_range, mentioned_years)
    if (requested_year_range or mentioned_years) and not year_filtered_docs:
        if requested_year_range:
            yield f"Không tìm thấy thông tin phù hợp cho năm học {requested_year_range}."
        else:
            year_text = ", ".join(sorted(mentioned_years))
            yield f"Không tìm thấy thông tin phù hợp cho năm bạn yêu cầu ({year_text})."
        return

    if year_filtered_docs and len(year_filtered_docs) != len(all_docs):
        logger.info(f"Đã lọc theo năm: còn {len(year_filtered_docs)}/{len(all_docs)} documents")
        all_docs = year_filtered_docs

    final_docs = advanced_rerank(question, all_docs, top_k=FINAL_TOP_K)

    context_parts = []
    total_chars = 0
    for doc in final_docs:
        page = doc.metadata.get('page_number', 'N/A')
        file_name = doc.metadata.get('source_file') or doc.metadata.get('source')
        # [YEAR-AWARE CHANGE] Gan nhan nam trong context de LLM bam dung nguon.
        doc_year = infer_doc_academic_year(doc)
        year_label = f"Năm {doc_year}" if doc_year != "ALL" else "Áp dụng nhiều năm"
        source = f"[{year_label} | {os.path.basename(file_name)} | Trang {page}]" if file_name else f"[{year_label} | Trang {page}]"
        block = f"{source}\n{doc.page_content}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        total_chars += len(block)
        context_parts.append(block)
    
    context = "\n\n---\n\n".join(context_parts)
    topic_hint = processed_data.get('topic') or processed_data.get('root_question') or question
    # [YEAR-AWARE CHANGE] Truyen rang buoc nam vao prompt.
    if requested_year_range:
        year_scope = requested_year_range
    elif mentioned_years:
        year_scope = ", ".join(sorted(mentioned_years))
    else:
        year_scope = None

    prompt = create_advanced_prompt(question, context, question_type, topic_hint, year_scope=year_scope)

    logger.info("Đang tạo câu trả lời cuối cùng ...")
    
    success = False
    # Thử với Groq 
    for _ in range(len(api_manager.groq_keys)):
        try:
            client = api_manager.get_groq_client()
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
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
                genai.configure(api_key=api_manager.get_gemini_key())
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