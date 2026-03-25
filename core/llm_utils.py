import time
import logging 
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)

def safe_stream(llm, prompt) :
    try:
        for chunk in llm.stream(prompt):
            text = getattr(chunk, "content", str(chunk))
            if text:
                yield text
    except Exception :
        logger.exception("Lỗi khi stream LLM:")
        yield "Lỗi khi stream LLM "
def safe_invoke(llm ,prompt : str, timeout : int =30, retries: int =2):
    last_error = None
    for attempt in range(1, retries+1):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(llm.invoke, prompt)
                return fut.result(timeout=timeout)
        except FuturesTimeoutError as e:
            last_error = e
            logger.warning(f" Lần {attempt}: LLM invoke timeout sau {timeout} giây. Đang thử lại...")
        except Exception as e:
            last_error = e
            logger.error(f"Lần {attempt}: Lỗi khi gọi LLM: {e}. Đang thử lại...")
        time.sleep(0.6*attempt)
        
    raise RuntimeError (f"LLM failed after {retries} attempts: {last_error}")  # Thêm delay nhỏ trước khi thử lại