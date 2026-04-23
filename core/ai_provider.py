import os
import logging
import threading
import groq

logger = logging.getLogger(__name__)

class AIProviderManager:
    def __init__(self):
        # Lấy danh sách keys 
        self.groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
        self.gemini_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
        self.groq_idx = 0
        self.gemini_idx = 0
        self._lock = threading.Lock() # Đảm bảo Thread-Safe khi có nhiều Request cùng lúc

    def get_groq_client(self):
        if not self.groq_keys: return None
        # Chỉ lấy key, không thay đổi state nên không cần lock
        return groq.Groq(api_key=self.groq_keys[self.groq_idx])

    def rotate_groq(self):
        with self._lock: # Khóa luồng khi xoay tua để tránh xung đột
            if len(self.groq_keys) > 1:
                self.groq_idx = (self.groq_idx + 1) % len(self.groq_keys)
                logger.info(f"Đã xoay sang Groq Key thứ {self.groq_idx + 1}")

    def get_gemini_key(self):
        if not self.gemini_keys: return None
        return self.gemini_keys[self.gemini_idx]

    def rotate_gemini(self):
        with self._lock:
            if len(self.gemini_keys) > 1:
                self.gemini_idx = (self.gemini_idx + 1) % len(self.gemini_keys)
                logger.info("Đã xoay sang Gemini Key dự phòng")

api_manager = AIProviderManager()