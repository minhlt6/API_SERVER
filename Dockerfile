FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code lên
COPY . .

# Mở cổng 7860 (Chuẩn của Hugging Face)
EXPOSE 7860

# Lệnh chạy server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]