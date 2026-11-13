# NetTutor AI

Ứng dụng hỏi đáp Mạng máy tính bằng tiếng Việt. Nội dung được truy xuất từ PDF trong `data/pdf`; câu trả lời hiển thị nguồn và số trang.

## Chạy ứng dụng

Trên Windows, sau khi cài đặt bên dưới, nhấp đúp `run_app.bat` để mở ứng dụng.
File này ưu tiên Python trong `.venv` nếu có. Đóng cửa sổ chạy hoặc nhấn `Ctrl+C`
để dừng ứng dụng.

Yêu cầu Python 3.11 trở lên. Tạo chỉ mục dùng embedding cục bộ, không cần API key.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Để tạo câu trả lời, tạo API key tại [Groq Console](https://console.groq.com/keys) và điền vào `.env`:

```dotenv
GROQ_API_KEY=gsk-your-api-key-here
GROQ_MODEL=openai/gpt-oss-20b
```

Thay key mẫu bằng key thật. Ứng dụng dùng `ChatGroq` từ `langchain-groq` để gọi Groq. Model `openai/gpt-oss-20b` được phục vụ trên Groq, không gọi API của OpenAI. Có thể đổi `GROQ_MODEL` sang model trong [danh sách Groq hỗ trợ](https://console.groq.com/docs/models). Các biến `CHAT_API_KEY`, `CHAT_BASE_URL`, `CHAT_MODEL` và cấu hình embedding cũ không còn được sử dụng. Thay model chat không cần tạo lại chỉ mục. Khởi động lại ứng dụng sau khi sửa `.env`:

```powershell
python -m streamlit run app.py
```

Nhấn **Tạo lại chỉ mục** ở thanh bên để đọc tất cả PDF và tạo FAISS index. Embedding cục bộ dùng đặc trưng ký tự để tìm đoạn văn liên quan và không tiêu thụ credit API. Các bước tạo câu trả lời và viết lại câu hỏi theo lịch sử dùng Groq API. Khi thêm hoặc thay PDF, nhấn nút tạo lại chỉ mục. Index được lưu trong `storage/faiss_index/`. Nếu chỉ mục cũ được tạo bằng embedding từ API, cần tạo lại chỉ mục để dùng embedding cục bộ.

Có thể tạo index bằng dòng lệnh:

```powershell
python -m src.ingestion.index_documents
```

PDF chỉ có ảnh quét cần OCR trước khi nạp; bộ nạp hiện tại chỉ đọc lớp văn bản của PDF.

Tìm kiếm chỉ giữ các đoạn có bình phương khoảng cách L2 không vượt quá
`MAX_RETRIEVAL_DISTANCE` (mặc định `1.5`). Giảm ngưỡng để lọc chặt hơn, tăng
ngưỡng để lấy nhiều kết quả hơn; nên hiệu chỉnh bằng câu hỏi thực tế cho bộ PDF
và embedding đang dùng. Đây là bộ lọc độ tương đồng, không bảo đảm mọi kết quả
đều liên quan. Khi không có đoạn đạt ngưỡng, ứng dụng báo không tìm thấy tài liệu.

Ứng dụng dùng tối đa 6 lượt hỏi đáp gần nhất để hiểu câu hỏi tiếp nối. Khi có
lịch sử, mỗi câu hỏi dùng thêm một lần gọi Chat API để tạo câu hỏi độc lập cho
bước tìm kiếm (kể cả khi sau đó không tìm thấy tài liệu). Câu trả lời vẫn phải
dựa vào tài liệu vừa truy xuất.
