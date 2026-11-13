
import os
from pathlib import Path
from dotenv import load_dotenv


# 1. Xác định thư mục gốc project
BASE_DIR = Path(__file__).resolve().parents[2]


# 2. Đọc file .env
load_dotenv(BASE_DIR / ".env")


# 3. Groq chat; embeddings run locally.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"


# 4. RAG Configuration
CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "200")
)

TOP_K = int(
    os.getenv("TOP_K", "3")
)
MAX_RETRIEVAL_DISTANCE = float(os.getenv("MAX_RETRIEVAL_DISTANCE", "1.5"))
if not 0 <= MAX_RETRIEVAL_DISTANCE <= 4:
    raise ValueError("MAX_RETRIEVAL_DISTANCE phải nằm trong khoảng 0 đến 4")


# 5. Storage Configuration
def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


PDF_DIRECTORY = project_path(os.getenv("PDF_DIRECTORY", "data/pdf"))
FAISS_INDEX_PATH = project_path(os.getenv("FAISS_INDEX_PATH", "storage/faiss_index"))


# 6. Kiểm tra cấu hình
if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE phải lớn hơn 0")

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP phải >= 0 và nhỏ hơn CHUNK_SIZE"
    )

if TOP_K <= 0:
    raise ValueError("TOP_K phải lớn hơn 0")
