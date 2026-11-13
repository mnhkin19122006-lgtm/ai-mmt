
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.embeddings.embedding_model import get_embedding_model
from src.utils.config import FAISS_INDEX_PATH


# 1. Tạo FAISS Vector Store
def create_vector_store(documents: list[Document]) -> FAISS:

    if not documents:
        raise ValueError("Danh sách documents không được rỗng")

    # Khởi tạo embedding model
    embeddings = get_embedding_model()

    # Chuyển chunks thành vectors và tạo FAISS index
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return vector_store


# 2. Lưu FAISS Vector Store
def save_vector_store(vector_store: FAISS) -> None:

    # Tạo thư mục lưu trữ nếu chưa tồn tại
    path = Path(FAISS_INDEX_PATH)
    path.mkdir(parents=True, exist_ok=True)

    # Lưu FAISS index
    vector_store.save_local(str(path))

    print(f"FAISS index saved at: {path}")


# 3. Đọc FAISS Vector Store đã lưu
def load_vector_store() -> FAISS:

    path = Path(FAISS_INDEX_PATH)

    # Kiểm tra file index
    if not (path / "index.faiss").is_file():
        raise FileNotFoundError(
            f"Không tìm thấy FAISS index tại: {path}"
        )

    if not (path / "index.pkl").is_file():
        raise FileNotFoundError(
            f"Không tìm thấy FAISS metadata tại: {path}"
        )

    # Khởi tạo lại embedding model
    embeddings = get_embedding_model()

    # Chỉ load index do chính mình tạo và quản lý
    vector_store = FAISS.load_local(
        folder_path=str(path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store
