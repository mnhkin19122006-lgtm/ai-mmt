
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(
    documents: list[Document]
) -> list[Document]:
    """
    Chia danh sách Document thành các chunks nhỏ.
    Giữ lại metadata để truy xuất nguồn tài liệu.
    """

    if not documents:
        return []

    # Kiểm tra cấu hình
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE phải lớn hơn 0")

    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP phải >= 0 và nhỏ hơn CHUNK_SIZE"
        )

    # Khởi tạo Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Chia văn bản thành chunks
    chunks = text_splitter.split_documents(documents)

    # Gắn số thứ tự chunk
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks
