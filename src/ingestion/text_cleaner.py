
import re
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Làm sạch văn bản được trích xuất từ PDF.
    """

    # 1. Thay tab bằng khoảng trắng
    text = text.replace("\t", " ")

    # 2. Xóa ký tự NULL
    text = text.replace("\x00", "")

    # Chuẩn hóa xuống dòng trước khi gộp các khoảng trắng khác.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)

    # 5. Xóa khoảng trắng đầu và cuối mỗi dòng
    text = "\n".join(
        line.strip() for line in text.split("\n")
    )

    # 6. Giảm số dòng trống liên tiếp
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    """
    Làm sạch danh sách Document.
    Giữ nguyên metadata của từng trang PDF.
    """

    cleaned_documents = []

    for doc in documents:
        cleaned_text = clean_text(doc.page_content)

        if not cleaned_text:
            continue

        cleaned_doc = Document(
            page_content=cleaned_text,
            metadata=doc.metadata.copy()
        )

        cleaned_documents.append(cleaned_doc)

    return cleaned_documents
