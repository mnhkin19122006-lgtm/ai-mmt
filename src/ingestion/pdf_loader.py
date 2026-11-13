
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    # Chuyển đường dẫn thành đối tượng Path
    path = Path(file_path)

    # Kiểm tra file có tồn tại không
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file PDF: {file_path}"
        )

    # Kiểm tra định dạng file
    if path.suffix.lower() != ".pdf":
        raise ValueError("File phải có định dạng PDF")

    # Khởi tạo PDF Loader
    loader = PyPDFLoader(str(path), mode="page")

    # Trích xuất nội dung PDF
    documents = loader.load()

    return documents
