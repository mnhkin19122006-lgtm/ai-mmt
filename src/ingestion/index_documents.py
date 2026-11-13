"""Build the local search index from every PDF in the configured directory."""

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_cleaner import clean_documents
from src.ingestion.text_spiltter import split_documents
from src.utils.config import PDF_DIRECTORY
from src.vetorstore.faiss_strore import create_vector_store, save_vector_store


def build_index() -> tuple[int, int]:
    pdf_paths = sorted(PDF_DIRECTORY.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Không tìm thấy PDF trong {PDF_DIRECTORY}")

    pages = []
    for pdf_path in pdf_paths:
        pages.extend(clean_documents(load_pdf(str(pdf_path))))

    chunks = split_documents(pages)
    if not chunks:
        raise ValueError("Không trích xuất được văn bản từ các PDF.")

    save_vector_store(create_vector_store(chunks))
    return len(pdf_paths), len(chunks)


if __name__ == "__main__":
    pdf_count, chunk_count = build_index()
    print(f"Indexed {pdf_count} PDF files and {chunk_count} text chunks.")
