
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.utils.config import TOP_K, MAX_RETRIEVAL_DISTANCE


# 1. Khởi tạo Retriever
def create_retriever(
    vector_store: FAISS,
    top_k: int = TOP_K,
    max_distance: float = MAX_RETRIEVAL_DISTANCE,
) -> BaseRetriever:

    if top_k <= 0:
        raise ValueError("TOP_K phải lớn hơn 0")
    if not 0 <= max_distance <= 4:
        raise ValueError("max_distance phải nằm trong khoảng 0 đến 4")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        # FAISS IndexFlatL2 trả về bình phương khoảng cách, càng nhỏ càng gần.
        search_kwargs={"k": top_k, "score_threshold": max_distance}
    )

    return retriever


# 2. Truy xuất các chunks liên quan
def retrieve_documents(
    retriever: BaseRetriever,
    question: str
) -> list[Document]:

    if not question.strip():
        raise ValueError("Câu hỏi không được để trống")

    documents = retriever.invoke(question)

    return documents


# 3. Chuyển chunks thành context cho LLM
def format_context(
    documents: list[Document]
) -> str:

    if not documents:
        return ""

    contexts = []

    for i, doc in enumerate(documents, start=1):

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")

        page_label = (
            str(page + 1)
            if isinstance(page, int)
            else "Unknown"
        )

        context = (
            f"[Document {i}]\n"
            f"Source: {source}\n"
            f"Page: {page_label}\n"
            f"Content:\n{doc.page_content}"
        )

        contexts.append(context)

    return "\n\n".join(contexts)
