"""Streamlit interface for the computer networks tutor."""

from pathlib import Path

import streamlit as st

from src.generation.rag_chain import generate_answer
from src.ingestion.index_documents import build_index
from src.retrieval.retriever import create_retriever
from src.utils.config import FAISS_INDEX_PATH, PDF_DIRECTORY, TOP_K
from src.vetorstore.faiss_strore import load_vector_store


st.set_page_config(page_title="NetTutor AI", page_icon="🌐")
st.title("NetTutor AI")
st.caption("Hỏi đáp môn Mạng máy tính dựa trên bài giảng PDF")

with st.sidebar:
    st.header("Tài liệu")
    pdfs = sorted(PDF_DIRECTORY.glob("*.pdf"))
    if pdfs:
        for pdf in pdfs:
            st.write(f"• {pdf.name}")
    else:
        st.info(f"Đặt file PDF vào {PDF_DIRECTORY}")

    index_ready = (FAISS_INDEX_PATH / "index.faiss").is_file() and (
        FAISS_INDEX_PATH / "index.pkl"
    ).is_file()
    st.write("Chỉ mục: " + ("đã có" if index_ready else "chưa tạo"))
    if st.button("Tạo lại chỉ mục", disabled=not pdfs):
        try:
            with st.spinner("Đang đọc PDF và tạo embeddings..."):
                pdf_count, chunk_count = build_index()
            st.session_state.pop("vector_store", None)
            st.success(f"Đã lập chỉ mục {pdf_count} PDF, {chunk_count} đoạn.")
            index_ready = True
        except Exception as exc:
            st.error(f"Không thể tạo chỉ mục: {exc}")

if not index_ready:
    st.info(
        "Nhấn ‘Tạo lại chỉ mục’ để bắt đầu. "
        "Embedding chạy cục bộ, không cần API key."
    )
    st.stop()

question = st.chat_input("Nhập câu hỏi về Mạng máy tính")
for entry in st.session_state.get("history", []):
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        for source in entry["sources"]:
            st.caption(source)

if question:
    try:
        with st.spinner("Đang tìm tài liệu và trả lời..."):
            if "vector_store" not in st.session_state:
                st.session_state.vector_store = load_vector_store()
            retriever = create_retriever(st.session_state.vector_store, TOP_K)
            result = generate_answer(
                question, retriever, history=st.session_state.get("history", [])
            )
        sources = []
        for doc in result["sources"]:
            source = Path(doc.metadata.get("source", "Không rõ nguồn")).name
            page = doc.metadata.get("page")
            label = f"{source}, trang {page + 1}" if isinstance(page, int) else source
            if label not in sources:
                sources.append(label)
        st.session_state.setdefault("history", []).append(
            {"question": question, "answer": result["answer"], "sources": sources}
        )
        st.rerun()
    except Exception as exc:
        st.error(f"Không thể trả lời: {exc}")
