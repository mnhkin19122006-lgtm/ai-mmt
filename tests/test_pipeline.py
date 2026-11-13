from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import pytest
from src.embeddings.embedding_model import LocalEmbeddings
from src.generation import rag_chain
from src.ingestion.text_cleaner import clean_text

from src.retrieval.retriever import create_retriever, format_context, retrieve_documents
from src.vetorstore import faiss_strore


def test_chat_provider_configuration(monkeypatch):
    captured = {}

    def fake_chat(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(rag_chain, "ChatGroq", fake_chat)
    monkeypatch.setattr(rag_chain, "GROQ_API_KEY", "provider-key")
    monkeypatch.setattr(rag_chain, "GROQ_MODEL", "provider-model")
    rag_chain.get_llm()

    assert captured["api_key"] == "provider-key"
    assert "base_url" not in captured
    assert captured["model"] == "provider-model"


@pytest.mark.parametrize("key", [None, "", "gsk-your-api-key-here"])
def test_missing_groq_key_fails_before_client_creation(monkeypatch, key):
    monkeypatch.setattr(rag_chain, "GROQ_API_KEY", key)
    monkeypatch.setattr(rag_chain, "ChatGroq", lambda **kwargs: pytest.fail("Unexpected client creation"))
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        rag_chain.get_llm()


def test_embeddings_stay_local_with_legacy_environment(monkeypatch):
    from src.embeddings.embedding_model import get_embedding_model

    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-key")
    assert isinstance(get_embedding_model(), LocalEmbeddings)


def test_local_embeddings_are_deterministic_and_retrieve_matching_text(monkeypatch):
    monkeypatch.setattr(faiss_strore, "get_embedding_model", LocalEmbeddings)
    documents = [
        Document(page_content="TCP là giao thức truyền tải tin cậy."),
        Document(page_content="Mạng không dây sử dụng sóng vô tuyến."),
    ]
    store = faiss_strore.create_vector_store(documents)
    result = retrieve_documents(create_retriever(store, 1), "giao thức TCP truyền tải")
    assert result[0].page_content == documents[0].page_content
    assert LocalEmbeddings().embed_query("TCP") == LocalEmbeddings().embed_query("TCP")


def test_faiss_save_load_and_retrieve(tmp_path, monkeypatch):
    embeddings = LocalEmbeddings()
    monkeypatch.setattr(faiss_strore, "get_embedding_model", lambda: embeddings)
    monkeypatch.setattr(faiss_strore, "FAISS_INDEX_PATH", tmp_path / "index")

    document = Document(
        page_content="TCP là giao thức truyền tải.",
        metadata={"source": "bai-giang.pdf", "page": 4, "chunk_id": 0},
    )
    faiss_strore.save_vector_store(faiss_strore.create_vector_store([document]))
    loaded = faiss_strore.load_vector_store()
    results = retrieve_documents(create_retriever(loaded, 1), "TCP là gì?")

    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == 0
    assert "Page: 5" in format_context(results)


@pytest.mark.parametrize("separator", ["\r", "\r\n", "\n"])
def test_clean_text_preserves_line_breaks(separator):
    assert clean_text(f"  first \t{separator} second  ") == "first\nsecond"


def test_unrelated_query_does_not_call_chat(monkeypatch):
    monkeypatch.setattr(faiss_strore, "get_embedding_model", LocalEmbeddings)
    store = faiss_strore.create_vector_store([
        Document(page_content="TCP provides reliable network transport."),
    ])

    def unexpected_chat():
        pytest.fail("No Chat API call should be made without relevant documents")

    monkeypatch.setattr(rag_chain, "get_llm", unexpected_chat)
    result = rag_chain.generate_answer("chocolate cake recipe", create_retriever(store))
    assert result["sources"] == []
    assert result["answer"] == "Không tìm thấy tài liệu liên quan."


@pytest.mark.parametrize("max_distance", [-1, 5, float("nan")])
def test_invalid_retrieval_distance(max_distance):
    with pytest.raises(ValueError):
        create_retriever(None, max_distance=max_distance)


@pytest.mark.parametrize("question,rewritten,has_sources", [
    ("Cho ví dụ về nó", "Ví dụ giao thức TCP truyền tải tin cậy", True),
    ("chocolate cake recipe", "chocolate cake recipe", False),
])
def test_history_used_for_search_and_answer(monkeypatch, question, rewritten, has_sources):
    monkeypatch.setattr(faiss_strore, "get_embedding_model", LocalEmbeddings)
    store = faiss_strore.create_vector_store([
        Document(page_content="TCP là giao thức truyền tải tin cậy."),
    ])
    calls = []

    def fake_chat(prompt):
        calls.append(prompt.to_messages())
        return AIMessage(content=rewritten if len(calls) == 1 else "Ví dụ TCP.")

    monkeypatch.setattr(rag_chain, "get_llm", lambda: RunnableLambda(fake_chat))
    searches = []
    original_retrieve = rag_chain.retrieve_documents

    def capture_search(retriever, query):
        searches.append(query)
        return original_retrieve(retriever, query)

    monkeypatch.setattr(rag_chain, "retrieve_documents", capture_search)
    history = [{"question": "TCP là gì?", "answer": "TCP truyền tải tin cậy."}] * 8
    result = rag_chain.generate_answer(question, create_retriever(store), history)
    assert searches == [rewritten]
    assert bool(result["sources"]) == has_sources
    assert result["question"] == question
    assert len(calls) == (2 if has_sources else 1)
    for messages in calls:
        assert len(messages) == 14  # system + 6 pairs + current question
        assert messages[1].content == history[-6]["question"]
        assert messages[2].content == history[-6]["answer"]
        assert messages[-1].content == question
    assert len(history) == 8


def test_chat_ui_passes_history(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from src.utils import config
    from src.retrieval import retriever as retrieval_module

    (tmp_path / "index.faiss").touch()
    (tmp_path / "index.pkl").touch()
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", tmp_path)
    monkeypatch.setattr(faiss_strore, "load_vector_store", lambda: object())
    monkeypatch.setattr(retrieval_module, "create_retriever", lambda *args: object())
    histories = []

    def fake_answer(question, retriever, history=None):
        histories.append(list(history or []))
        return {"question": question, "answer": "TCP example", "sources": []}

    monkeypatch.setattr(rag_chain, "generate_answer", fake_answer)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run()
    app.chat_input[0].set_value("What is TCP?").run()
    app.chat_input[0].set_value("Give an example").run()
    assert not app.exception
    assert not app.error
    assert histories[0] == []
    assert histories[1][0]["question"] == "What is TCP?"
    assert len(app.session_state["history"]) == 2
