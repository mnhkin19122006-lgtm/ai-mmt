
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import HashingVectorizer



class LocalEmbeddings(Embeddings):
    """Deterministic character n-gram vectors for offline PDF search."""

    def __init__(self):
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), n_features=2048,
            alternate_sign=False, norm="l2",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# 1. Khởi tạo Embedding Model
def get_embedding_model() -> Embeddings:

    return LocalEmbeddings()


# 2. Chuyển các đoạn văn bản thành vector
def embed_texts(texts: list[str]) -> list[list[float]]:

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Danh sách chứa văn bản rỗng")

    embeddings = get_embedding_model()

    vectors = embeddings.embed_documents(texts)

    return vectors


# 3. Chuyển câu hỏi thành vector
def embed_question(question: str) -> list[float]:

    if not question.strip():
        raise ValueError("Câu hỏi không được để trống")

    embeddings = get_embedding_model()

    vector = embeddings.embed_query(question)

    return vector
