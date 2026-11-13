
from langchain_core.retrievers import BaseRetriever

from src.retrieval.retriever import retrieve_documents
from src.generation.rag_chain import generate_answer


# 1. Tính các chỉ số Retrieval
def calculate_metrics(
    retrieved_ids: list[int],
    relevant_ids: list[int],
    k: int
) -> dict:

    if k <= 0:
        raise ValueError("K phải lớn hơn 0")

    if not relevant_ids:
        raise ValueError(
            "Cần có ít nhất một chunk đúng để đánh giá"
        )

    # Chỉ lấy Top K chunks
    retrieved_ids = retrieved_ids[:k]

    # Các chunk đúng được gán nhãn trước
    relevant_set = set(relevant_ids)

    # Số chunk đúng được tìm thấy
    correct = len(set(retrieved_ids) & relevant_set)

    # Precision@K
    precision = correct / k

    # Recall@K
    recall = correct / len(relevant_set)

    # Hit@K
    hit = 1 if correct > 0 else 0

    # Reciprocal Rank: vị trí chunk đúng đầu tiên
    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            reciprocal_rank = 1 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "hit": hit,
        "reciprocal_rank": reciprocal_rank
    }


# 2. Đánh giá Retriever với nhiều câu hỏi
def evaluate_retrieval(
    retriever: BaseRetriever,
    test_cases: list[dict],
    k: int = 3
) -> dict:

    if not test_cases:
        raise ValueError("Danh sách test_cases rỗng")

    results = []

    for case in test_cases:

        question = case["question"]
        relevant_ids = case["relevant_chunk_ids"]

        # Tìm tài liệu bằng Retriever
        documents = retrieve_documents(
            retriever,
            question
        )

        # Lấy chunk_id từ metadata
        retrieved_ids = []

        for doc in documents[:k]:
            chunk_id = doc.metadata.get("chunk_id")

            if not isinstance(chunk_id, int):
                raise ValueError(
                    "Document thiếu chunk_id hợp lệ"
                )

            retrieved_ids.append(chunk_id)

        # Tính metric
        metrics = calculate_metrics(
            retrieved_ids,
            relevant_ids,
            k
        )

        results.append({
            "question": question,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            **metrics
        })

    # Tính trung bình trên tất cả câu hỏi
    n = len(results)

    summary = {
        "precision_at_k": sum(
            r["precision"] for r in results
        ) / n,

        "recall_at_k": sum(
            r["recall"] for r in results
        ) / n,

        "hit_at_k": sum(
            r["hit"] for r in results
        ) / n,

        "mrr_at_k": sum(
            r["reciprocal_rank"] for r in results
        ) / n
    }

    return {
        "total_questions": n,
        "k": k,
        "summary": summary,
        "details": results
    }


# 3. Thu thập câu trả lời AI để đánh giá
def evaluate_generation(
    retriever: BaseRetriever,
    questions: list[str]
) -> list[dict]:

    results = []

    for question in questions:

        # Gọi RAG Chain
        result = generate_answer(
            question=question,
            retriever=retriever
        )

        # Lưu nội dung và nguồn để kiểm tra
        sources = []

        for doc in result["sources"]:
            sources.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "page": doc.metadata.get("page"),
                "content": doc.page_content
            })

        results.append({
            "question": question,
            "answer": result["answer"],
            "sources": sources
        })

    return results
