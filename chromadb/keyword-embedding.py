"""Sample ChromaDB flow using student and university data."""

from pprint import pprint
from typing import List, Union

import chromadb


class KeywordEmbedding:
    """Very small embedding function keyed on domain-specific terms."""

    KEYWORDS = (
        "engineering",
        "computer",
        "business",
        "arts",
        "ai",
        "robotics",
        "finance",
        "research",
        "design",
        "health",
    )

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in input:
            lowered = text.lower()
            vector = [float(lowered.count(keyword)) for keyword in self.KEYWORDS]
            embeddings.append(vector)
        return embeddings

    def name(self) -> str:
        return "keyword-embedding"

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_query(self, input: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(input, str):
            return self([input])
        return self(input)


def main() -> None:
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="student_search",
        embedding_function=KeywordEmbedding(),
    )

    students = [
        {
            "id": "student_alice",
            "description": (
                "Alice Johnson is a computer science undergraduate at Stanford University "
                "focused on AI research, robotics labs, and interdisciplinary design projects."
            ),
            "metadata": {
                "student_name": "Alice Johnson",
                "university": "Stanford University",
                "degree_level": "bachelors",
                "focus_area": "artificial intelligence",
            },
        },
        {
            "id": "student_brian",
            "description": (
                "Brian Smith is pursuing an MBA at the Wharton School with an emphasis on "
                "finance, entrepreneurship incubators, and applied business analytics."
            ),
            "metadata": {
                "student_name": "Brian Smith",
                "university": "University of Pennsylvania (Wharton)",
                "degree_level": "masters",
                "focus_area": "finance",
            },
        },
        {
            "id": "student_carmen",
            "description": (
                "Carmen Lee is a biomedical engineering doctoral candidate at MIT working on "
                "robotics-assisted surgery and healthcare design innovation."
            ),
            "metadata": {
                "student_name": "Carmen Lee",
                "university": "MIT",
                "degree_level": "phd",
                "focus_area": "biomedical engineering",
            },
        },
        {
            "id": "student_diego",
            "description": (
                "Diego Fernandez studies visual arts at UCLA with projects that blend digital "
                "design, interactive media, and creative robotics installations."
            ),
            "metadata": {
                "student_name": "Diego Fernandez",
                "university": "UCLA",
                "degree_level": "bachelors",
                "focus_area": "visual arts",
            },
        },
    ]

    collection.upsert(
        ids=[student["id"] for student in students],
        documents=[student["description"] for student in students],
        metadatas=[student["metadata"] for student in students],
    )

    print("Top matches for a robotics-centric search query:\n")
    robotics_results = collection.query(
        query_texts=["student exploring robotics research in an engineering program"],
        n_results=2,
    )
    pprint(robotics_results)

    print("\nFiltering to masters programs focused on finance:\n")
    finance_results = collection.query(
        query_texts=["finance and business analytics"],
        n_results=3,
        where={"degree_level": "masters"},
    )
    pprint(finance_results)


if __name__ == "__main__":
    main()
