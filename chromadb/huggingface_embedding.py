"""ChromaDB example that uses a Hugging Face sentence-transformer model for embeddings.

Prerequisites
-------------
1. Install the Sentence Transformers package (includes Hugging Face dependencies):
   `pip install sentence-transformers`
2. Optional: set `HF_HOME` if you want to control where the model cache lives.

The script persists data in `./chroma_db`, allowing repeated runs without reloading
documents. The default model (`sentence-transformers/all-MiniLM-L6-v2`) is light
enough to run on CPU for quick experiments.
"""

from __future__ import annotations

import os
from pprint import pprint
from typing import List, Sequence, Union

import chromadb
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbeddingFunction:
    """Embedding function backed by a local SentenceTransformer model."""

    def __init__(self, model_id: str | None = None, device: str = "cpu") -> None:
        resolved_model = model_id or os.getenv(
            "HUGGINGFACE_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
        )
        token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv(
            "HUGGINGFACEHUB_API_TOKEN"
        )
        token_kwargs = {"use_auth_token": token} if token else {}

        self._model_id = resolved_model
        self._model_url = f"https://huggingface.co/{resolved_model}"
        self._model = SentenceTransformer(resolved_model, device=device, **token_kwargs)

    def name(self) -> str:
        return f"huggingface:{self._model_id}"

    @property
    def model_url(self) -> str:
        return self._model_url

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        val = [vector.astype(float).tolist() for vector in embeddings]
        print(val)
        return val

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_documents(self, input: Sequence[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_query(self, input: Union[str, Sequence[str]]) -> List[List[float]]:
        if isinstance(input, str):
            return self._embed([input])
        return self._embed(input)


def main() -> None:
    embedding_fn = HuggingFaceEmbeddingFunction()
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="Students_HuggingFace",
        embedding_function=embedding_fn,
    )

    print(f"Using Hugging Face model: {embedding_fn.name()}")
    print(f"Model URL: {embedding_fn.model_url}\n")

    student_info = (
        "Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA, "
        "is a member of the programming and chess clubs who enjoys pizza, swimming, "
        "and hiking in her free time in hopes of working at a tech company after "
        "graduating from the University of Washington."
    )

    club_info = (
        "The university chess club provides an outlet for students to come together "
        "and enjoy playing the classic strategy game of chess. Members of all skill "
        "levels are welcome, from beginners learning the rules to experienced "
        "tournament players. The club typically meets a few times per week to play "
        "casual games, participate in tournaments, analyze famous chess matches, "
        "and improve members' skills."
    )

    university_info = (
        "The University of Washington, founded in 1861 in Seattle, is a public "
        "research university with over 45,000 students across three campuses in "
        "Seattle, Tacoma, and Bothell. As the flagship institution of the six public "
        "universities in Washington state, UW encompasses over 500 buildings and "
        "20 million square feet of space, including one of the largest library "
        "systems in the world."
    )

    collection.upsert(
        documents=[student_info, club_info, university_info],
        metadatas=[
            {"source": "student info"},
            {"source": "club info"},
            {"source": "university info"},
        ],
        ids=["student_profile", "chess_club", "university_profile"],
    )

    print("Top matches for 'student name' question using Hugging Face embeddings:\n")
    results = collection.query(
        query_texts=["What is the student name?"],
        n_results=2,
    )
    pprint(results)


if __name__ == "__main__":
    main()
