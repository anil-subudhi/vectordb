"""ChromaDB example that uses OpenAI's embedding API for student/university search.

Prerequisites
-------------
1. Install the OpenAI Python SDK in the active environment:
   `pip install openai`
2. Export an API key (replace with your own key):
   `export OPENAI_API_KEY="sk-..."`  # Keep keys out of source control.

The script stores data in `./chroma_db`, so it can be run repeatedly without
losing prior inserts.
"""

from __future__ import annotations

import os
from pprint import pprint
from typing import List, Sequence, Union

import chromadb
from openai import OpenAI


class OpenAIEmbeddingFunction:
    """Embedding function that proxies to OpenAI's embeddings endpoint."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        encoding_format: str = "float",
    ) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is missing. "
                "Create a key in the OpenAI dashboard and export it before running this script."
            )

        self._client = OpenAI()
        self._model = model
        self._encoding_format = encoding_format

    def name(self) -> str:
        return f"openai:{self._model}"

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            encoding_format=self._encoding_format,
        )
        return [list(map(float, item.embedding)) for item in response.data]

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_documents(self, input: Sequence[str]) -> List[List[float]]:
        return self._embed(input)

    def embed_query(self, input: Union[str, Sequence[str]]) -> List[List[float]]:
        if isinstance(input, str):
            return self._embed([input])
        return self._embed(input)


def main() -> None:
    embedding_fn = OpenAIEmbeddingFunction()
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="Students_OpenAI",
        embedding_function=embedding_fn,
    )

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

    print("Top matches for 'student name' question using OpenAI embeddings:\n")
    results = collection.query(
        query_texts=["What is the student name?"],
        n_results=2,
    )
    pprint(results)


if __name__ == "__main__":
    main()
