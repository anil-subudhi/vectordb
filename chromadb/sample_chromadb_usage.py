# sample_chromadb_usage.py
# Sample script demonstrating basic ChromaDB usage

import chromadb

def main():
    # Create or connect to a local ChromaDB in-memory
    client = chromadb.Client()

    # Create a collection named "my_collection"
    collection = client.create_collection(name="my_collection")

    # Add some sample documents with vectors
    collection.add(
        documents=["Document 1", "Document 2", "Document 3"],
        embeddings=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.25, 0.25, 0.25]],
        ids=["doc1", "doc2", "doc3"]
    )

    # Query the collection with a sample vector
    results = collection.query(
        query_embeddings=[[0.2, 0.2, 0.2]],
        n_results=2
    )

    print("Query results:", results)

if __name__ == "__main__":
    main()