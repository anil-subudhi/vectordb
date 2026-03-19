from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Initialize embedding model and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="rag_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add documents
docs = [
    Document(page_content="ChromaDB stores vector embeddings", metadata={"source": "doc1"}),
    Document(page_content="LangChain simplifies LLM application development", metadata={"source": "doc2"}),
]
vector_store.add_documents(docs)

# Similarity search
results = vector_store.similarity_search("vector database", k=2)
for doc in results:
    print(doc.page_content)