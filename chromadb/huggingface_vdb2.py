from sentence_transformers import SentenceTransformer
from chromadb import Client, Settings
import chromadb

# Initialize the embedding model
# all-MiniLM-L6-v2 is a lightweight but effective model, a good balance of 
# speed/quality
#Models : all-MiniLM-L6-v2 / all-mpnet-base-v2 / paraphrase-multilingual-mpnet-base-v2 / 
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB as our vector store
# Using in-memory storage for this example
chroma_client = Client(Settings(is_persistent=False))
collection = chroma_client.create_collection(name="climate_docs")

# Example documents to process
documents = [
    "Climate change is affecting global weather patterns, causing more extreme events.",
    "Rising sea levels threaten coastal communities worldwide.",
    "Greenhouse gas emissions continue to rise despite international  agreements."
]

# Create embeddings for our documents
# model.encode() converts text to dense vectors (embeddings)
embeddings = model.encode(documents)

# Store documents and their embeddings
# ChromaDB expects embeddings as lists, so we convert numpy arrays
collection.add(
    embeddings=[e.tolist() for e in embeddings],
    documents=documents,    
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# Process a query
query = "How does climate change affect weather?"
query_embedding = model.encode(query)

# Search for similar documents
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=2
)


# Print results
for doc in results['documents'][0]:
    print(f"Retrieved document: {doc}")
    
    
    