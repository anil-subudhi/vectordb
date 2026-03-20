from sentence_transformers import SentenceTransformer
import numpy as np

# Load our two models
monolingual_model = SentenceTransformer('all-MiniLM-L6-v2')
multilingual_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Example sentences in different languages
sentences = {
    'english': "The weather is beautiful today.",
    'spanish': "El clima está hermoso hoy.",
    'french': "Le temps est magnifique aujourd'hui.",
    'german': "Das Wetter ist heute wunderschön."
}

# Function to compute similarity between embeddings
def compute_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Compare embeddings across languages
def compare_sentences(model, sentences):
    # Generate embeddings for all sentences
    print(sentences)
    embeddings = {
        lang: model.encode(text, convert_to_numpy=True)
        for lang, text in sentences.items()
        }
    
    # Compare each pair
    print(f"\nSimilarity scores for {model.__class__.__name__}:")
    for lang1 in sentences:
        for lang2 in sentences:
            if lang1 < lang2:  # Avoid duplicate comparisons
                sim = compute_similarity(embeddings[lang1], embeddings[lang2])
                print(f"{lang1} vs {lang2}: {sim:.4f}")

# Test both models
for model in [monolingual_model, multilingual_model]:
    compare_sentences(model, sentences)
    
# Example of batch processing for efficiency
texts = list(sentences.values())
batch_embeddings = multilingual_model.encode(texts, batch_size=8)
