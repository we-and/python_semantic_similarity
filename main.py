from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example sentence pairs
sentence_pairs = [
    ("A dog is bigger than a mouse.", "A mouse is smaller than a dog."),
    ("I like to watch movies.", "I enjoy watching films."),
    ("The weather is sunny today.", "It is raining today.")
]

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to calculate similarity between pairs of sentences
def compute_similarity(model, sentence_pairs):
    embeddings = [model.encode([pair[0], pair[1]]) for pair in sentence_pairs]
    similarities = [cosine_similarity([emb[0]], [emb[1]])[0][0] for emb in embeddings]
    return similarities

# Calculate similarities for the provided pairs
similarities = compute_similarity(model, sentence_pairs)
for pair, similarity in zip(sentence_pairs, similarities):
    print(f"Sentence 1: {pair[0]}\nSentence 2: {pair[1]}\nSimilarity: {similarity:.4f}\n")