import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('sentence_pairs.csv')  # Ensure your dataset has columns 'sentence1' and 'sentence2'

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(df, batch_size=100):
    similarities = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        embeddings1 = model.encode(batch['sentence1'].tolist(), convert_to_tensor=True, show_progress_bar=True)
        embeddings2 = model.encode(batch['sentence2'].tolist(), convert_to_tensor=True, show_progress_bar=True)

        # Move tensors to CPU and convert to numpy arrays before calculating cosine similarity
        embeddings1 = embeddings1.cpu().numpy()
        embeddings2 = embeddings2.cpu().numpy()

        # Compute cosine similarity
        batch_similarity = cosine_similarity(embeddings1, embeddings2)
        # Extract the diagonal elements which represent the similarity between corresponding pairs
        similarities.extend(batch_similarity.diagonal())
    return similarities


# Calculate similarities
df['similarity'] = calculate_similarity(df)

# Optionally save the results to a new CSV
df.to_csv('enriched_dataset_with_similarities.csv', index=False)
