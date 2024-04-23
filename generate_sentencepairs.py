import pandas as pd
import random

# Define some sample sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I love eating pizza on weekends.",
    "She enjoys playing tennis in the park.",
    "Artificial intelligence will shape the future of humanity.",
    "The car broke down on the way to the store.",
    "He reads a book once every week.",
    "Watching movies at night is fun.",
    "The weather today is gloomy and cold.",
    "I am learning to play the guitar.",
    "Technology is evolving at a rapid pace."
]

# Generate sentence pairs
def generate_sentence_pairs(n):
    pairs = []
    for _ in range(n):
        s1 = random.choice(sentences)
        s2 = random.choice(sentences)
        pairs.append([s1, s2])
    return pairs

# Number of pairs you want to generate
num_pairs = 10000  # Adjust this number based on how large you want the dataset

# Generate the pairs
sentence_pairs = generate_sentence_pairs(num_pairs)

# Create a DataFrame
df = pd.DataFrame(sentence_pairs, columns=['Sentence1', 'Sentence2'])

# Save to CSV
df.to_csv('sentence_pairs.csv', index=False)

print(f"Generated CSV file with {num_pairs} sentence pairs.")