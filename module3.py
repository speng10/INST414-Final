import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Load the Spotify data
df = pd.read_csv('spotify_tracks.csv')

print(df.head())

# Define different feature sets for comparison
features_set1 = ['popularity']  # Compare based on popularity
features_set2 = ['danceability']  # Compare based on danceability
features_set3 = ['tempo']  # Compare based on tempo

# Convert necessary columns to numeric
for col in features_set1 + features_set2 + features_set3:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=features_set1 + features_set2 + features_set3)
df.reset_index(inplace=True, drop=True)

# Get names and genres for display purposes
names = df[['artists', 'track_name', 'track_genre']].values.tolist()

# Get first target track 
target_track_index = 90000  # Index of the track to compare
print(target_track_index)
target_track = df.loc[target_track_index, features_set1 + features_set2 + features_set3].astype(float)

# Calculate distances for popularity
popularity_distances = euclidean_distances(df[features_set1], [target_track[0:1]])[:, 0]
popularity_query_distances = list(zip(df.index, popularity_distances))

# Calculate distances for danceability
danceability_distances = euclidean_distances(df[features_set2], [target_track[1:2]])[:, 0]
danceability_query_distances = list(zip(df.index, danceability_distances))

# Calculate distances for tempo
tempo_distances = euclidean_distances(df[features_set3], [target_track[2:3]])[:, 0]
tempo_query_distances = list(zip(df.index, tempo_distances))

# Print most similar tracks by popularity
print("Most Similar Tracks by Popularity:")
for idx, distance in sorted(popularity_query_distances, key=lambda x: x[1])[:10]:
    print(f"{names[idx][0]} - {names[idx][1]} - Genre: {names[idx][2]} - Distance: {distance}")

# Print most similar tracks by duration
print("\nMost Similar Tracks by Danceability:")
for idx, distance in sorted(danceability_query_distances, key=lambda x: x[1])[:10]:
    print(f"{names[idx][0]} - {names[idx][1]} - Genre: {names[idx][2]} - Distance: {distance}")

# Print most similar tracks by tempo
print("\nMost Similar Tracks by Tempo:")
for idx, distance in sorted(tempo_query_distances, key=lambda x: x[1])[:10]:
    print(f"{names[idx][0]} - {names[idx][1]} - Genre: {names[idx][2]} - Distance: {distance}")


# Define a function to calculate Jaccard similarity for genres
def calculate_jaccard_similarity(genre_A, genre_B):
    # Convert genre strings to sets
    set_A = set(genre_A.split(','))
    set_B = set(genre_B.split(','))
    
    intersection = len(set_A.intersection(set_B))
    union = len(set_A.union(set_B))
    
    return intersection / union if union != 0 else 0

# Calculate Jaccard similarity for the target track's genre
jaccard_similarities = []
for idx in range(len(df)):
    similarity = calculate_jaccard_similarity(df.loc[target_track_index, 'track_genre'], df.loc[idx, 'track_genre'])
    jaccard_similarities.append((idx, similarity))
    
    
# Print most similar tracks by Jaccard similarity for genres
print("\nMost Similar Tracks by Genre (Jaccard Similarity):")
for idx, similarity in sorted(jaccard_similarities, key=lambda x: x[1], reverse=True)[:10]:
    print(f"{names[idx][0]} - {names[idx][1]} - Genre: {names[idx][2]} - Jaccard Similarity: {similarity:.4f}")