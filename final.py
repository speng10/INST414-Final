import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import euclidean_distances, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to preprocess data
def preprocess_data(filepath):
    """
    Load and preprocess data from the given file path.
    """
    df = pd.read_csv(filepath)

    # Ensure necessary columns exist
    required_columns = [
        'track_id', 'track_name', 'artists', 'track_genre', 'popularity',
        'explicit', 'danceability', 'energy', 'key', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_columns)

    # Remove duplicate songs based on track name and artists
    df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first')

    # Reset index for easier handling
    df.reset_index(drop=True, inplace=True)

    return df


# Function to calculate Euclidean distances
def calculate_euclidean_similarity(features, target_index, data_frame):
    """
    Calculate Euclidean distances between the target track and all other tracks.
    """
    target_features = data_frame.loc[target_index, features].values.reshape(1, -1)
    distances = euclidean_distances(data_frame[features], target_features)[:, 0]

    # Pair each track with its distance
    distances = list(zip(data_frame.index, distances))
    distances.sort(key=lambda x: x[1])  # Sort by distance

    return distances

# Function to calculate Euclidean similarity for individual features
def calculate_feature_similarity(feature, target_index, data_frame):
    """
    Calculate Euclidean distances for a single feature.
    """
    target_value = data_frame.loc[target_index, feature]
    distances = abs(data_frame[feature] - target_value)

    # Pair each track with its distance
    distances = list(zip(data_frame.index, distances))
    distances.sort(key=lambda x: x[1])  # Sort by distance

    return distances

# Function to train Random Forest model
def train_random_forest(X, y):
    """
    Train a Random Forest Classifier and evaluate feature importance.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_classifier.fit(X_train, y_train)

    # Evaluate accuracy
    accuracy = rf_classifier.score(X_test, y_test)
    print(f"Random Forest Accuracy: {accuracy:.2f}")

    # Classification report and confusion matrix
    y_pred = rf_classifier.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = rf_classifier.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
    print(feature_importance)

    return rf_classifier

# Main script
def main():
    # Load data
    filepath = 'spotify_tracks.csv'
    df = preprocess_data(filepath)

    # Define features for analysis
    numeric_features = ['popularity', 'danceability', 'energy', 'loudness',
                        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    # Encode track genres
    label_encoder = LabelEncoder()
    if 'track_genre' in df.columns:
        df['track_genre_encoded'] = label_encoder.fit_transform(df['track_genre'])

    # Define the target track by name and artist
    target_track_name = "Stand by Me"
    target_artist_name = "Ben E. King"

    # Find the target track in the DataFrame
    target_track = df[(df['track_name'] == target_track_name) & (df['artists'] == target_artist_name)]

    if target_track.empty:
        raise ValueError(f"The track '{target_track_name}' by '{target_artist_name}' was not found in the dataset.")
    elif len(target_track) > 1:
        print(f"Warning: Multiple entries found for '{target_track_name}' by '{target_artist_name}'. Using the first match.")

    target_track_index = target_track.index[0]  # Use the index of the first match
    
    # Calculate similarity for the target track
    print(f"\nTarget Track: '{target_track_name}' by '{target_artist_name}'")

    target_track_features = target_track[numeric_features].values.reshape(1, -1)

    # Euclidean distances for all tracks
    euclidean_distances_list = calculate_euclidean_similarity(numeric_features, target_track_index, df)
    print("Top 5 most similar tracks by overall Euclidean distance:")
    for idx, dist in euclidean_distances_list[:5]:
        print(f"{df.loc[idx, 'track_name']} by {df.loc[idx, 'artists']} (Distance: {dist:.2f})")

    # Train Random Forest model
    if 'track_genre_encoded' in df.columns:
        X = df[numeric_features]
        y = df['track_genre_encoded']
        rf_classifier = train_random_forest(X, y)

        # target track genre
        actual_genre = df.loc[target_track_index, 'track_genre']
        print(f"\nReal Genre of Target Track: {actual_genre}")
        # Predict genre for the target track
        target_track_features = df.loc[target_track_index, numeric_features].values.reshape(1, -1)
        predicted_genre = label_encoder.inverse_transform(rf_classifier.predict(target_track_features))[0]
        print(f"\nPredicted Genre for Target Track: {predicted_genre}")

        # Recommend tracks from the same predicted genre
        same_genre_tracks = df[df['track_genre'] == predicted_genre]
        same_genre_distances = euclidean_distances(same_genre_tracks[numeric_features], target_track_features)[:, 0]
        same_genre_recommendations = list(zip(same_genre_tracks.index, same_genre_distances))

        print(f"\nTop Recommended Tracks in the Predicted Genre '{predicted_genre}':")
        for idx, distance in sorted(same_genre_recommendations, key=lambda x: x[1])[:10]:
            print(f"{df.loc[idx, 'track_name']} by {df.loc[idx, 'artists']} - Genre: {df.loc[idx, 'track_genre']} - Distance: {distance:.4f}")

if __name__ == "__main__":
    main()