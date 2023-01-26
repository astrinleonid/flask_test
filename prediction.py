from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import nltk
from nltk import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')


def increase_percentage(percentage):
    return min(100, percentage + (100 - percentage) * 0.7)


def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z]+", ' ', text)
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    # Join the words back into a single string
    return " ".join(words)


def model(df, text, num_events=20):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the data into numerical vectors
    df_vectors = vectorizer.fit_transform(df['description'])
    text = vectorizer.transform([preprocess(text)])

    # Find the similarity between the given event from New York and the events in Tel Aviv
    cosine_similarities = cosine_similarity(text, df_vectors)

    # Find the most similar events in Tel Aviv
    most_similar_indexes = cosine_similarities.argsort()[0][-num_events:][::-1]
    most_similar_events = df.iloc[most_similar_indexes].copy()
    most_similar_events.loc[:, 'similarity_percentage'] = cosine_similarities[0][most_similar_indexes] * 100
    most_similar_events.loc[:, 'similarity_percentage'] = most_similar_events['similarity_percentage'].map(increase_percentage)
    return pd.DataFrame(most_similar_events)

