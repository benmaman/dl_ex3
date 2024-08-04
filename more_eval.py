from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def text_similarity(text1, text2):
    """
    Calculate the cosine similarity between two texts using TF-IDF vectorization.

    Parameters:
    text1 (str): The first text to compare.
    text2 (str): The second text to compare.

    Returns:
    float: Cosine similarity score between the two texts.
    """

    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Transform the texts into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate the cosine similarity between the vectors
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    # Return the similarity score (the value is in a 2D array, so we take the first element)
    return similarity_matrix[0][0]

if __name__ == '__main__':
    # Example usage
    text1 = "This is a sample text."
    text2 = "This text is a sample."

    similarity_score = text_similarity(text1, text2)
    print(f"Cosine Similarity: {similarity_score}")