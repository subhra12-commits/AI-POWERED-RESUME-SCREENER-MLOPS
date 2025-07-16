# test_feature.py

from src.feature_engineering import get_tfidf_vectors
from src.preprocessing import clean_text
from sklearn.metrics.pairwise import cosine_similarity

def main():
    jd = "Looking for a data scientist with Python, Flask, SQL, and ML experience"
    resumes = [
        "Experienced in Python, ML, and SQL",
        "Flask developer with Python knowledge",
        "Worked on machine learning and deep learning"
    ]

    # Clean text
    jd_clean = clean_text(jd)
    resumes_clean = [clean_text(r) for r in resumes]

    # Get vectors
    jd_vector, resume_vectors = get_tfidf_vectors(jd_clean, resumes_clean)

    # Print vector shapes
    print("✅ JD vector shape:", jd_vector.shape)
    print("✅ Resumes vector shape:", resume_vectors.shape)

    # Compute similarity
    similarity_scores = cosine_similarity(jd_vector, resume_vectors).flatten()

    # Print similarity scores
    print("\n🔍 Similarity Scores:")
    for i, score in enumerate(similarity_scores):
        print(f"Resume {i+1} Score: {score:.4f}")

# 🧠 This makes sure the script runs when executed directly
if __name__ == "__main__":
    main()
