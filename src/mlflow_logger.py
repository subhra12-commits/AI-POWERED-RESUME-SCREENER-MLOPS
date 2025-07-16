# # src/mlflow_logger.py

# import mlflow
# import datetime
# import os

# def log_resume_ranking(jd, top_matches):
#     """
#     Logs the job description and top-ranked resumes with their match scores to MLflow.
    
#     Args:
#         jd (str): Job description text.
#         top_matches (list of tuples): Each tuple is (resume_text, similarity_score).
#     """
#     with mlflow.start_run(run_name="Resume Screening"):
#         # Log JD
#         mlflow.log_param("Job_Description", jd[:250])  # log first 250 chars only

#         # Log top resume scores
#         for i, (resume_text, score) in enumerate(top_matches):
#             mlflow.log_metric(f"Resume_{i+1}_Score", round(score * 100, 2))

#         # Save top resumes to a file
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         file_path = f"top_resumes_{timestamp}.txt"
#         with open(file_path, "w", encoding="utf-8") as f:
#             for i, (resume, score) in enumerate(top_matches):
#                 f.write(f"\n===== Resume {i+1} (Score: {score:.2%}) =====\n")
#                 f.write(resume + "\n\n")

#         mlflow.log_artifact(file_path)

#         # Clean up the file after logging
#         os.remove(file_path)


import mlflow
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered = [w for w in words if w not in stop_words]
    return " ".join(filtered)


def get_bert_vectors(jd, resumes):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode([jd] + resumes)
    jd_vec = embeddings[0].reshape(1, -1)
    resume_vecs = embeddings[1:]
    return jd_vec, resume_vecs


def rank_resumes(jd_vec, resume_vecs, resumes):
    similarities = cosine_similarity(jd_vec, resume_vecs)[0]
    ranked = sorted(zip(resumes, similarities), key=lambda x: x[1], reverse=True)
    return ranked


# # ✅ This is the function you'll import in app.py
# def  log_resume_ranking(job_description: str):
#     mlflow.set_tracking_uri("http://127.0.0.1:5000")
#     mlflow.set_experiment("Resume_Screener_BERT")

#     with mlflow.start_run():
#         mlflow.log_param("embedding_model", "MiniLM-L6-v2")
#         mlflow.log_param("job_description", job_description[:100])

#         # Load resume dataset
#         df = pd.read_csv(r"E:/resume-screener-mlops/archive/UpdatedResumeDataSet.csv")
#         resumes = df['Resume'].astype(str).tolist()

#         # Clean
#         resumes_clean = [clean_text(r) for r in resumes]
#         jd_clean = clean_text(job_description)

#         # Vectorize & Rank
#         jd_vec, resume_vecs = get_bert_vectors(jd_clean, resumes_clean)
#         ranked = rank_resumes(jd_vec, resume_vecs, resumes)

#         # Log Top 5 Scores
#         for i, (resume_text, score) in enumerate(ranked[:5]):
#             mlflow.log_metric(f"Top{i+1}_MatchScore", round(score * 100, 2))

#         # Save Top Matches
#         os.makedirs("outputs", exist_ok=True)
#         output_path = os.path.join("outputs", "top_matches.txt")
#         with open(output_path, "w", encoding="utf-8") as f:
#             for i, (resume, score) in enumerate(ranked[:5]):
#                 f.write(f"Resume {i+1} - Score: {score:.2%}\n")
#                 f.write(resume + "\n\n")

#         mlflow.log_artifact(output_path)




def log_resume_ranking(job_description: str, results: list):
    import mlflow
    import os

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Resume_Screener_BERT")

    with mlflow.start_run():
        mlflow.log_param("embedding_model", "MiniLM-L6-v2")
        mlflow.log_param("job_description", job_description[:100])

        # Log Top 5 Scores directly from results
        for i, (resume_text, score) in enumerate(results):
            mlflow.log_metric(f"Top{i+1}_MatchScore", round(score * 100, 2))

        # Save Top Matches
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", "top_matches.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for i, (resume, score) in enumerate(results):
                f.write(f"Resume {i+1} - Score: {score:.2%}\n")
                f.write(resume + "\n\n")

            mlflow.log_artifact(output_path)
    print("✅ Logged to MLflow.")
    return results  # ✅ Return top results to display in Flask UI
