from sentence_transformers import SentenceTransformer

# Load the model once
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast & accurate

def get_bert_vectors(jd, resumes):
    # jd: string, resumes: list of strings
    all_text = [jd] + resumes
    embeddings = model.encode(all_text, convert_to_tensor=True)
    jd_vector = embeddings[0]
    resume_vectors = embeddings[1:]
    return jd_vector, resume_vectors
