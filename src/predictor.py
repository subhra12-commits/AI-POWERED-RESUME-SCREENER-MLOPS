from sentence_transformers import util

def rank_resumes(jd_vector, resume_vectors, original_resumes):
    similarity_scores = util.cos_sim(jd_vector, resume_vectors).flatten()
    ranked = sorted(zip(original_resumes, similarity_scores), key=lambda x: x[1], reverse=True)
    return [(res, float(score)) for res, score in ranked]
