
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.mlflow_logger import log_resume_ranking
# At top of app.py
print("✅ File Exists:", os.path.isfile("src/mlflow_logger.py"))
# Before import
import importlib.util
spec = importlib.util.spec_from_file_location("mlflow_logger", "src/mlflow_logger.py")
print("✅ Can Load:", spec is not None)
from flask import Flask, request, render_template
from src.data_loader import load_dataset
from src.preprocessing import clean_text
from src.feature_engineering import get_bert_vectors
from src.predictor import rank_resumes


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    jd = ""
    results = []

    if request.method == 'POST':
        jd = request.form['jd']
        df = load_dataset()
        resumes = df['Resume'].tolist()

        # clean
        jd_clean = clean_text(jd)
        resumes_clean = [clean_text(r) for r in resumes]

        # embed + score
        jd_vec, res_vecs = get_bert_vectors(jd_clean, resumes_clean)
        ranked = rank_resumes(jd_vec, res_vecs, resumes)

        results = ranked[:5]
        labels = [f'Resume {i+1}' for i in range(len(results))]
        scores = [round(score * 100, 2) for _, score in results]

         # ✅ Log to MLflow
        print("🚀 Logging to MLflow...")
        log_resume_ranking(jd, results)


        return render_template('index.html', results=results, jd=jd, labels=labels, scores=scores)

    return render_template('index.html', results=None)



if __name__ == '__main__':
    app.run(debug=True, port=5001)
