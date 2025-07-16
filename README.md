# 🧠 Resume Screener - AI-Based Candidate Filtering

**"Let AI read resumes, match job requirements, and shortlist the right candidates—instantly."**

A complete **End-to-End Machine Learning pipeline** that parses resumes, matches them with job descriptions using NLP, ranks them by relevance, and offers insights via a clean Flask dashboard. Built with **BERT**, **MLflow**, and **Flask**, and containerized with **Docker**—this project simulates a real-world HR assistant powered by AI.

---

## 🚀 Live Demo

🌐 Web App: `http://localhost:5000`  
🧾 Upload resumes + job description → get matching score, top resumes, and downloadable output.

---

## 🧠 Tech Stack

| Layer              | Tools Used                                                           |
|--------------------|----------------------------------------------------------------------|
| **ML/NLP Model**    | BERT (sentence-transformers), Scikit-learn                          |
| **Resume Parsing** | PyMuPDF / pdfminer, docx, pandas                                     |
| **Similarity Engine** | Cosine Similarity, TF-IDF, Sentence Embeddings                  |
| **Tracking**        | MLflow (for metrics, parameters, artifacts)                         |
| **Deployment**      | Flask UI + HTML + Bootstrap + Docker                                |
| **MLOps**           | MLflow, DVC (optional), GitHub                                       |
| **Visualization**   | Matplotlib, Seaborn, Plotly (optional for charts)                  |

---

## 📁 Folder Structure

resume-screener/
├── app/ # Flask app (UI, backend, API)
├── data/ # Input resumes, job descriptions, processed data
│ ├── resumes/ # Raw PDF/DOCX resumes
│ ├── job_descs/ # Job descriptions for matching
│ └── processed/ # Cleaned CSVs, match results
├── models/ # Trained or saved NLP models
├── templates/ # HTML templates for Flask UI
├── static/ # CSS, JS, and Bootstrap assets
├── Dockerfile # For Docker container
├── requirements.txt # All Python dependencies
├── README.md # This file
└── utils/ # Helper scripts: parsing, scoring, preprocessing



---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/resume-screener.git
cd resume-screener
2. Install dependencies
pip install -r requirements.txt
3. Run the Flask app locally
python app/app.py
🖥 Visit: http://127.0.0.1:5000

🐳 Run with Docker
docker build -t resume-screener .
docker run -p 5000:5000 resume-screener
🔁 Pipeline Overview
Stage	Script/File	Description
Data Ingestion	utils/parse_resumes.py	Extracts text from PDF/DOCX resumes
Preprocessing	utils/clean_text.py	Cleans and tokenizes resume/job text
Embedding	utils/embedder.py	Converts text into BERT embeddings
Similarity Scoring	utils/match_score.py	Cosine similarity between resumes and job desc.
MLflow Logging	mlflow_logger.py	Tracks runs, scores, hyperparams, and artifacts
Frontend	app/app.py, templates/	Flask app to upload resumes and view results

📊 Sample Output
✅ Top 5 Matching Candidates

📈 Matching Score Visualization

📄 Download CSV of Results

📦 Logs in MLflow UI

💡 Why BERT for Matching?
Think of BERT as an HR manager who reads between the lines—it understands the meaning behind resume sentences, not just keywords.

Deep semantic understanding

Handles context like "Python (not snake) developer"

Pre-trained on millions of documents

💥 What You'll Learn from This Project
End-to-End NLP pipeline development

Real-world resume parsing and vectorization

Semantic similarity with BERT and cosine distance

Flask UI design for ML apps

MLflow tracking + Docker deployment

🤝 Contributing
💡 Got improvements? PRs welcome!

Fork the repo

Create a new branch: git checkout -b feature-x

Commit your changes: git commit -am 'Add feature x'

Push the branch: git push origin feature-x

Open a pull request 🚀

📬 Contact
GitHub: @subhra12-commits
