from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data.papers import paper_database

model = SentenceTransformer('all-MiniLM-L6-v2')
db_embeddings = model.encode([p["abstract"] for p in paper_database])

def retrieve_papers(queries, top_k=5):
    all_scores = []

    for q in queries:
        q_emb = model.encode([q])
        sims = cosine_similarity(q_emb, db_embeddings)[0]
        all_scores.append(sims)

    avg_scores = np.mean(all_scores, axis=0)
    top_indices = np.argsort(avg_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "title": paper_database[idx]["title"],
            "abstract": paper_database[idx]["abstract"],
            "score": float(avg_scores[idx])
        })

    return results
