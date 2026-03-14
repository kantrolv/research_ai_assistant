def rerank(papers):
    return sorted(papers, key=lambda x: x["score"], reverse=True)
