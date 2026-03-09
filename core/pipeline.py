from agents.planner import plan_query
from agents.query_expander import expand_query
from agents.retriever import retrieve_papers
from agents.reranker import rerank
from agents.reasoner import generate_answer
from agents.critic import critique_answer
from agents.summarizer import summarize

def run_pipeline(user_query):

    # 🧠 Plan
    plan = plan_query(user_query)

    # 🔍 Expand
    expanded_queries = expand_query(plan["original_query"])

    # 📚 Retrieve
    papers = retrieve_papers(expanded_queries)

    # 🎯 Rerank
    ranked_papers = rerank(papers)

    # 🧠 Generate Answer (RAG)
    answer = generate_answer(user_query, ranked_papers)

    # 🛡️ Improve Answer
    improved_answer = critique_answer(answer)

    # 🧾 Summary
    summary = summarize(improved_answer)

    return {
        "answer": improved_answer,
        "summary": summary,
        "sources": [p["title"] for p in ranked_papers]
    }
