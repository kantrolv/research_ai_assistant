from core.pipeline import run_pipeline

if __name__ == "__main__":
    query = input("Enter your query: ")

    result = run_pipeline(query)

    print("\n🧠 Answer:\n", result["answer"])
    print("\n🧾 Summary:\n", result["summary"])
    print("\n📚 Sources:")
    for s in result["sources"]:
        print("-", s)
