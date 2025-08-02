from muVERA import MuVERAIndex

def main():
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Graph neural networks are powerful tools for relational data.",
        "Transformers are replacing recurrent neural networks in NLP.",
        "Cheetahs are the fastest land animals.",
        "Birds migrate across continents during winter."
    ]

    query = "fast animals"

    index = MuVERAIndex(dim=384)
    for doc in corpus:
        index.add_document(doc)

    results = index.rerank(query, top_k=3)
    print(f"\nQuery: '{query}'")
    for doc_text, score in results:
        print(f" {doc_text}\n→ Score: {score:.4f}\n")

if __name__ == "__main__":
    main()
