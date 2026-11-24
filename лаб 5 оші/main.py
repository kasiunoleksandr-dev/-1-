import os
from core import SimpleRAG


API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6LxEXbz7-JgE8r8QqTDiue3wJJ1xiBq7TpX-PEAPkKRMA")


def main():
    rag = SimpleRAG(api_key=API_KEY)

    # Індексація документів
    print("=== ІНДЕКСАЦІЯ ДОКУМЕНТІВ ===")
    rag.index_documents(docs_folder="./docs")

    # Інтерактивний режим для тестування запитів
    print("\n=== ТЕСТУВАННЯ RAG-СИСТЕМИ ===")
    print("Введи 'exit' для виходу або 'compare' для порівняння з RAG / без RAG.\n")

    while True:
        user_input = input("Введи запитання: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        if user_input.lower() == "compare":
            q = input("Введи запитання для порівняння (Додаткове завдання*): ").strip()
            print("\n--- Відповідь БЕЗ RAG ---")
            print(rag.ask_without_rag(q))

            print("\n--- Відповідь З RAG ---")
            ans, chunks = rag.ask_with_rag(q)
            print(ans)
            print("Використані джерела:", [c["source"] for c in chunks])
            print("-" * 40 + "\n")
            continue
        answer, chunks = rag.ask_with_rag(user_input, top_k=2)

        print("\n[Знайдені фрагменти (Retrieval)]:")
        for c in chunks:
            print(f" -> {c['id']} ({c['source']}): {c['text'][:100]}...")

        print("\n[Відповідь LLM]:")
        print(answer)
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()