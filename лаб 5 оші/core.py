import os
import chromadb
from google import genai
from google.genai import types


class SimpleRAG:

    def __init__(self, api_key: str, db_path: str = "./chroma_db"):
        # Ініціалізація клієнта Gemini
        self.client = genai.Client(api_key=api_key)
        self.embed_model = "gemini-embedding-001"
        self.llm_model = "gemini-3.5-flash-lite"

        # Ініціалізація локального векторного сховища ChromaDB (Завдання 4)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="lab5_docs"
        )

    def _chunk_text(
        self, text: str, chunk_size: int = 400, overlap: int = 50
    ) -> list[str]:
        """Розбиває текст на фрагменти з невеликим перекриттям (Завдання 2)."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def get_embedding(self, text: str, task_type: str) -> list[float]:
        """Створює embedding-вектор для тексту через Gemini API (Завдання 3)."""
        response = self.client.models.embed_content(
            model=self.embed_model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return response.embeddings[0].values

    def index_documents(self, docs_folder: str = "./docs"):
        """Завантажує документи, ріже на чанки, векторизує та зберігає в базу."""
        # Очищаємо старі дані, якщо перезапускаємо індексацію
        if self.collection.count() > 0:
            self.chroma_client.delete_collection("lab5_docs")
            self.collection = self.chroma_client.create_collection(
                name="lab5_docs"
            )

        ids, documents, embeddings, metadatas = [], [], [], []
        chunk_counter = 1

        for filename in os.listdir(docs_folder):
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(docs_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = self._chunk_text(text)
            print(f"Файл '{filename}' розбито на {len(chunks)} фрагментів.")

            for chunk in chunks:
                chunk_id = f"chunk_{chunk_counter:02d}"
                # Для документів використовуємо task_type="RETRIEVAL_DOCUMENT"
                vector = self.get_embedding(
                    chunk, task_type="RETRIEVAL_DOCUMENT"
                )

                ids.append(chunk_id)
                documents.append(chunk)
                embeddings.append(vector)
                metadatas.append({"source": filename})
                chunk_counter += 1

        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            print(f"\nУспішно збережено {len(ids)} фрагментів у векторне сховище!")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Шукає найбільш релевантні фрагменти до запиту (Завдання 5)."""
        # Для пошукового запиту використовуємо task_type="RETRIEVAL_QUERY"
        query_vector = self.get_embedding(query, task_type="RETRIEVAL_QUERY")

        results = self.collection.query(
            query_embeddings=[query_vector], n_results=top_k
        )

        retrieved_items = []
        for i in range(len(results["ids"][0])):
            retrieved_items.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i],
            })
        return retrieved_items

    def ask_with_rag(self, question: str, top_k: int = 3) -> tuple[str, list[dict]]:
        """Повний цикл RAG: пошук контексту + генерація відповіді LLM (Завдання 6)."""
        retrieved_chunks = self.retrieve(question, top_k=top_k)

        context_text = "\n\n---\n\n".join(
            [f"[Джерело: {c['source']}]\n{c['text']}" for c in retrieved_chunks]
        )

        prompt = f"""Використовуючи ТІЛЬКИ наведений нижче контекст, дай точну відповідь на запитання користувача.
Якщо відповіді на запитання немає в контексті, прямо повідом про це (наприклад: "У наданих документах немає інформації для відповіді на це запитання") і не вигадуй факти.

Контекст:
{context_text}

Запитання:
{question}"""

        response = self.client.models.generate_content(
            model=self.llm_model, contents=prompt
        )
        return response.text, retrieved_chunks

    def ask_without_rag(self, question: str) -> str:
        """Прямий запит до LLM без контексту (для Додаткового завдання*)."""
        response = self.client.models.generate_content(
            model=self.llm_model, contents=question
        )
        return response.text