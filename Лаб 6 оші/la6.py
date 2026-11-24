import os
import time
from google import genai
from core import SimpleRAG

# Встав свій API-ключ Gemini
API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6LxEXbz7-JgE8r8QqTDiue3wJJ1xiBq7TpX-PEAPkKRMA")

# 3 категорії відповідно до твоїх doc1.txt, doc2.txt, doc3.txt
CATEGORIES = ["комп'ютерні мережі", "нейронні мережі", "бази даних"]

# Тестові дані (Завдання 1 та 5): (текст, правильна_категорія)
TEST_DATASET = [
    ("Маршрутизатор перенаправляє пакети між підмережами на основі таблиці маршрутизації та IP-адрес.", "комп'ютерні мережі"),
    ("Протокол TCP гарантує надійну доставку пакетів із підтвердженням отримання.", "комп'ютерні мережі"),
    ("Згорткові шари ефективно виділяють ознаки зображень завдяки застосуванню фільтрів.", "нейронні мережі"),
    ("Зворотне поширення помилки використовується для оновлення ваг у багатошаровому перцептроні.", "нейронні мережі"),
    ("Для прискорення пошуку записів у таблиці за ключовим полем створюють B-tree індекс.", "бази даних"),
    ("Транзакція повинна відповідати принципам атомарності, узгодженості, ізольованості та довговічності.", "бази даних"),
]


class Lab6:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3.5-flash-lite"
        self.rag = SimpleRAG(api_key=api_key, db_path="./chroma_lab6")

    def prepare_rag(self):
        """Індексує doc1.txt, doc2.txt, doc3.txt з папки docs (Завдання 3)."""
        print("=== Індексація бази даних для RAG ===")
        self.rag.index_documents(docs_folder="./docs")
        print("=" * 40 + "\n")

    def _clean_answer(self, raw_text: str) -> str:
        """Очищає відповідь моделі від зайвих символів."""
        text = raw_text.strip().lower().replace(".", "").replace('"', "")
        for cat in CATEGORIES:
            if cat in text:
                return cat
        return text

    def approach_1_prompt_engineering(self, text: str) -> str:
        """Підхід 1: Zero-shot Prompt Engineering (Завдання 2)."""
        prompt = f"""Визнач категорію наведеного навчального тексту.
Можливі категорії: комп'ютерні мережі, нейронні мережі, бази даних.
Поверни ТІЛЬКИ одну назву категорії без пояснень і розділових знаків.

Текст: {text}"""

        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return self._clean_answer(response.text)

    def approach_2_rag(self, text: str) -> str:
        """Підхід 2: Retrieval-Augmented Generation (Завдання 3)."""
        chunks = self.rag.retrieve(text, top_k=2)
        context = "\n---\n".join([f"[Файл: {c['source']}]\n{c['text']}" for c in chunks])

        prompt = f"""Визнач категорію тексту, спираючись на знайдений контекст із бази знань.
Можливі категорії: комп'ютерні мережі, нейронні мережі, бази даних.
Поверни ТІЛЬКИ одну назву категорії без пояснень.

Контекст із бази знань:
{context}

Текст для класифікації: {text}"""

        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return self._clean_answer(response.text)

    def approach_3_finetuning_imitation(self, text: str) -> str:
        """Підхід 3: Імітація Fine-tuning через Few-shot prompting (Завдання 4)."""
        prompt = f"""Класифікуй навчальний текст за категорією, дотримуючись наведених прикладів.

Текст: Налаштування статичної маршрутизації, DNS та VLAN на комутаторах локальної мережі.
Категорія: комп'ютерні мережі

Текст: Модель OSI описує сім рівнів взаємодії мережевих протоколів під час передачі даних.
Категорія: комп'ютерні мережі

Текст: Функція активації ReLU прибирає від'ємні значення сигналів між прихованими шарами моделі.
Категорія: нейронні мережі

Текст: Градієнтний спуск мінімізує функцію втрат під час навчання глибокої моделі.
Категорія: нейронні мережі

Текст: Зовнішній ключ забезпечує цілісність зв'язків між двома реляційними таблицями у SQL.
Категорія: бази даних

Текст: Нормалізація таблиць дозволяє уникнути надлишковості та аномалій оновлення даних.
Категорія: бази даних

Текст: {text}
Категорія:"""

        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return self._clean_answer(response.text)

    def run(self):
        self.prepare_rag()

        results = []
        correct_counts = {"prompt": 0, "rag": 0, "few_shot": 0}

        print("Виконується тестування на 6 прикладах...\n")
        for idx, (text, expected) in enumerate(TEST_DATASET, start=1):
            p_ans = self.approach_1_prompt_engineering(text)
            r_ans = self.approach_2_rag(text)
            f_ans = self.approach_3_finetuning_imitation(text)

            if p_ans == expected:
                correct_counts["prompt"] += 1
            if r_ans == expected:
                correct_counts["rag"] += 1
            if f_ans == expected:
                correct_counts["few_shot"] += 1

            results.append((idx, text, expected, p_ans, r_ans, f_ans))
            print(f"[{idx}/{len(TEST_DATASET)}] Очікувалось: '{expected}' -> Prompt: '{p_ans}' | RAG: '{r_ans}' | Few-shot: '{f_ans}'")
            time.sleep(12)

        total = len(TEST_DATASET)
        print("\n" + "=" * 95)
        print("ТАБЛИЦЯ РЕЗУЛЬТАТІВ ДЛЯ ЗВІТУ (ЗАВДАННЯ 5)")
        print("=" * 95)
        print(f"{'№':<3} | {'Вхідний текст (скорочено)':<32} | {'Правильна':<19} | {'Prompt':<19} | {'RAG':<19} | {'Fine-tuning (ім.)':<19}")
        print("-" * 95)
        for idx, text, exp, p, r, f in results:
            short_t = (text[:29] + "...") if len(text) > 32 else text
            print(f"{idx:<3} | {short_t:<32} | {exp:<19} | {p:<19} | {r:<19} | {f:<19}")

        print("-" * 95)
        print(f"Правильних відповідей (з {total}): Prompt = {correct_counts['prompt']}, RAG = {correct_counts['rag']}, Імітація Fine-tuning = {correct_counts['few_shot']}")
        print(f"Accuracy (%): Prompt = {correct_counts['prompt']/total*100:.1f}%, RAG = {correct_counts['rag']/total*100:.1f}%, Імітація Fine-tuning = {correct_counts['few_shot']/total*100:.1f}%")


if __name__ == "__main__":
    experiment = Lab6(api_key=API_KEY)
    experiment.run()