import os
import math
import datetime
import requests
from google import genai
from google.genai import types


def calculator(expression: str) -> str:
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return f"Результат обчислення: {result}"
    except Exception as e:
        return f"Помилка обчислення '{expression}': {e}"


def search_wikipedia(query: str) -> str:
    try:
        url = "https://uk.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "redirects": 1,
            "titles": query
        }
        headers = {"User-Agent": "StudentAssistantAgent/1.0"}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        pages = response.json().get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return f"За запитом '{query}' у Вікіпедії нічого не знайдено."
            extract = page_data.get("extract", "")
            return f"Довідка ({page_data.get('title')}): {extract[:500]}..."
        return "Не вдалося обробити відповідь пошуку."
    except Exception as e:
        return f"Помилка пошуку: {e}"


def date_helper(target_date: str = "") -> str:
    today = datetime.date.today()
    if not target_date:
        return f"Сьогоднішня дата: {today.strftime('%Y-%m-%d')}."
    try:
        target = datetime.datetime.strptime(target_date.strip(), "%Y-%m-%d").date()
        delta = (target - today).days
        if delta > 0:
            return f"Сьогодні {today}. До дати {target} залишилося {delta} дн."
        elif delta == 0:
            return f"Сьогодні {today}. Ця подія відбувається сьогодні!"
        else:
            return f"Сьогодні {today}. Дата {target} вже минула ({abs(delta)} дн. тому)."
    except ValueError:
        return "Невірний формат дати. Використовуй 'YYYY-MM-DD'."



class StudentAssistantAgent:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Не знайдено GEMINI_API_KEY.")
        
        self.client = genai.Client(api_key=key)
        
        # ЯВНА ВНУТРІШНЯ ПАМ'ЯТЬ АГЕНТА (зберігає історію діалогу)
        self.memory: list[dict[str, str]] = []
        
        self.system_instruction = (
            "Ти — інтелектуальний АІ-агент 'Помічник студента'.\n"
            "Доступні інструменти:\n"
            "1. `calculator` — для точних математичних обчислень;\n"
            "2. `search_wikipedia` — для пошуку визначень та довідок у Вікіпедії;\n"
            "3. `date_helper` — для перевірки поточної дати та підрахунку днів до дедлайну.\n\n"
            "ПРАВИЛА:\n"
            "- Перед виконанням складного завдання або викликом інструмента ОБОВ'ЯЗКОВО виведи:\n"
            "📋 План дій:\n1. ...\n2. ...\n"
            "- Для розрахунків, пошуку визначень чи дат завжди використовуй відповідний інструмент.\n"
            "- Обов'язково враховуй попередні повідомлення з історії діалогу."
        )

    def _build_conversation_context(self, new_message: str) -> list[types.Content]:
        contents = []
        for entry in self.memory:
            contents.append(
                types.Content(
                    role=entry["role"],
                    parts=[types.Part.from_text(text=entry["text"])]
                )
            )
        # Додаємо поточний запит користувача
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=new_message)]
            )
        )
        return contents

    def ask(self, user_message: str) -> str:
        """Обробляє запит, оновлює внутрішню пам'ять і повертає відповідь."""
        # 1. Формуємо контекст із пам'яті + нове повідомлення
        contents = self._build_conversation_context(user_message)

        # 2. Викликаємо модель із підключеними інструментами
        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                tools=[calculator, search_wikipedia, date_helper],
                temperature=0.2,
            )
        )
        
        reply_text = response.text

        #  Зберігаємо запит користувача та відповідь агента у внутрішню пам'ять
        self.memory.append({"role": "user", "text": user_message})
        self.memory.append({"role": "model", "text": reply_text})

        return reply_text

    def show_memory(self) -> str:
        if not self.memory:
            return "Пам'ять порожня."
        lines = []
        for i, msg in enumerate(self.memory, 1):
            role_label = "Студент" if msg["role"] == "user" else "Агент"
            lines.append(f"{i}. [{role_label}]: {msg['text'][:80]}...")
        return "\n".join(lines)