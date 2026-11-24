import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


MODEL = "gemini-3.5-flash-lite"
PAUSE = 5                 
OUTPUT_FILE = "results.md"

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / "apikey.env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

report = []


def ask(prompt, system=None, retries=4):
    """Надсилає запит до моделі. При помилках 429/503 чекає і повторює."""
    config = types.GenerateContentConfig(system_instruction=system) if system else None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=prompt, config=config
            )
            time.sleep(PAUSE)
            return (response.text or "").strip()
        except errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Помилка {e.code}, чекаю {wait} с і повторюю...")
                time.sleep(wait)
            else:
                raise


def record(title, prompt, answer, system=None):
    """Виводить у консоль і додає до звіту."""
    print(f"\n=== {title} ===")
    print(f"PROMPT: {prompt}")
    print(f"ВІДПОВІДЬ: {answer}")
    block = [f"### {title}", ""]
    if system:
        block += [f"**System role:** {system}", ""]
    block += ["**Prompt:**", "", "```", prompt, "```", "",
              "**Відповідь моделі:**", "", answer, "", "**Аналіз:** _(напишіть свій висновок)_", "", "---", ""]
    report.append("\n".join(block))


# ---------- Завдання 1: п'ять типів запитів ----------
report.append(f"# Лабораторна робота №3\n\nВикористана модель: **{MODEL}**\n\n## Завдання 1. Типи запитів\n")

text_to_translate = (
    "Large language models are neural networks trained on huge amounts of text. "
    "They learn to predict the next word in a sequence, which allows them to "
    "answer questions, write code and translate between languages."
)

text_to_summarize = (
    "Машинне навчання — це розділ штучного інтелекту, який вивчає методи побудови "
    "алгоритмів, здатних навчатися на даних. Замість того щоб програмувати кожне "
    "правило вручну, розробник надає системі набір прикладів, а алгоритм сам "
    "знаходить закономірності. Розрізняють навчання з учителем, коли дані мають "
    "правильні відповіді, навчання без учителя, коли модель шукає структуру в "
    "нерозмічених даних, та навчання з підкріпленням, де агент отримує винагороду "
    "за дії. Машинне навчання застосовують у розпізнаванні зображень, рекомендаційних "
    "системах, обробці природної мови та медичній діагностиці."
)

tasks_1 = [
    ("1. Пояснення", "Поясни простими словами, що таке ооп. Не більше 5 речень."),
    ("2. Генерація тексту", "Напиши короткий текст (6-8 речень) на тему «найкращі мови програмування на 2026 рік»."),
    ("3. Переклад", f"Переклади англійський текст на українську мову:\n\n{text_to_translate}"),
    ("4. Узагальнення", f"Стисни наведений текст до 2 речень, зберігши головну думку:\n\n{text_to_summarize}"),
    ("5. Генерація списку", "Склади список із 7 популярних бібліотек Python для машинного навчання. Для кожної вкажи одним реченням, для чого вона потрібна."),
]
for title, prompt in tasks_1:
    record(title, prompt, ask(prompt))

# ---------- Завдання 2: zero-shot та few-shot ----------
report.append("## Завдання 2. Zero-shot і few-shot\n")

new_description = "Система групує клієнтів магазину за схожою поведінкою покупок, не знаючи заздалегідь жодних категорій."

zero_shot = (
    "Визнач тип задачі машинного навчання (класифікація, регресія чи кластеризація) "
    f"для такого опису:\n\n{new_description}"
)

few_shot = f"""Визнач тип задачі машинного навчання для опису. Відповідай строго у форматі:
Тип: <назва>
Причина: <одне коротке речення>

Опис: Модель прогнозує ціну квартири за її площею та районом.
Тип: Регресія
Причина: Результат є неперервним числовим значенням.

Опис: Система визначає, чи є лист спамом.
Тип: Класифікація
Причина: Результат належить до одного з наперед заданих класів.

Опис: Алгоритм розбиває новини на тематичні групи без заздалегідь заданих тем.
Тип: Кластеризація
Причина: Групи формуються за схожістю, без розмічених даних.

Опис: {new_description}
"""

record("Zero-shot", zero_shot, ask(zero_shot))
record("Few-shot", few_shot, ask(few_shot))

# ---------- Завдання 3: удосконалення prompt ----------
report.append("## Завдання 3. Удосконалення prompt\n")

basic_prompt = "Розкажи про штучний інтелект."
improved_prompt = (
    "Ти викладач інформатики, який читає вступну лекцію для студентів першого курсу.\n"
    "Завдання: розкажи, що таке штучний інтелект.\n"
    "Контекст: студенти знають основи програмування, але не знайомі з машинним навчанням.\n"
    "Формат відповіді: 1) визначення (2 речення); 2) три основні напрями (список із коротким поясненням); "
    "3) два приклади застосування в повсякденному житті.\n"
    "Обсяг: до 150 слів. Мова: українська, простий стиль без зайвого жаргону."
)

record("Початковий prompt", basic_prompt, ask(basic_prompt))
record("Вдосконалений prompt", improved_prompt, ask(improved_prompt))

# ---------- Завдання 4: системна роль ----------
report.append("## Завдання 4. Системна роль\n")

role_question = "Поясни, що таке перенавчання (overfitting) у машинному навчанні. Відповідь до 100 слів."
roles = {
    "Викладач": "Ти досвідчений викладач інформатики. Пояснюй зрозуміло, з аналогіями та прикладами для студентів.",
    "Програміст": "Ти senior-розробник Python. Відповідай практично, технічно точно, наводь короткий приклад коду або практичну пораду.",
    "Науковець": "Ти науковий співробітник у галузі машинного навчання. Відповідай формально, точною термінологією, згадуй метрики та методи регуляризації.",
}
for role_name, system_text in roles.items():
    record(f"Роль: {role_name}", role_question, ask(role_question, system=system_text), system=system_text)

# ---------- Збереження ----------
(BASE_DIR / OUTPUT_FILE).write_text("\n".join(report), encoding="utf-8")
print(f"\nГотово! Результати збережено у файлі {OUTPUT_FILE}")