import json
import time
from google import genai
from google.genai import types

client = genai.Client(api_key="AQ.Ab8RN6LxEXbz7-JgE8r8QqTDiue3wJJ1xiBq7TpX-PEAPkKRMA")
MODEL = "gemini-3.5-flash-lite"


def task_1():
    print("\n--- ЗАВДАННЯ 1: Простий vs Просунутий prompt ---")
    sample_text = (
        "Нейронні мережі глибокого навчання (Deep Learning) використовують багатошарові "
        "перцептрони для автоматичного вилучення ознак із необроблених даних. Це потребує "
        "великих обсягів розмічених даних та потужних GPU, проте забезпечує високу точність "
        "у розпізнаванні зображень та обробці природної мови."
    )

    prompt_basic = f"Проаналізуй наведений текст та визнач його основні ідеї:\n{sample_text}"
    prompt_advanced = f"""
    Ти — викладач університету з дисципліни «Штучний інтелект».
    Проаналізуй наведений текст для студентів 2-го курсу.
    Виконай завдання поетапно:
    1. Визнач основну тему (1 речення).
    2. Виділи 3 ключові ідеї списком.
    3. Випиши 3 основні терміни.
    4. Сформулюй висновок (до 25 слів).
    Обмеження: використовуй лише факти з тексту.
    Текст: {sample_text}
    """

    for name, p in [("ЗВИЧАЙНИЙ", prompt_basic), ("ПРОСУНУТИЙ", prompt_advanced)]:
        res = client.models.generate_content(model=MODEL, contents=p)
        print(f"\n[{name} PROMPT]:\n{res.text}")


def task_2():
    print("\n--- ЗАВДАННЯ 2: Декомпозиція задачі ---")
    text = (
        "Протокол HTTP є протоколом прикладного рівня моделі OSI, що працює за принципом "
        "клієнт-сервер. Він не зберігає стан (stateless). Для збереження сесій використовуються "
        "Cookies або токени. Версія HTTP/3 працює поверх протоколу QUIC (на базі UDP)."
    )

    single = client.models.generate_content(
        model=MODEL,
        contents=f"Створи 2 тестові питання з 4 варіантами відповіді та вкажи правильний за текстом:\n{text}"
    ).text
    print("\n[ОДИН ЗАГАЛЬНИЙ PROMPT]:\n", single)

    # Ланцюжок кроків
    facts = client.models.generate_content(
        model=MODEL, contents=f"Випиши 3 ключові факти з тексту:\n{text}"
    ).text
    questions = client.models.generate_content(
        model=MODEL, contents=f"Сформулюй 2 складні запитання на основі цих фактів:\n{facts}"
    ).text
    final_test = client.models.generate_content(
        model=MODEL, contents=f"Додай до запитань по 4 варіанти відповіді (А-Г) та вкажи правильний:\n{questions}"
    ).text
    print("\n[РЕЗУЛЬТАТ ДЕКОМПОЗИЦІЇ]:\n", final_test)


def task_3():
    print("\n--- ЗАВДАННЯ 3: Імітація ReAct ---")
    article = (
        "У 2023 році кількість пристроїв IoT у світі перевищила 15 мільярдів. "
        "Більшість датчиків передають дані через протокол MQTT. Очевидно, що за 5 років "
        "реляційні бази даних повністю зникнуть з ринку IoT, бо вони непридатні для потоків."
    )
    prompt = f"""
    Відділи об'єктивні факти від суб'єктивних висновків автора.
    Суворо дотримуйся структури:
    1. План дій
    2. Виконання (аналіз кожного речення)
    3. Перевірка
    4. Фінальний результат (категорії «Факти» та «Висновки автора»)
    Текст: {article}
    """
    res = client.models.generate_content(model=MODEL, contents=prompt)
    print(res.text)


def task_4():
    print("\n--- ЗАВДАННЯ 4: Структурований результат (JSON) ---")
    text = "Алгоритм Quicksort використовує стратегію «розділяй і володарюй». У середньому працює за O(n log n), у найгіршому випадку — O(n^2)."
    prompt = f"""
    Поверни аналіз тексту у форматі JSON зі структурою:
    {{"topic": "...", "key_ideas": ["..."], "keywords": ["..."], "conclusion": "..."}}
    Текст: {text}
    """
    res = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    print("Сирий JSON:\n", res.text)
    data = json.loads(res.text)
    print("\nРозпарсено у Python -> Тема:", data["topic"])


def task_5():
    print("\n--- ЗАВДАННЯ 5: Робота з великим контекстом (Chunking) ---")
    doc = (
        "Розділ 1. Архітектура фон Неймана зберігає команди і дані в спільній пам'яті. " * 12 +
        "\nРозділ 2. Гарвардська архітектура фізично розділяє пам'ять інструкцій і даних. " * 12 +
        "\nРозділ 3. Сучасні CPU комбінують підходи: кеш L1 розділений, а ОЗП спільна. " * 12
    )

    no_chunk = client.models.generate_content(
        model=MODEL, contents=f"Зроби детальне узагальнення документа:\n{doc}"
    ).text

    # Розбиття на фрагменти (chunking)
    chunk_size = 700
    chunks = [doc[i:i + chunk_size] for i in range(0, len(doc), chunk_size)]
    summaries = []
    for idx, ch in enumerate(chunks, 1):
        s = client.models.generate_content(
            model=MODEL, contents=f"Коротко узагальни фрагмент {idx}:\n{ch}"
        ).text
        summaries.append(f"Фрагмент {idx}: {s}")

    final_chunked = client.models.generate_content(
        model=MODEL, contents="Об'єднай ці узагальнення в єдиний підсумок:\n" + "\n".join(summaries)
    ).text

    print(f"\n[БЕЗ CHUNKING]:\n{no_chunk}")
    print(f"\n[ПІСЛЯ CHUNKING ({len(chunks)} фрагментів)]:\n{final_chunked}")


def task_6():
    print("\n--- ЗАВДАННЯ 6: Параметри temperature та top-p ---")
    prompt = "Придумай слоган та короткий опис (2 речення) для стартапу розумних навушників-перекладачів."

    for temp in [0.0, 0.5, 1.0]:
        res = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temp, top_p=0.95)
        )
        print(f"\n[Temperature = {temp}]:\n{res.text.strip()}")
        time.sleep(1)

    for p_val in [0.5, 0.8, 1.0]:
        res = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.8, top_p=p_val)
        )
        print(f"\n[Top-P = {p_val}]:\n{res.text.strip()}")
        time.sleep(1)


if __name__ == "__main__":
    tasks = {
        "1": task_1,
        "2": task_2,
        "3": task_3,
        "4": task_4,
        "5": task_5,
        "6": task_6,
    }
    while True:
        choice = input("\nВведіть номер завдання (1-6) або '0' для виходу: ").strip()
        if choice == "0":
            break
        elif choice in tasks:
            tasks[choice]()
        else:
            print("Невірний вибір. Введіть число від 1 до 6.")