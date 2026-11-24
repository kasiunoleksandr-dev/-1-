import os
from core import StudentAssistantAgent

def main():
    print("=== АІ-агент: Помічник студента ===")
    print("Команди: 'exit' — вихід, 'memory' — переглянути внутрішню пам'ять агента.\n")

    
    os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6LxEXbz7-JgE8r8QqTDiue3wJJ1xiBq7TpX-PEAPkKRMA"

    try:
        agent = StudentAssistantAgent()
    except ValueError as e:
        print(f"Помилка ініціалізації: {e}")
        return

    while True:
        user_input = input("Студент: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Роботу завершено.")
            break
        if user_input.lower() == "memory":
            print(f"\n--- Внутрішня пам'ять (self.memory) ---\n{agent.show_memory()}\n" + "-" * 40 + "\n")
            continue

        print("\nАгент думає...")
        try:
            reply = agent.ask(user_input)
            print(f"\nАгент:\n{reply}\n" + "-" * 50 + "\n")
        except Exception as e:
            print(f"Помилка під час виконання запиту: {e}\n")

if __name__ == "__main__":
    main()