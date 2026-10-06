import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import core


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторна робота №1: Розпізнавання образів")
        self.geometry("960x490")
        self.resizable(False, False)

        # Модель та шлях до папки з навчальними зображеннями
        self.clf = core.SimpleClassifier()
        self.train_folder = "train_images"

        # Первинне зчитування з папки при старті
        self.train_from_folder(show_msg=False)

        self.img_size = 240

        # Полотно малювання
        self.drawn_img = Image.new("RGB", (self.img_size, self.img_size), "white")
        self.draw = ImageDraw.Draw(self.drawn_img)
        self.last_x = None
        self.last_y = None

        # Завантажене зображення
        self.loaded_img = None
        self.loaded_tk = None

        self._build_ui()

    def _build_ui(self):
        # 1. ВЕРХНЯ ПАНЕЛЬ З УСІМА КНОПКАМИ ТА ФУНКЦІЯМИ
        top_bar = tk.Frame(self, relief=tk.RAISED, bd=1, padx=8, pady=8)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top_bar, text="Завантажити BMP", command=self.load_bmp).pack(side=tk.LEFT, padx=4)
        tk.Button(top_bar, text="Навчання (з папки)", command=self.choose_train_folder_and_train).pack(side=tk.LEFT, padx=4)
        tk.Button(top_bar, text="Розпізнати зображення", command=self.recognize_loaded).pack(side=tk.LEFT, padx=4)
        tk.Button(top_bar, text="Розпізнати малюнок", command=self.recognize_drawn).pack(side=tk.LEFT, padx=4)
        tk.Button(top_bar, text="Очистити", command=self.clear_canvas).pack(side=tk.LEFT, padx=4)

        tk.Label(top_bar, text="|  Метод / Норма:").pack(side=tk.LEFT, padx=(10, 4))
        self.method_var = tk.StringVar(value="Евклідова")
        method_cb = ttk.Combobox(
            top_bar,
            textvariable=self.method_var,
            values=["Евклідова", "Манхеттенська", "Чебишевська", "Ортогональні області"],
            state="readonly",
            width=20
        )
        method_cb.pack(side=tk.LEFT, padx=4)

        # 2. ОСНОВНА ЧАСТИНА (ТРИ КОЛОНКИ)
        main_frame = tk.Frame(self, padx=12, pady=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Полотно для малювання
        col1 = tk.Frame(main_frame)
        col1.pack(side=tk.LEFT, padx=8)

        tk.Label(col1, text="Малюнок (мишкою):", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.draw_canvas = tk.Canvas(col1, width=self.img_size, height=self.img_size, bg="white", relief=tk.SOLID, bd=1)
        self.draw_canvas.pack()
        self.draw_canvas.bind("<Button-1>", self.on_press)
        self.draw_canvas.bind("<B1-Motion>", self.on_drag)
        self.draw_canvas.bind("<ButtonRelease-1>", self.on_release)

        # Завантажений BMP файл
        col2 = tk.Frame(main_frame)
        col2.pack(side=tk.LEFT, padx=8)

        tk.Label(col2, text="Завантажене зображення:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.load_canvas = tk.Canvas(col2, width=self.img_size, height=self.img_size, bg="white", relief=tk.SOLID, bd=1)
        self.load_canvas.pack()
        self.load_label = tk.Label(col2, text="Файл не вибрано", fg="gray", font=("Arial", 8))
        self.load_label.pack(pady=2)

        # Результати та вектор ознак
        col3 = tk.Frame(main_frame)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)

        tk.Label(col3, text="Результат розпізнавання:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.result_lbl = tk.Label(
            col3,
            text="Очікування дій...",
            font=("Arial", 11, "bold"),
            relief=tk.GROOVE,
            bd=2,
            padx=10,
            pady=8,
            anchor=tk.W
        )
        self.result_lbl.pack(fill=tk.X, pady=4)

        # Статус навчання
        self.status_train_lbl = tk.Label(col3, text=f"Навчання: {self.train_summary}", fg="green", font=("Arial", 8), anchor=tk.W)
        self.status_train_lbl.pack(fill=tk.X, pady=(2, 4))

        tk.Label(col3, text="Оцінки по класах (0, 1, 2, 3, 4):", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(4, 2))
        self.scores_lbl = tk.Label(col3, text="Клас 0: -\nКлас 1: -\nКлас 2: -\nКлас 3: -\nКлас 4:-", font=("Arial", 9), justify=tk.LEFT, anchor=tk.W)
        self.scores_lbl.pack(fill=tk.X)

        tk.Label(col3, text="Вектор ознак (4х5 = 20 комірок):", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(6, 2))
        self.vec_txt = tk.Text(col3, height=4, font=("Consolas", 8), wrap=tk.WORD)
        self.vec_txt.pack(fill=tk.X)

    # Події малювання
    def on_press(self, event):
        self.last_x = event.x
        self.last_y = event.y
        r = 7
        self.draw_canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="black", outline="black")
        self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill="black", outline="black")

    def on_drag(self, event):
        if self.last_x and self.last_y:
            r = 7
            self.draw_canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=14, capstyle=tk.ROUND, fill="black")
            self.draw_canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="black", outline="black")
            self.draw.line([(self.last_x, self.last_y), (event.x, event.y)], width=14, fill="black", joint="curve")
            self.draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill="black", outline="black")
        self.last_x = event.x
        self.last_y = event.y

    def on_release(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.draw_canvas.delete("all")
        self.drawn_img = Image.new("RGB", (self.img_size, self.img_size), "white")
        self.draw = ImageDraw.Draw(self.drawn_img)
        self.result_lbl.config(text="Полотно очищено")
        self.scores_lbl.config(text="Клас 0: -\nКлас 1: -\nКлас 2: -\nКлас 3: -\nКлас 4:")
        self.vec_txt.delete("1.0", tk.END)

    # Навчання з папки
    def train_from_folder(self, folder=None, show_msg=True):
        if folder:
            self.train_folder = folder

        data, total, stats = core.load_training_from_folder(self.train_folder)
        self.clf.train(data)

        stat_str = ", ".join(f"Клас {k}: {v}" for k, v in sorted(stats.items()))
        self.train_summary = f"Навчено на {total} файлах ({stat_str})"

        if hasattr(self, 'status_train_lbl'):
            self.status_train_lbl.config(text=f"Навчання: {self.train_summary}")

        if show_msg:
            msg = f"Успішно зчитано {total} файлів із папки '{self.train_folder}':\n"
            for k, v in sorted(stats.items()):
                msg += f"- Клас {k}: {v} файлів\n"
            messagebox.showinfo("Навчання завершено", msg)

    def choose_train_folder_and_train(self):
        folder = filedialog.askdirectory(initialdir=self.train_folder, title="Оберіть папку для навчання (з підпапками класів)")
        if folder:
            self.train_from_folder(folder=folder, show_msg=True)

    # --- Завантаження та розпізнавання ---
    def load_bmp(self):
        path = filedialog.askopenfilename(filetypes=[("Зображення BMP / PNG", "*.bmp;*.png;*.jpg"), ("Всі файли", "*.*")])
        if not path:
            return

        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        self.loaded_img = img
        self.loaded_tk = ImageTk.PhotoImage(img)

        self.load_canvas.delete("all")
        self.load_canvas.create_image(self.img_size // 2, self.img_size // 2, image=self.loaded_tk)
        self.load_label.config(text=f"Завантажено: {os.path.basename(path)}")

    def run_recognition(self, image_to_test, source_name):
        abs_v, norm_v = core.extract_features(image_to_test, rows=4, cols=5)
        if sum(abs_v) == 0:
            messagebox.showwarning("Увага", f"Порожній {source_name}! Спочатку створіть або завантажте зображення.")
            return

        method = self.method_var.get()
        winner, score_text, all_scores = self.clf.predict(norm_v, method=method)

        self.result_lbl.config(text=f"Розпізнано: [Клас {winner}] ({score_text})")

        lines = []
        for k in sorted(all_scores.keys()):
            prefix = "-> " if k == winner else "   "
            lines.append(f"{prefix}Клас {k}: {all_scores[k]}")
        self.scores_lbl.config(text="\n".join(lines))

        vec_str = ", ".join(f"{x:.4f}" for x in norm_v)
        self.vec_txt.delete("1.0", tk.END)
        self.vec_txt.insert(tk.END, f"X_norm = [{vec_str}]")

    def recognize_drawn(self):
        self.run_recognition(self.drawn_img, "малюнок")

    def recognize_loaded(self):
        if self.loaded_img is None:
            messagebox.showwarning("Увага", "Спочатку натисніть «Завантажити BMP»!")
            return
        self.run_recognition(self.loaded_img, "файл")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()