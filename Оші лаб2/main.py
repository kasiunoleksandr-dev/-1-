import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

from core import (
    load_image_absolute, normalize_vector, binarize_vector, load_dataset_from_folder,
    RosenblattPerceptron, HopfieldNetwork, HammingNetwork
)

class Lab2TeacherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторна робота №2")
        self.geometry("950x720")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.perc_model = None
        self.X_train_perc = None
        self.y_train_perc = None
        self.current_perc_x = None
        self.current_perc_prediction = None
        
        self.hopfield = None
        self.hamming = None
        self.patterns_h = []
        self.current_h_x = None

        self._build_perceptron_tab()
        self._build_hopfield_hamming_tab()

    def _build_perceptron_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Завдання 1: Перцептрон Розенблата")

        folder_frame = ttk.LabelFrame(tab, text="Навчання за папкою тренування (training/)")
        folder_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Button(folder_frame, text="1. Завантажити папку training", command=self.load_training_folder).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Label(folder_frame, text="Крок навч. (r):").pack(side=tk.LEFT, padx=5)
        self.lr_entry = ttk.Entry(folder_frame, width=5)
        self.lr_entry.insert(0, "0.1")
        self.lr_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(folder_frame, text="2. Ініціалізувати перцептрон", command=self.init_perceptron).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(folder_frame, text="3. Запустити авто-навчання", command=self.train_all_epochs).pack(side=tk.LEFT, padx=5, pady=5)

        test_frame = ttk.LabelFrame(tab, text="Тестування поодинокого образу та режим вчителя")
        test_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Button(test_frame, text="Завантажити тестовий образ", command=self.load_test_image_perc).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(test_frame, text="Класифікувати", command=self.classify_perceptron).pack(side=tk.LEFT, padx=5, pady=5)

        content_frame = ttk.Frame(tab)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.img_lbl_perc = ttk.Label(content_frame, text="Зображення", borderwidth=2, relief="groove")
        self.img_lbl_perc.grid(row=0, column=0, padx=10, pady=5, sticky="n")

        vec_frame = ttk.LabelFrame(content_frame, text="Вектори ознак")
        vec_frame.grid(row=0, column=1, sticky="nsew", padx=10)
        self.txt_vectors_perc = tk.Text(vec_frame, width=45, height=8)
        self.txt_vectors_perc.pack(fill=tk.BOTH, expand=True)

        teacher_frame = ttk.LabelFrame(content_frame, text="Режим вчителя (ручне підтвердження / корекція)")
        teacher_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        self.lbl_perc_res = ttk.Label(teacher_frame, text="Результат класифікації: -", font=("Arial", 11, "bold"))
        self.lbl_perc_res.pack(pady=3)

        btn_frame = ttk.Frame(teacher_frame)
        btn_frame.pack(pady=3)
        ttk.Button(btn_frame, text="Правильно (Не міняти ваги)", command=lambda: self.teacher_feedback(True)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Помилка (Має бути Клас 1)", command=lambda: self.teacher_feedback(False, 1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Помилка (Має бути Клас 2)", command=lambda: self.teacher_feedback(False, -1)).pack(side=tk.LEFT, padx=5)

        log_frame = ttk.LabelFrame(content_frame, text="Протокол навчання та стан ваг")
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        content_frame.rowconfigure(2, weight=1)
        content_frame.columnconfigure(1, weight=1)

        self.txt_log_perc = tk.Text(log_frame, height=8)
        self.txt_log_perc.pack(fill=tk.BOTH, expand=True)

    def load_training_folder(self):
        folder = filedialog.askdirectory(title="Оберіть папку training")
        if not folder: return
        
        self.X_train_perc, self.y_train_perc = load_dataset_from_folder(folder)
        count = len(self.X_train_perc)
        if count == 0:
            messagebox.showwarning("Увага", "У папці не знайдено підпапок class_1 / class_2 з BMP зображеннями.")
            return
            
        c1 = np.sum(self.y_train_perc == 1)
        c2 = np.sum(self.y_train_perc == -1)
        self.txt_log_perc.insert(tk.END, f"--- Завантажено датасет з: {folder} ---\n")
        self.txt_log_perc.insert(tk.END, f"Усього зразків: {count} (Клас 1: {c1}, Клас 2: {c2}).\n")
        self.init_perceptron()

    def init_perceptron(self):
        dim = len(self.X_train_perc[0]) if self.X_train_perc is not None and len(self.X_train_perc) > 0 else 256
        lr = float(self.lr_entry.get())
        self.perc_model = RosenblattPerceptron(input_dim=dim, learning_rate=lr)
        self.txt_log_perc.insert(tk.END, f"Перцептрон ініціалізовано малими випадковими вагами (r={lr}).\n")
        self.txt_log_perc.see(tk.END)

    def train_all_epochs(self):
        if self.perc_model is None or self.X_train_perc is None or len(self.X_train_perc) == 0:
            messagebox.showerror("Помилка", "Спочатку завантажте папку тренування!")
            return

        self.txt_log_perc.insert(tk.END, "\n--- Початок автоматичного навчання по папці ---\n")
        total_samples = len(self.X_train_perc)
        max_epochs = 100
        
        for epoch in range(1, max_epochs + 1):
            errors = self.perc_model.train_epoch(self.X_train_perc, self.y_train_perc)
            acc = ((total_samples - errors) / total_samples) * 100
            self.txt_log_perc.insert(tk.END, f"Епоха {epoch:02d}: Помилок = {errors}/{total_samples} | Точність: {acc:.1f}%\n")
            if errors == 0:
                self.txt_log_perc.insert(tk.END, f" Навчання успішно завершено на епосі {epoch}! Досягнуто 100% збіжності.\n")
                break
        
        self.txt_log_perc.see(tk.END)

    def load_test_image_perc(self):
        filepath = filedialog.askopenfilename(filetypes=[("BMP/PNG/JPG", "*.bmp;*.png;*.jpg")])
        if not filepath: return
        
        img = Image.open(filepath)
        self.tk_img_perc = ImageTk.PhotoImage(img.resize((100, 100)))
        self.img_lbl_perc.config(image=self.tk_img_perc)

        abs_vec = load_image_absolute(filepath, use_crop=False)
        norm_vec = normalize_vector(abs_vec)
        self.current_perc_x = binarize_vector(abs_vec)

        self.txt_vectors_perc.delete(1.0, tk.END)
        self.txt_vectors_perc.insert(tk.END, f"Абсолютний вектор (перші 15):\n{abs_vec[:15]}...\n\n")
        self.txt_vectors_perc.insert(tk.END, f"Нормований вектор (перші 15):\n{np.round(norm_vec[:15], 2)}...\n")

    def classify_perceptron(self):
        if not self.perc_model or self.current_perc_x is None:
            messagebox.showwarning("Увага", "Завантажте тестовий образ та ініціалізуйте перцептрон.")
            return
        self.current_perc_prediction = self.perc_model.predict(self.current_perc_x)
        res_text = "Клас 1 (+1)" if self.current_perc_prediction == 1 else "Клас 2 (-1)"
        self.lbl_perc_res.config(text=f"Результат класифікації: {res_text}")
        self.txt_log_perc.insert(tk.END, f"Класифіковано тестовий образ: Y = {self.current_perc_prediction} ({res_text})\n")
        self.txt_log_perc.see(tk.END)

    def teacher_feedback(self, is_correct, target=None):
        if not self.perc_model or self.current_perc_prediction is None: return
        
        if is_correct:
            self.txt_log_perc.insert(tk.END, " Вчитель: Правильно. Ваги не змінюються.\n")
        else:
            self.txt_log_perc.insert(tk.END, f" Вчитель: Помилка! Корекція ваг перцептрона до цілі {target}...\n")
            self.perc_model.correct_weights(self.current_perc_x, target, self.current_perc_prediction)
            self.txt_log_perc.insert(tk.END, f"Нові ваги (фрагмент): {np.round(self.perc_model.weights[:8], 3)}... Зсув b = {self.perc_model.bias:.3f}\n")
        self.txt_log_perc.see(tk.END)

    def _build_hopfield_hamming_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Завдання 2: Хопфілд та Хеммінг (3 класи)")

        top_frame = ttk.Frame(tab)
        top_frame.pack(fill=tk.X, pady=5)

        ttk.Button(top_frame, text="Завантажити Еталони (3 шт)", command=self.load_etalons).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Обчислити матриці ШНМ", command=self.init_hop_ham).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Завантажити невідомий образ", command=self.load_unknown_h).pack(side=tk.LEFT, padx=5)

        self.lbl_unknown = ttk.Label(top_frame, text="Невідомий образ", relief="solid")
        self.lbl_unknown.pack(side=tk.RIGHT, padx=20)

        mid_frame = ttk.Frame(tab)
        mid_frame.pack(fill=tk.BOTH, expand=True)

        self.txt_matrix = tk.Text(mid_frame, height=15)
        self.txt_matrix.pack(fill=tk.BOTH, padx=10, pady=5)

        bot_frame = ttk.Frame(tab)
        bot_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(bot_frame, text="Класифікувати Хопфілдом", command=self.recognize_hopfield).pack(side=tk.LEFT, padx=10)
        ttk.Button(bot_frame, text="Класифікувати Хеммінгом", command=self.recognize_hamming).pack(side=tk.LEFT, padx=10)

        self.lbl_hh_res = ttk.Label(bot_frame, text="Результат: -", font=("Arial", 11, "bold"))
        self.lbl_hh_res.pack(side=tk.LEFT, padx=20)

    def load_etalons(self):
        filepaths = filedialog.askopenfilenames(title="Оберіть рівно 3 еталонних BMP", filetypes=[("BMP/PNG/JPG", "*.bmp;*.png;*.jpg")])
        if len(filepaths) != 3:
            messagebox.showwarning("Увага", "Потрібно обрати рівно 3 файли для трьох класів!")
            return
        
        self.patterns_h = []
        for fp in filepaths:
            abs_vec = load_image_absolute(fp, use_crop=True)
            bin_vec = binarize_vector(abs_vec)
            self.patterns_h.append(bin_vec)
        
        self.txt_matrix.delete(1.0, tk.END)
        self.txt_matrix.insert(tk.END, "Завантажено 3 еталони. Проведено автокропінг та бінаризацію у {-1, 1}.\n")

    def init_hop_ham(self):
        if len(self.patterns_h) != 3: return
        
        dim = len(self.patterns_h[0])
        self.hopfield = HopfieldNetwork(dim)
        self.hopfield.train(self.patterns_h)

        self.hamming = HammingNetwork(num_classes=3, input_dim=dim)
        self.hamming.train(self.patterns_h)

        self.txt_matrix.delete(1.0, tk.END)
        self.txt_matrix.insert(tk.END, f"Матриця Хопфілда W (фрагмент 5x5):\n{self.hopfield.weights[:5, :5]}\n\n")
        self.txt_matrix.insert(tk.END, f"Ваги 1-го шару Хеммінга W (фрагмент 5x3):\n{self.hamming.weights[:5, :]}\n")

    def load_unknown_h(self):
        filepath = filedialog.askopenfilename(filetypes=[("BMP/PNG/JPG", "*.bmp;*.png;*.jpg")])
        if not filepath: return
        
        img = Image.open(filepath)
        self.tk_img_unk = ImageTk.PhotoImage(img.resize((80, 80)))
        self.lbl_unknown.config(image=self.tk_img_unk)

        abs_vec = load_image_absolute(filepath, use_crop=True)
        self.current_h_x = binarize_vector(abs_vec)
        self.txt_matrix.insert(tk.END, "\n--- Невідомий образ завантажено (проведено автокропінг) ---\n")

    def recognize_hopfield(self):
        if not self.hopfield or self.current_h_x is None: return
        res_vec = self.hopfield.recognize(self.current_h_x)
        
        matches = [np.array_equal(res_vec, p) for p in self.patterns_h]
        if any(matches):
            class_idx = matches.index(True) + 1
            self.lbl_hh_res.config(text=f"Хопфілд: Збіг з Класом {class_idx}")
        else:
            self.lbl_hh_res.config(text="Хопфілд: Хибний стан (перехресна асоціація)")

    def recognize_hamming(self):
        if not self.hamming or self.current_h_x is None: return
        winner_idx = self.hamming.recognize(self.current_h_x)
        self.lbl_hh_res.config(text=f"Хеммінг: Віднесено до Класу {winner_idx + 1}")

if __name__ == "__main__":
    app = Lab2TeacherApp()
    app.mainloop()