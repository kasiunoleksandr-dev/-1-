import os
import math

import numpy as np
from PIL import Image, ImageDraw
# 1. Метрики відстаней
def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def manhattan_dist(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))

def chebyshev_dist(a, b):
    return max(abs(x - y) for x, y in zip(a, b))

# 2. Побудова вектора ознак за сіткою 4х5
def extract_features(img, rows=4, cols=5):
    gray = img.convert("L")
    w, h = gray.size
    cell_w = w / cols
    cell_h = h / rows

    abs_vec = []
    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(c * cell_w), int(r * cell_h)
            x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
            cell = gray.crop((x1, y1, x2, y2))
            count = sum(1 for p in cell.getdata() if p < 128)
            abs_vec.append(count)

    total = sum(abs_vec)
    if total > 0:
        norm_vec = [round(c / total, 4) for c in abs_vec]
    else:
        norm_vec = [0.0] * len(abs_vec)

    return abs_vec, norm_vec

# 3. Класифікатор
class SimpleClassifier:
    def __init__(self):
        self.clusters = {}

    def train(self, training_data):
        self.clusters.clear()
        for label, vectors in training_data.items():
            arr = np.array(vectors, dtype=float)
            self.clusters[label] = {
                "center": np.mean(arr, axis=0).tolist(),
                "min": np.min(arr, axis=0).tolist(),
                "max": np.max(arr, axis=0).tolist()
            }

    def predict(self, test_vec, method="Евклідова"):
        if not self.clusters:
            return None, "Модель не навчена", {}

        if method == "Ортогональні області":
            best_label = None
            min_violations = float("inf")
            scores = {}
            for label, data in self.clusters.items():
                v = sum(1 for x, mn, mx in zip(test_vec, data["min"], data["max"]) if x < mn - 0.001 or x > mx + 0.001)
                scores[label] = v
                if v < min_violations:
                    min_violations = v
                    best_label = label
            return best_label, f"порушень: {min_violations}", scores

        dist_fn = {
            "Евклідова": euclidean_dist,
            "Манхеттенська": manhattan_dist,
            "Чебишевська": chebyshev_dist
        }.get(method, euclidean_dist)

        best_label = None
        min_dist = float("inf")
        distances = {}

        for label, data in self.clusters.items():
            d = dist_fn(test_vec, data["center"])
            distances[label] = round(d, 4)
            if d < min_dist:
                min_dist = d
                best_label = label

        return best_label, f"відстань = {min_dist:.4f}", distances


# 4. Зчитування навчальних файлів із РЕАЛЬНОЇ папки на диску
def load_training_from_folder(folder_path="train_images"):
    
    if not os.path.exists(folder_path):
        create_train_dataset(folder_path)

    training_data = {}
    stats = {}
    total_files = 0

    for item in sorted(os.listdir(folder_path)):
        full_sub = os.path.join(folder_path, item)
        if os.path.isdir(full_sub):
            label = item.replace("class_", "").strip()
            training_data[label] = []

            for fname in sorted(os.listdir(full_sub)):
                if fname.lower().endswith((".bmp", ".png", ".jpg")):
                    fpath = os.path.join(full_sub, fname)
                    img = Image.open(fpath)
                    _, norm_vec = extract_features(img, rows=4, cols=5)
                    training_data[label].append(norm_vec)
                    total_files += 1

            stats[label] = len(training_data[label])

    return training_data, total_files, stats


    
def create_train_dataset(folder_path="train_images"):
    os.makedirs(folder_path, exist_ok=True)
    classes = ["0", "1", "2", "3"]

    for d in classes:
        sub = os.path.join(folder_path, f"class_{d}")
        os.makedirs(sub, exist_ok=True)

        for i in range(1, 11):
            w, h = 240, 240
            img = Image.new("RGB", (w, h), "white")
            draw = ImageDraw.Draw(img)

            shift_x = ((i % 3) - 1) * 8
            shift_y = ((i // 3) - 1) * 8
            width = 14 + (i % 3) * 2

            pad_x = 40 + shift_x
            pad_y = 35 + shift_y
            right = w - 40 + shift_x
            bottom = h - 35 + shift_y

            if d == "0":
                draw.ellipse([pad_x, pad_y, right, bottom], outline="black", width=width)
            elif d == "1":
                cx = w // 2 + shift_x
                draw.line([(cx, pad_y), (cx, bottom)], fill="black", width=width)
                draw.line([(cx, pad_y), (cx - 35, pad_y + 40)], fill="black", width=width)
                draw.line([(cx - 40, bottom), (cx + 40, bottom)], fill="black", width=width)
            elif d == "2":
                mid_y = int(h * 0.45) + shift_y
                draw.arc([pad_x, pad_y, right, mid_y], 180, 0, fill="black", width=width)
                draw.line([(right, pad_y + int((mid_y - pad_y) * 0.5)), (pad_x + 10, bottom)], fill="black", width=width)
                draw.line([(pad_x, bottom), (right, bottom)], fill="black", width=width)
            elif d == "3":
                mid_y = int(h * 0.5) + shift_y
                draw.arc([pad_x, pad_y, right, mid_y + 10], 210, 45, fill="black", width=width)
                draw.arc([pad_x, mid_y - 10, right, bottom], 315, 150, fill="black", width=width)

            filename = os.path.join(sub, f"{d}_{i:02d}.bmp")
            img.save(filename, format="BMP")