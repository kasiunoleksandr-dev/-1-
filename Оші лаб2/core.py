import os
import numpy as np
from PIL import Image

# ==========================================
# 1. ОБРОБКА ЗОБРАЖЕНЬ ТА ВЕКТОРІВ ОЗНАК
# ==========================================

def crop_and_resize(img, target_size=(16, 16)):
    """
    Обрізання зображення по контуру цифри (Bounding Box) та масштабує.
    Це усуває домінування білого тла.
    """
    img_gray = img.convert('L')
    arr = np.array(img_gray)
    
    non_white = np.where(arr < 200)
    if len(non_white[0]) > 0 and len(non_white[1]) > 0:
        ymin, ymax = np.min(non_white[0]), np.max(non_white[0])
        xmin, xmax = np.min(non_white[1]), np.max(non_white[1])
        ymin, ymax = max(0, ymin - 1), min(arr.shape[0], ymax + 1)
        xmin, xmax = max(0, xmin - 1), min(arr.shape[1], xmax + 1)
        img_cropped = img_gray.crop((xmin, ymin, xmax, ymax))
    else:
        img_cropped = img_gray
        
    return img_cropped.resize(target_size)

def load_image_absolute(image_path, target_size=(16, 16), use_crop=True):
    """Повертає абсолютний вектор ознак (значення пікселів 0-255)."""
    img = Image.open(image_path)
    if use_crop:
        img_resized = crop_and_resize(img, target_size)
    else:
        img_resized = img.convert('L').resize(target_size)
    return np.array(img_resized).flatten()

def normalize_vector(abs_vector):
    """Нормування вектора ознак (масштабування до [0, 1])."""
    return abs_vector / 255.0

def binarize_vector(abs_vector, threshold=128):
    """Бінаризація вектора у значення {-1, 1}."""
    return np.where(abs_vector < threshold, 1, -1)

def load_dataset_from_folder(training_path, target_size=(16, 16)):
    """Зчитує навчальні зразки з папки training ."""
    X, y = [], []
    class_1_dir = os.path.join(training_path, 'class_1')
    class_2_dir = os.path.join(training_path, 'class_2')
    
    if os.path.exists(class_1_dir):
        for fname in sorted(os.listdir(class_1_dir)):
            if fname.lower().endswith(('.bmp', '.png', '.jpg')):
                fpath = os.path.join(class_1_dir, fname)
                abs_v = load_image_absolute(fpath, target_size, use_crop=False)
                X.append(binarize_vector(abs_v))
                y.append(1)
                
    if os.path.exists(class_2_dir):
        for fname in sorted(os.listdir(class_2_dir)):
            if fname.lower().endswith(('.bmp', '.png', '.jpg')):
                fpath = os.path.join(class_2_dir, fname)
                abs_v = load_image_absolute(fpath, target_size, use_crop=False)
                X.append(binarize_vector(abs_v))
                y.append(-1)
                
    return np.array(X), np.array(y)

# ==========================================
# 2. ПЕРЦЕПТРОН РОЗЕНБЛАТА
# ==========================================

class RosenblattPerceptron:
    def __init__(self, input_dim, learning_rate=0.1, init_weights=None):
        self.learning_rate = learning_rate
        self.bias = np.random.uniform(0.0, 1.0) if init_weights is None else init_weights.get('b', 0.5)
        if init_weights and 'w' in init_weights:
            self.weights = np.array(init_weights['w'])
        else:
            self.weights = np.random.uniform(0.0, 1.0, size=input_dim)

    def predict(self, x):
        net = np.dot(self.weights, x) - self.bias
        return 1 if net > 0 else -1

    def correct_weights(self, x, target, prediction):
        error = target - prediction
        if error != 0:
            self.weights += self.learning_rate * error * x
            self.bias -= self.learning_rate * error
            return True
        return False

    def train_epoch(self, X_train, y_train):
        errors_count = 0
        for x, target in zip(X_train, y_train):
            pred = self.predict(x)
            if self.correct_weights(x, target, pred):
                errors_count += 1
        return errors_count

# ==========================================
# 3. МЕРЕЖА ХОПФІЛДА
# ==========================================

class HopfieldNetwork:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.zeros((input_dim, input_dim))

    def train(self, patterns):
        self.weights = np.zeros((self.input_dim, self.input_dim))
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)

    def recognize(self, x, max_iter=300):
        """Асинхронне оновлення нейронів для уникнення хибних осциляцій."""
        y = np.copy(x)
        for _ in range(max_iter):
            y_old = np.copy(y)
            indices = np.arange(self.input_dim)
            np.random.shuffle(indices)
            for idx in indices:
                net = np.dot(self.weights[idx], y)
                if net > 0:
                    y[idx] = 1
                elif net < 0:
                    y[idx] = -1
            if np.array_equal(y, y_old):
                break
        return y

# ==========================================
# 4. МЕРЕЖА ХЕМІНГА
# ==========================================

class HammingNetwork:
    def __init__(self, num_classes, input_dim):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.weights = np.zeros((input_dim, num_classes))
        self.biases = np.zeros(num_classes)
        self.epsilon = 1.0 / (num_classes * 2.0)

    def train(self, patterns):
        for k, p in enumerate(patterns):
            self.weights[:, k] = p / 2.0
            self.biases[k] = self.input_dim / 2.0

    def recognize(self, x, max_iter=300):
        y1 = np.dot(x, self.weights) + self.biases
        y2 = np.copy(y1)
        for _ in range(max_iter):
            y2_new = np.zeros_like(y2)
            for j in range(self.num_classes):
                sum_others = np.sum(y2) - y2[j]
                s_j = y2[j] - self.epsilon * sum_others
                y2_new[j] = s_j if s_j > 0 else 0
            if np.sum(y2_new > 0) <= 1 or np.allclose(y2, y2_new):
                y2 = y2_new
                break
            y2 = y2_new
        return np.argmax(y2)