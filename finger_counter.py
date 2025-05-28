import numpy as np
import cv2
import os
import time
import pickle
from collections import deque
import argparse


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []

        # Инициализация весов и смещений
        for i in range(len(self.layer_sizes) - 1):
            # Инициализация Xavier/Glorot
            scale = np.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))

        # Параметры обучения
        self.learning_rate = 0.001
        self.reg_lambda = 0.0001
        self.momentum = 0.9
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.v_biases = [np.zeros_like(b) for b in self.biases]

        # Для ранней остановки
        self.best_weights = None
        self.best_biases = None

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        # Устойчивый softmax
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        # Прямое распространение через все слои, кроме последнего
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.relu(z))

        # Последний слой с softmax
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(self.softmax(z))

        return self.activations[-1]

    def compute_loss(self, y):
        m = y.shape[0]
        probs = self.activations[-1]

        # Устойчивая кросс-энтропия
        clipped_probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        correct_log_probs = -np.log(clipped_probs[range(m), y])
        data_loss = np.sum(correct_log_probs) / m

        # Регуляризация L2
        reg_loss = 0.5 * self.reg_lambda * sum(np.sum(w * w) for w in self.weights)

        return data_loss + reg_loss

    def backward(self, X, y):
        m = y.shape[0]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # Ошибка на выходном слое
        delta = self.activations[-1].copy()
        delta[range(m), y] -= 1
        delta /= m

        # Градиенты для последнего слоя
        grads_w[-1] = np.dot(self.activations[-2].T, delta) + self.reg_lambda * self.weights[-1]
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Обратное распространение через скрытые слои
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l + 1].T) * self.relu_derivative(self.z_values[l])
            grads_w[l] = np.dot(self.activations[l].T, delta) + self.reg_lambda * self.weights[l]
            grads_b[l] = np.sum(delta, axis=0, keepdims=True)

        # Обновление весов с momentum
        for i in range(len(self.weights)):
            self.v_weights[i] = self.momentum * self.v_weights[i] + self.learning_rate * grads_w[i]
            self.v_biases[i] = self.momentum * self.v_biases[i] + self.learning_rate * grads_b[i]
            self.weights[i] -= self.v_weights[i]
            self.biases[i] -= self.v_biases[i]

    def train(self, X, y, epochs=1000, batch_size=64, validation_split=0.2):
        # Разделение на обучающую и валидационную выборки
        idx = np.random.permutation(X.shape[0])
        split = int(len(idx) * (1 - validation_split))
        train_idx, val_idx = idx[:split], idx[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        best_val_acc = 0.0
        no_improve = 0
        patience = 20
        min_delta = 0.001

        for epoch in range(epochs):
            # Уменьшение learning rate
            if epoch % 100 == 0 and epoch > 0:
                self.learning_rate *= 0.9
                print(f"Learning rate reduced to {self.learning_rate:.6f}")

            # Перемешивание данных
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx]
            y_train = y_train[idx]

            epoch_loss = 0
            num_batches = int(np.ceil(X_train.shape[0] / batch_size))

            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, X_train.shape[0])
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # Прямое и обратное распространение
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

                # Расчет потерь
                batch_loss = self.compute_loss(y_batch)
                if not np.isnan(batch_loss):
                    epoch_loss += batch_loss

            # Валидация
            val_output = self.forward(X_val)
            val_loss = self.compute_loss(y_val)
            val_acc = self.accuracy(X_val, y_val)

            # Средние потери за эпоху
            epoch_loss /= num_batches

            if epoch % 10 == 0:
                train_acc = self.accuracy(X_train, y_train)
                print(f"Epoch {epoch}: Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}, "
                      f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

            # Условия ранней остановки
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                no_improve = 0
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
                print(f"New best val accuracy: {best_val_acc:.2f}%")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}. Best val accuracy: {best_val_acc:.2f}%")
                    self.weights = [w.copy() for w in self.best_weights]
                    self.biases = [b.copy() for b in self.best_biases]
                    break

            # Дополнительное условие остановки при достижении целевой точности
            if val_acc >= 95.0:
                print(f"Target accuracy of 95% reached at epoch {epoch}")
                break

        # Финал обучения
        if self.best_weights is not None:
            self.weights = [w.copy() for w in self.best_weights]
            self.biases = [b.copy() for b in self.best_biases]

        print("Training completed")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        return self.forward(X)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100

    def save_model(self, filename):
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.layer_sizes = model_data['layer_sizes']


class ImageProcessor:
    @staticmethod
    def preprocess_image(image, size=(64, 64)):
        # Конвертация в YCrCb - лучше для детекции кожи
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Диапазон цвета кожи в YCrCb
        lower_skin = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)

        # Создание маски кожи
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Морфологические операции для улучшения маски
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

        # Поиск контуров на маске кожи
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, skin_mask

        # Фильтрация контуров по размеру и форме
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # слишком маленькие контуры пропускаем
                continue

            # Проверка соотношения сторон
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # не подходит для руки
                continue

            valid_contours.append(cnt)

        if not valid_contours:
            return None, None, skin_mask

        # Выбираем самый большой из валидных контуров
        hand_contour = max(valid_contours, key=cv2.contourArea)

        # Получение ограничивающего прямоугольника
        x, y, w, h = cv2.boundingRect(hand_contour)

        # Увеличиваем область для захвата всей руки
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        # Обрезка изображения до области руки
        hand_roi = skin_mask[y:y + h, x:x + w]

        # Изменение размера
        resized = cv2.resize(hand_roi, size)

        # Нормализация
        normalized = resized / 255.0

        # Проверка на NaN и бесконечности
        if np.isnan(normalized).any() or np.isinf(normalized).any():
            normalized = np.nan_to_num(normalized)

        # Выравнивание в вектор
        flattened = normalized.flatten()

        return flattened, (x, y, w, h), skin_mask

    @staticmethod
    def capture_training_data(output_dir="training_data", classes=6, samples_per_class=50):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть камеру")
            return

        print("Сбор данных для обучения:")
        print("0: Кулак (0 пальцев)")
        print("1-5: Количество пальцев")
        print("Нажмите 'c' для захвата изображения")
        print("Нажмите 'q' для завершения")

        for class_id in range(classes):
            class_dir = os.path.join(output_dir, str(class_id))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            print(f"\nГотовьтесь к захвату изображений для класса {class_id}")
            print("Нажмите любую клавишу, когда будете готовы...")
            cv2.waitKey(0)

            count = 0
            while count < samples_per_class:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Отображение инструкции
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Class: {class_id} - Captured: {count}/{samples_per_class}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Показать обработанную руку
                processed, bbox, skin_mask = ImageProcessor.preprocess_image(frame)
                if processed is not None:
                    hand_img = (processed.reshape(64, 64) * 255).astype(np.uint8)
                    hand_img_display = cv2.resize(hand_img, (128, 128))
                    cv2.imshow("Hand Preview", hand_img_display)

                cv2.imshow("Capture Training Data", display_frame)

                key = cv2.waitKey(1)
                if key == ord('c'):
                    if processed is not None:
                        # Сохранение данных
                        filename = os.path.join(class_dir, f"sample_{count}.npy")
                        np.save(filename, processed)
                        count += 1
                        print(f"Saved sample {count} for class {class_id}")
                    else:
                        print("Hand not detected! Try again.")
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        cap.release()
        cv2.destroyAllWindows()
        print("Завершение сбора данных")

    @staticmethod
    def load_training_data(data_dir="training_data"):
        X = []
        y = []

        for class_id in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_id)
            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                if filename.endswith(".npy"):
                    filepath = os.path.join(class_dir, filename)
                    data = np.load(filepath)
                    X.append(data)
                    y.append(int(class_id))

        X = np.array(X)
        y = np.array(y)

        # Нормализация данных
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
        X_normalized = (X - mean) / std

        return X_normalized, y

    @staticmethod
    def augment_data(X, y):
        augmented_X = []
        augmented_y = []

        for i in range(len(X)):
            img = X[i].reshape(64, 64)
            label = y[i]

            # Оригинальное изображение
            augmented_X.append(img.flatten())
            augmented_y.append(label)

            # Горизонтальное отражение
            flipped = cv2.flip(img, 1)
            augmented_X.append(flipped.flatten())
            augmented_y.append(label)

            # Повороты
            for angle in [-15, -10, -5, 5, 10, 15]:
                M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
                rotated = cv2.warpAffine(img, M, (64, 64))
                augmented_X.append(rotated.flatten())
                augmented_y.append(label)

            # Сдвиги
            for dx, dy in [(5, 0), (-5, 0), (0, 5), (0, -5)]:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(img, M, (64, 64))
                augmented_X.append(shifted.flatten())
                augmented_y.append(label)

            # Добавление шума
            noise = np.random.normal(0, 0.05, (64, 64))
            noisy_img = np.clip(img + noise, 0, 1)
            augmented_X.append(noisy_img.flatten())
            augmented_y.append(label)

        return np.array(augmented_X), np.array(augmented_y)


class FingerCounterApp:
    def __init__(self, model_path=None):
        input_size = 64 * 64
        hidden_sizes = [128, 64]  # Два скрытых слоя
        output_size = 6

        self.model = NeuralNetwork(input_size, hidden_sizes, output_size)

        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path)
            print(f"Модель загружена из {model_path}")

    def train_model(self, data_dir="training_data", save_path="finger_counter_model.pkl"):
        print("Загрузка данных для обучения...")
        X, y = ImageProcessor.load_training_data(data_dir)
        print(f"Загружено {X.shape[0]} образцов")

        # Проверка данных
        print(f"Min value: {np.min(X)}, Max value: {np.max(X)}")
        print(f"NaN values: {np.isnan(X).sum()}, Inf values: {np.isinf(X).sum()}")

        # Аугментация данных
        print("Аугментация данных...")
        X_aug, y_aug = ImageProcessor.augment_data(X, y)
        print(f"После аугментации: {X_aug.shape[0]} образцов")

        # Проверка распределения классов
        unique, counts = np.unique(y_aug, return_counts=True)
        print("Распределение классов:")
        for label, count in zip(unique, counts):
            print(f"Класс {label}: {count} образцов ({count / len(y_aug) * 100:.2f}%)")

        # Обучение
        print("Начало обучения...")
        start_time = time.time()
        self.model.train(X_aug, y_aug, epochs=1000, batch_size=64)
        training_time = time.time() - start_time
        print(f"Обучение завершено за {training_time:.2f} секунд")

        # Сохранение модели
        self.model.save_model(save_path)
        print(f"Модель сохранена в {save_path}")

        # Оценка точности
        accuracy = self.model.accuracy(X_aug, y_aug)
        print(f"Точность на обучающих данных: {accuracy:.2f}%")

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть камеру")
            return

        print("Запуск распознавания пальцев...")
        print("Нажмите 'q' для выхода")

        prediction_history = deque(maxlen=15)
        confidence_history = deque(maxlen=15)
        min_confidence = 0.6  # Минимальная уверенность для принятия решения

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Обработка изображения
            processed, bbox, skin_mask = ImageProcessor.preprocess_image(frame)

            # Отображение маски кожи
            skin_mask_display = cv2.resize(skin_mask, (320, 240)) if skin_mask is not None else np.zeros((240, 320),
                                                                                                         dtype=np.uint8)
            cv2.imshow("Skin Mask", skin_mask_display)

            if processed is not None:
                # Предсказание
                probs = self.model.predict_proba(processed.reshape(1, -1))
                prediction = np.argmax(probs)
                confidence = np.max(probs)

                # Фильтрация по уверенности
                if confidence > min_confidence:
                    prediction_history.append(prediction)
                    confidence_history.append(confidence)

                # Сглаживание предсказаний
                if prediction_history:
                    # Взвешенное среднее по уверенности
                    weights = np.array(confidence_history)
                    weights /= weights.sum()  # Нормализация весов
                    smoothed_pred = np.round(np.dot(prediction_history, weights))
                else:
                    smoothed_pred = prediction

                # Отрисовка результатов
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Fingers: {int(smoothed_pred)}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}",
                            (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Показать обработанное изображение руки
                hand_img = (processed.reshape(64, 64) * 255).astype(np.uint8)
                hand_img_display = cv2.resize(hand_img, (128, 128))
                cv2.imshow("Hand", hand_img_display)

            # Отображение FPS
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Finger Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finger Counter Application")
    parser.add_argument("--capture", action="store_true", help="Capture training data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--run", action="store_true", help="Run the finger counter")
    parser.add_argument("--model", type=str, default="finger_counter_model.pkl",
                        help="Path to model file")

    args = parser.parse_args()

    app = FingerCounterApp(args.model)

    if args.capture:
        ImageProcessor.capture_training_data()

    if args.train:
        app.train_model()

    if args.run:
        app.run()