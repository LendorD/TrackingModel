import argparse
import time
from collections import deque
import pickle
import cv2
import os
import numpy as np


class NeuralNetworkBackprop:
    def __init__(self, input_size, hidden_sizes, output_size):
        # Архитектура сети
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        # Параметры модели
        self.weights = []  # Матрицы весов для каждого слоя
        self.biases = []  # Векторы смещений для каждого слоя

        # Инициализация весов и смещений
        for i in range(len(self.layer_sizes) - 1):
            # Инициализация Xavier/Glorot
            scale = np.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))

        # Параметры оптимизации
        self.learning_rate = 0.001
        self.reg_lambda = 0.0001

        # Для ранней остановки
        self.best_weights = None
        self.best_biases = None

    # Основная функция активации для скрытых слоев
    def relu(self, x):
        return np.maximum(0, x)

    # Функция активации для выходного слоя
    def softmax(self, x):
        # Устойчивый softmax
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    # Прямое распространение (Forward Pass)
    def forward(self, X):
        # Временные хранилища для прямого прохода
        self.activations = [X]  # Активации каждого слоя
        self.z_values = []  # Входные значения перед активацией

        # Прямое распространение через все слои, кроме последнего
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.relu(z))

        # Последний слой с softmax
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(self.softmax(z))

        return self.activations[-1]  # Возвращает выходные вероятности

    # Вычисление потерь (Loss Function)
    def compute_loss(self, y):
        # Устойчивая кросс-энтропия
        m = y.shape[0]
        probs = self.activations[-1]

        # Устойчивая кросс-энтропия
        clipped_probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        correct_log_probs = -np.log(clipped_probs[range(m), y])
        data_loss = np.sum(correct_log_probs) / m

        # Регуляризация L2
        reg_loss = 0.5 * self.reg_lambda * sum(np.sum(w * w) for w in self.weights)

        return data_loss + reg_loss

    # Производная ReLU
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    # Обратное распространение ошибки
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

        # Обновление весов
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    # Обучение сети
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

            # Обработка мини-батчей
            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, X_train.shape[0])
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # Прямое распространение
                self.forward(X_batch)

                # Обратное распространение и обновление весов
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
                if no_improve >= patience and epoch >= 150:
                    print(f"Early stopping at epoch {epoch}. Best val accuracy: {best_val_acc:.2f}%")
                    self.weights = [w.copy() for w in self.best_weights]
                    self.biases = [b.copy() for b in self.best_biases]
                    break

            # Дополнительное условие остановки
            if val_acc >= 95.0 and epoch >= 150:
                print(f"Target accuracy of 95% reached at epoch {epoch}")
                break

        # Финал обучения
        if self.best_weights is not None:
            self.weights = [w.copy() for w in self.best_weights]
            self.biases = [b.copy() for b in self.best_biases]

        print("Training completed")

    # Предсказание классов
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)  # Индекс максимальной вероятности

    # Предсказание вероятностей
    def predict_proba(self, X):
        return self.forward(X)

    # Вычисление точности
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100

    # Сохранение модели
    def save_model(self, filename):
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

    # Загрузка модели
    def load_model(self, filename):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.layer_sizes = model_data['layer_sizes']

class RockPaperScissorsGame:
    def __init__(self, model_path=None):
        input_size = 64 * 64
        hidden_sizes = [128, 64]
        output_size = 3  # Камень, Бумага, Ножницы

        self.model = NeuralNetworkBackprop(input_size, hidden_sizes, output_size)

        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path)
            print(f"Модель загружена из {model_path}")

        self.gesture_names = ["Rock", "Paper", "Scissors"]
        self.history = deque(maxlen=5)  # История последних игр

    def train_model(self, data_dir="rps_data", save_path="rps_model_back.pkl"):
        print("Загрузка данных для обучения...")
        X, y = ImageProcessor.load_training_data(data_dir)
        print(f"Загружено {X.shape[0]} образцов")

        print("Аугментация данных...")
        X_aug, y_aug = ImageProcessor.augment_data(X, y)
        print(f"После аугментации: {X_aug.shape[0]} образцов")

        print("Начало обучения...")
        start_time = time.time()
        self.model.train(X_aug, y_aug, epochs=1000, batch_size=64)
        training_time = time.time() - start_time
        print(f"Обучение завершено за {training_time:.2f} секунд")

        self.model.save_model(save_path)
        print(f"Модель сохранена в {save_path}")

        accuracy = self.model.accuracy(X_aug, y_aug)
        print(f"Точность на обучающих данных: {accuracy:.2f}%")

    def determine_winner(self, gesture1, gesture2):
        """Определяет победителя по правилам игры"""
        if gesture1 == gesture2:
            return 0  # Ничья

        # Правила: Камень > Ножницы, Ножницы > Бумага, Бумага > Камень
        if (gesture1 == 0 and gesture2 == 2) or \
                (gesture1 == 2 and gesture2 == 1) or \
                (gesture1 == 1 and gesture2 == 0):
            return 1  # Игрок 1 побеждает

        return 2  # Игрок 2 побеждает

    def run(self, source=0):
        cap = cv2.VideoCapture(source)

        # Получаем FPS видео, если это файл
        if isinstance(source, str):
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Значение по умолчанию
            frame_delay = int(1000 / fps)  # Задержка в мс
        else:
            frame_delay = 1  # Для камеры

        player1_score = 0
        player2_score = 0
        draw_count = 0
        game_start_time = None
        countdown_duration = 3
        result_duration = 3
        game_state = "ready"
        player_choices = [None, None]
        winner = None

        print("Запуск игры Камень-Ножницы-Бумага...")
        print("Нажмите 'q' для выхода")

        while True:
            start_time = time.time() * 1000  # Текущее время в миллисекундах

            ret, frame = cap.read()
            if not ret:
                # Если видео закончилось, перезапускаем
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            current_time = time.time()

            # Обработка изображения для двух рук
            hands, bboxes, _ = ImageProcessor.preprocess_image(frame, max_hands=2)

            # Фильтруем руки с правильным размером
            valid_hands = []
            valid_bboxes = []
            for hand, bbox in zip(hands, bboxes):
                if hand.size == 64 * 64:  # Проверяем размер вектора
                    valid_hands.append(hand)
                    valid_bboxes.append(bbox)
            hands = valid_hands
            bboxes = valid_bboxes

            # Сортируем руки по горизонтальной позиции (слева направо)
            if len(bboxes) >= 2:
                sorted_indices = sorted(range(len(bboxes)), key=lambda i: bboxes[i][0])
                bboxes = [bboxes[i] for i in sorted_indices]
                hands = [hands[i] for i in sorted_indices]

            # Отображение интерфейса
            if game_state == "ready":
                cv2.putText(frame, "Prepare hands", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to start", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Отображение рук
                for i in range(min(2, len(bboxes))):
                    x, y, w, h = bboxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Player {i + 1}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                key = cv2.waitKey(1)
                if key == 32:  # SPACE
                    game_state = "countdown"
                    game_start_time = current_time

            elif game_state == "countdown":
                # Обратный отсчет
                time_left = countdown_duration - (current_time - game_start_time)
                if time_left > 0:
                    count = int(np.ceil(time_left))
                    cv2.putText(frame, str(count),
                                (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                else:
                    game_state = "result"
                    game_start_time = current_time

                    # Распознавание жестов
                    player_choices = [None, None]
                    if len(hands) >= 2:
                        for i in range(2):
                            # Проверка размера перед передачей в сеть
                            if hands[i].size == 64 * 64:
                                input_data = hands[i].reshape(1, -1)
                                if input_data.shape[1] == 4096:  # 64*64=4096
                                    probs = self.model.predict_proba(input_data)
                                    player_choices[i] = np.argmax(probs)

                    # Определение победителя
                    if None not in player_choices:
                        winner = self.determine_winner(player_choices[0], player_choices[1])
                        if winner == 1:
                            player1_score += 1
                        elif winner == 2:
                            player2_score += 1
                        else:
                            draw_count += 1

                        # Сохранение истории
                        self.history.append({
                            "player1": self.gesture_names[player_choices[0]],
                            "player2": self.gesture_names[player_choices[1]],
                            "result": winner
                        })

            elif game_state == "result":
                # Отображение результата
                time_left = result_duration - (current_time - game_start_time)
                if time_left > 0:
                    # Отображение жестов
                    for i in range(min(2, len(bboxes))):
                        x, y, w, h = bboxes[i]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        if i < len(player_choices) and player_choices[i] is not None:
                            cv2.putText(frame, f"PLAYER {i + 1}: {self.gesture_names[player_choices[i]]}",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Отображение результата
                    if winner == 0:
                        result_text = "DRAW!"
                        color = (255, 255, 0)
                    elif winner == 1:
                        result_text = "PLAYER 1 WINS!"
                        color = (0, 255, 255)
                    else:
                        result_text = "PLAYER 2 WINS!"
                        color = (0, 165, 255)

                    cv2.putText(frame, result_text,
                                (frame.shape[1] // 2 - 150, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                else:
                    game_state = "ready"

            # Отображение счета
            cv2.putText(frame, f"Score: Player1 {player1_score} - Player2 {player2_score} Draws: {draw_count}",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Отображение истории
            y_pos = frame.shape[0] - 60
            for i, game in enumerate(reversed(self.history)):
                if i >= 3:  # Показываем только последние 3 игры
                    break
                text = f"{game['player1']} vs {game['player2']}: "
                if game['result'] == 0:
                    text += "Draw"
                elif game['result'] == 1:
                    text += "P1"
                else:
                    text += "P2"
                cv2.putText(frame, text, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos -= 20

            # Отображение FPS для видеофайлов
            if isinstance(source, str):
                current_fps = 1000 / max(1, (time.time() * 1000 - start_time))
                cv2.putText(frame, f"FPS: {current_fps:.1f}/{fps:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow("Rock Paper Scissors", frame)

            # Рассчитываем время обработки и корректируем задержку
            processing_time = time.time() * 1000 - start_time
            adjusted_delay = max(1, frame_delay - int(processing_time))

            key = cv2.waitKey(adjusted_delay) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class ImageProcessor:
    @staticmethod
    def preprocess_image(image, size=(64, 64), max_hands=2):
        # Конвертация в YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Диапазон цвета кожи
        lower_skin = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)

        # Создание маски кожи
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Морфологические операции
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

        # Поиск контуров
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Фильтрация контуров
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # слишком маленькие
                continue

            # Проверка соотношения сторон
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue

            valid_contours.append(cnt)

        # Сортируем контуры по размеру (от большего к меньшему)
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

        hands = []
        bboxes = []

        # Обрабатываем не более max_hands контуров
        for i in range(min(max_hands, len(valid_contours))):
            cnt = valid_contours[i]
            x, y, w, h = cv2.boundingRect(cnt)

            # Увеличиваем область
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)

            # Обрезка изображения
            hand_roi = skin_mask[y:y + h, x:x + w]

            # Изменение размера
            resized = cv2.resize(hand_roi, size)

            # Нормализация
            normalized = resized / 255.0

            # Проверка на NaN
            if np.isnan(normalized).any() or np.isinf(normalized).any():
                normalized = np.nan_to_num(normalized)

            # Выравнивание в вектор
            flattened = normalized.flatten()

            hands.append(flattened)
            bboxes.append((x, y, w, h))

        return hands, bboxes, skin_mask

    @staticmethod
    def capture_training_data(output_dir="training_data", classes=6, samples_per_class=100):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rock Paper Scissors Game")
    parser.add_argument("--capture", action="store_true", help="Capture training data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--play", action="store_true", help="Play the game")
    parser.add_argument("--model", type=str, default="rps_model_back.pkl", help="Path to model file")

    args = parser.parse_args()

    if args.capture:
        ImageProcessor.capture_training_data(output_dir="rps_data", classes=3, samples_per_class=200)

    if args.train:
        game = RockPaperScissorsGame()
        game.train_model(data_dir="rps_data", save_path=args.model)

    if args.play:
        game = RockPaperScissorsGame(args.model)
        game.run()