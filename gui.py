import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont, QImage, QPixmap
from PyQt5.QtCore import QTimer
import numpy as np
import cv2
import os
import pickle
import time
from collections import deque

from BackError import ImageProcessor, RockPaperScissorsGame


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Камень ножницы бумага")
        self.setGeometry(100, 100, 600, 450)  # Увеличили размер окна

        # Центральный виджет и основной макет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setAlignment(Qt.AlignCenter)

        # Заголовок
        title_label = QLabel("Камень ножницы бумага")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Группа выбора источника
        source_group = QGroupBox("Выберите источник данных")
        source_layout = QVBoxLayout(source_group)

        self.source_combo = QComboBox()
        self.source_combo.addItem("Камера (режим реального времени)")
        self.source_combo.addItem("Видеофайл")
        self.source_combo.addItem("Изображение")
        source_layout.addWidget(self.source_combo)

        self.file_path_label = QLabel("Файл не выбран")
        self.file_path_label.setStyleSheet("color: gray;")
        source_layout.addWidget(self.file_path_label)

        self.browse_button = QPushButton("Выбрать файл...")
        self.browse_button.clicked.connect(self.browse_file)
        self.browse_button.setVisible(False)
        source_layout.addWidget(self.browse_button)

        main_layout.addWidget(source_group)

        # Обработчик изменения выбора источника
        self.source_combo.currentIndexChanged.connect(self.update_source_ui)

        # Группа действий
        action_group = QGroupBox("Действия")
        action_layout = QVBoxLayout(action_group)

        # Кнопки действий
        self.capture_button = QPushButton("Сбор данных обучения")
        self.capture_button.setIcon(QIcon.fromTheme("camera"))
        self.capture_button.clicked.connect(self.capture_data)

        self.train_button = QPushButton("Тренировка модели")
        self.train_button.setIcon(QIcon.fromTheme("tools"))
        self.train_button.clicked.connect(self.train_model)

        # Группа для кнопок запуска моделей
        run_group = QGroupBox("Запуск распознавания")
        run_layout = QHBoxLayout(run_group)

        self.run_gd_button = QPushButton("Модель (Градиентный спуск)")
        self.run_gd_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.run_gd_button.clicked.connect(lambda: self.run_recognition('gradient'))
        self.run_gd_button.setStyleSheet("background-color: #4CAF50; color: white;")

        self.run_backprop_button = QPushButton("Модель (Обратное распространение)")
        self.run_backprop_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.run_backprop_button.clicked.connect(lambda: self.run_recognition('back'))
        self.run_backprop_button.setStyleSheet("background-color: #2196F3; color: white;")

        run_layout.addWidget(self.run_gd_button)
        run_layout.addWidget(self.run_backprop_button)

        action_layout.addWidget(self.capture_button)
        action_layout.addWidget(self.train_button)
        action_layout.addWidget(run_group)

        main_layout.addWidget(action_group)

        # Статус бар
        self.statusBar().showMessage("Готов к работе")

        # Инициализация переменных
        self.selected_file = ""
        self.current_game = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Виджет для отображения видео (если нужно)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label)
        self.video_label.hide()  # Скрываем по умолчанию

    def update_source_ui(self, index):
        if index == 0:  # Камера
            self.browse_button.setVisible(False)
            self.file_path_label.setText("Используется встроенная камера")
            self.file_path_label.setStyleSheet("color: green;")
        else:  # Видеофайл или изображение
            self.browse_button.setVisible(True)
            if not self.selected_file:
                self.file_path_label.setText("Файл не выбран")
                self.file_path_label.setStyleSheet("color: gray;")

    def browse_file(self):
        index = self.source_combo.currentIndex()
        if index == 1:  # Видеофайл
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Выберите видеофайл", "",
                "Видео файлы (*.mp4 *.avi *.mov);;Все файлы (*)"
            )
        else:  # Изображение
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Выберите изображение", "",
                "Изображения (*.jpg *.png *.bmp);;Все файлы (*)"
            )

        if file_path:
            self.selected_file = file_path
            self.file_path_label.setText(file_path)
            self.file_path_label.setStyleSheet("color: blue;")
            self.statusBar().showMessage(f"Выбран файл: {file_path}")

    def capture_data(self):
        self.statusBar().showMessage("Запуск сбора данных...")
        QApplication.processEvents()  # Обновляем интерфейс

        # Здесь будет вызов вашей функции сбора данных
        ImageProcessor.capture_training_data(output_dir="rps_data", classes=3, samples_per_class=200)

        self.statusBar().showMessage("Сбор данных завершен!")

    def train_model(self):
        self.statusBar().showMessage("Начало обучения моделей...")
        QApplication.processEvents()  # Обновляем интерфейс

        # Обучаем модель градиентного спуска
        self.statusBar().showMessage("Обучение модели (Градиентный спуск)...")
        game_gd = RockPaperScissorsGame('gd')
        game_gd.train_model(data_dir="rps_data", save_path="rps_model_gradient")

        # Обучаем модель обратного распространения
        self.statusBar().showMessage("Обучение модели (Обратное распространение)...")
        game_bp = RockPaperScissorsGame('backprop')
        game_bp.train_model(data_dir="rps_data", save_path="rps_model_back")

        self.statusBar().showMessage("Обучение моделей завершено!")

    def run_recognition(self, model_type):
        source_index = self.source_combo.currentIndex()
        model_name = "Градиентный спуск" if model_type == 'gd' else "Обратное распространение"

        if source_index == 0:  # Камера
            self.statusBar().showMessage(f"Запуск распознавания с камеры ({model_name})...")
            source = 0
        elif source_index == 1:  # Видеофайл
            if not self.selected_file:
                self.statusBar().showMessage("Ошибка: Не выбран видеофайл!")
                return
            self.statusBar().showMessage(f"Запуск распознавания из видео: {self.selected_file} ({model_name})")
            source = self.selected_file
        else:  # Изображение
            if not self.selected_file:
                self.statusBar().showMessage("Ошибка: Не выбрано изображение!")
                return
            self.statusBar().showMessage(f"Анализ изображения: {self.selected_file} ({model_name})")
            source = self.selected_file

        QApplication.processEvents()  # Обновляем интерфейс

        # Создаем игру с выбранным типом модели
        model_path = f"rps_model_{model_type}.pkl"
        self.current_game = RockPaperScissorsGame(model_path)

        if source_index == 2:  # Для изображения специальная обработка
            self.process_image(self.current_game, source)
        else:
            # Для камеры и видео запускаем в отдельном потоке
            self.video_label.show()
            self.cap = cv2.VideoCapture(source)
            self.timer.start(30)  # Обновление каждые 30 мс

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        # Обработка и отображение кадра
        hands, bboxes, skin_mask = ImageProcessor.preprocess_image(frame, max_hands=2)

        # Отображение результатов
        display_frame = self.draw_results(frame, hands, bboxes)

        # Конвертация для QLabel
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def draw_results(self, frame, hands, bboxes):
        display_frame = frame.copy()

        # Отображение рук
        player_choices = [None, None]
        if len(hands) >= 2:
            for i in range(2):
                x, y, w, h = bboxes[i]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Распознавание жеста
                if hands[i].size == 64 * 64:
                    input_data = hands[i].reshape(1, -1)
                    probs = self.current_game.model.predict_proba(input_data)
                    player_choices[i] = np.argmax(probs)

                    # Отображение результата
                    gesture_name = self.current_game.gesture_names[player_choices[i]]
                    cv2.putText(display_frame, f"Player {i + 1}: {gesture_name}",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Определение победителя
        if None not in player_choices:
            winner = self.current_game.determine_winner(player_choices[0], player_choices[1])

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

            cv2.putText(display_frame, result_text,
                        (frame.shape[1] // 2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return display_frame

    def process_image(self, game, image_path):
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            self.statusBar().showMessage("Ошибка: Не удалось загрузить изображение!")
            return

        # Обработка изображения
        hands, bboxes, _ = ImageProcessor.preprocess_image(image, max_hands=2)

        # Создаем копию для отображения результатов
        result_image = image.copy()

        if len(hands) >= 2:
            # Распознавание жестов
            gestures = []
            for hand in hands[:2]:
                probs = game.model.predict_proba(hand.reshape(1, -1))
                gesture_idx = np.argmax(probs)
                gestures.append(gesture_idx)

            # Определение победителя
            winner = game.determine_winner(gestures[0], gestures[1])

            # Отрисовка результатов
            for i, (bbox, gesture_idx) in enumerate(zip(bboxes[:2], gestures)):
                x, y, w, h = bbox
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_image, f"Player {i + 1}: {game.gesture_names[gesture_idx]}",
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

            cv2.putText(result_image, result_text,
                        (image.shape[1] // 2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(result_image, "Не обнаружены две руки!",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Показ результата
        cv2.imshow("Результат распознавания", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.statusBar().showMessage("Анализ изображения завершен!")


if __name__ == "__main__":
    # Создаем основное приложение
    app = QApplication(sys.argv)

    # Устанавливаем стиль для более современного вида
    app.setStyle("Fusion")

    # Создаем и показываем главное окно
    window = MainWindow()
    window.show()

    # Запускаем цикл обработки событий
    sys.exit(app.exec_())