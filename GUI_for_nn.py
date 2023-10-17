import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import Image, ImageOps, ImageGrab
import numpy as np


model = load_model('../16_model.keras')


def predict_digit(img):
    # изменение размера изображений на 28x28
    img = img.resize((28, 28))
    # используем библиотеку opencv для инверсирования цветов
    img = ImageOps.invert(img)
    # конвертируем rgb в grayscale
    img = img.convert('L')
    img = np.array(img)

    # изменение размерности для поддержки модели ввода и нормализации
    img1 = img.reshape(-1, 28, 28, 1)

    img1 = img1 / 255

    # предсказание цифры

    res = model.predict(img1)
    print(res)
    res = list(res)
    print(res)
    return np.argmax(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Создание элементов
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Думаю..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Распознать", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Очистить", command=self.clear_all)

        # Сетка окна
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        rect = (self.winfo_x(), self.winfo_y(), self.winfo_x() + self.canvas.winfo_width(), self.winfo_y() + self.canvas.winfo_height())
        im = ImageGrab.grab(rect)
        digit = predict_digit(im)
        self.label.configure(text=str(digit))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


app = App()
mainloop()