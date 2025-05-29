import tkinter as tk
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageDraw

from main import NeuralNetwork2

GRID_SIZE = 28
CELL_SIZE = 10
CANVAS_HEIGHT = 280
CANVAS_WIDTH = 280
WHITE_THRESHOLD = 128
cell_width = CANVAS_WIDTH // GRID_SIZE
cell_height = CANVAS_HEIGHT // GRID_SIZE

class GUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("MNIST rozpoznawanie liczb")
        self.grid = []
        self.pixels = []
        self.canvas = tk.Canvas(self, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
        self.canvas.pack()
        self.grid_state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.cell_coords = []
        self.canvas.bind('<B1-Motion>', self.draw_number)

        self.create_grid()
        self.create_panel()

        print("Loading MNIST data...")
        training_data = np.loadtxt("mnist_train.csv", delimiter=',', dtype=np.float32, skiprows=1)
        test_data = np.loadtxt("mnist_test.csv", delimiter=',', dtype=np.float32, skiprows=1)
        print("training_data.shape = ", training_data.shape, " ,  test_data.shape = ", test_data.shape)

        output_nodes = 10
        epochs = 1
        self.nn = NeuralNetwork2(784, 100, 10, 0.1)
        for i in range(epochs):
            for step in range(len(training_data)):
                target_data = np.zeros(output_nodes) + 0.01
                target_data[int(training_data[step, 0])] = 0.99
                input_data = ((training_data[step, 1:] / 255.0) * 0.99) + 0.01
                self.nn.train(np.array(input_data, ndmin=2), np.array(target_data, ndmin=2))
                if step % 2000 == 0:
                    print("step = ", step,  ",  loss_val = ", self.nn.loss_val())

    def create_grid(self):
        for row in range(GRID_SIZE):
            grid_row = []
            for col in range(GRID_SIZE):
                x1 = col * CELL_SIZE
                y1 = row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                cell = self.canvas.create_rectangle(x1, y1, x2, y2, outline='black', fill='white')
                self.cell_coords.append((x1, y1, x2, y2))
                grid_row.append(cell)
                self.grid_state[row][col] = 0
                self.grid.append(cell)
            self.grid.append(grid_row)
        # self.grid.append(cell)
        # self.cell_coords.append((int(x1), int(y1)))

    def create_panel(self):
        panel = tk.Frame(self)
        panel.pack(pady=10)
        clean_button = tk.Button(panel, text="Clean", command=self.clean)
        clean_button.pack(side='left', padx=10)
        self.update()

    def draw_number(self, event):
        x, y = event.x, event.y
        item = self.canvas.find_closest(x, y)[0]
        x1, y1, x2, y2 = self.canvas.coords(item)
        row = int((y1 + y2) / 2 // CELL_SIZE)
        col = int((x1 + x2) / 2 // CELL_SIZE)
        self.canvas.itemconfig(item, fill='black')
        self.grid_state[row][col] = 1
        current_image = np.array(self.grid_state, dtype=float).reshape(1, 784)
        prediction = self.nn.predict(current_image)
        print(f"Predicted digit: {prediction}")

    def toPredict(self):
        current_image = np.array(self.grid_state, dtype=float).reshape(1, 784)
        prediction = self.nn.predict(current_image)
        print(f"Predicted digit: {prediction}")

    def clean(self):
        self.grid_state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.canvas.delete("all")
        self.grid = []
        self.create_grid()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(master=root)
    app.pack()
    app.mainloop()

