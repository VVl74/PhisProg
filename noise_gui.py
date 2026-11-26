"""Desktop interface for the NACA 0012 noise model."""

import math
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from airfoil_noise import predict


FIELDS = (
    ("Частота", "Гц", "800"),
    ("Угол атаки", "°", "1,5"),
    ("Длина хорды", "м", "0,3048"),
    ("Скорость потока", "м/с", "71,3"),
    ("Толщина вытеснения", "м", "0,00336729"),
)
FREQUENCIES = np.geomspace(200, 20_000, 160)
BACKGROUND = "#edf2f6"
PANEL = "#ffffff"
INK = "#183044"
MUTED = "#4a6072"
ACCENT = "#006c70"


def read_parameters(fields):
    values = []
    for index, (field, (label, _, _)) in enumerate(zip(fields, FIELDS)):
        try:
            value = float(field.strip().replace(",", "."))
        except ValueError:
            raise ValueError(f"«{label}» — введите число.") from None
        if not math.isfinite(value):
            raise ValueError(f"«{label}» — введите конечное число.")
        if index != 1 and value <= 0:
            raise ValueError(f"«{label}» — значение должно быть больше нуля.")
        values.append(value)
    return values


def predict_spectrum(parameters):
    conditions = np.tile(parameters, (len(FREQUENCIES) + 1, 1))
    conditions[1:, 0] = FREQUENCIES
    levels, outside = predict(conditions)
    if not levels.isfinite().all():
        raise ValueError("Не удалось рассчитать уровень для этих параметров.")
    return levels[0].item(), levels[1:].numpy(), outside[0].item(), outside[1:].any().item()


class NoiseWindow:
    def __init__(self, root):
        self.root = root
        root.title("Аэродинамический шум — NACA 0012")
        root.geometry("1050x660")
        root.minsize(780, 640)
        root.configure(background=BACKGROUND)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TEntry", padding=8, fieldbackground=PANEL, foreground=INK)
        style.configure("TButton", padding=(9, 9), foreground=INK)
        style.configure("Accent.TButton", background=ACCENT, foreground=PANEL)
        style.map("Accent.TButton", background=[("active", "#00565a")])

        header = tk.Frame(root, bg=BACKGROUND)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(18, 16))
        self.label(header, "Аэродинамический шум", 27, background=BACKGROUND, bold=True).pack(anchor="w")
        self.label(header, "Прогноз для профиля NACA 0012", 14, color=MUTED,
                   background=BACKGROUND).pack(anchor="w", pady=(3, 0))

        form = tk.Frame(root, bg=PANEL, padx=18, pady=16)
        form.grid(row=1, column=0, sticky="nsew", padx=(24, 12), pady=(0, 24))
        form.columnconfigure(0, weight=1)
        self.label(form, "Параметры", 17, bold=True).grid(row=0, column=0, columnspan=2, sticky="w")
        self.fields = []
        for index, (label, unit, initial) in enumerate(FIELDS):
            row = 2 * index + 1
            self.label(form, label, 13).grid(row=row, column=0, sticky="w", pady=(9, 3))
            field = tk.StringVar(root, initial)
            self.fields.append(field)
            entry = ttk.Entry(form, textvariable=field, width=18)
            entry.grid(row=row + 1, column=0, sticky="ew")
            if index == 0:
                self.first_entry = entry
            self.label(form, unit, 12, color=MUTED).grid(row=row + 1, column=1, sticky="w", padx=(9, 0))

        ttk.Button(form, text="Рассчитать", command=self.calculate,
                   style="Accent.TButton").grid(row=11, column=0, columnspan=2, sticky="ew", pady=(14, 5))
        ttk.Button(form, text="Подставить пример", command=self.load_example).grid(
            row=12, column=0, columnspan=2, sticky="ew"
        )

        result = tk.Frame(root, bg=PANEL, padx=18, pady=14)
        result.grid(row=1, column=1, sticky="nsew", padx=(0, 24), pady=(0, 24))
        self.label(result, "Уровень на заданной частоте", 14, color=MUTED).pack(anchor="w")
        self.level = self.label(result, "—", 30, color=ACCENT, bold=True)
        self.level.pack(anchor="w")
        self.status = self.label(result, "", 12, color=MUTED)
        self.status.configure(wraplength=430)
        self.status.pack(anchor="w", pady=(2, 2))
        result.bind("<Configure>", lambda event: self.status.configure(wraplength=max(200, event.width - 40)))

        figure = Figure(figsize=(5.5, 3.0), dpi=100, layout="constrained", facecolor=PANEL)
        self.ax = figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(figure, master=result)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        root.bind("<Return>", self.calculate)
        root.bind("<KP_Enter>", self.calculate)
        self.calculate()
        self.first_entry.focus_set()

    def label(self, parent, text, size, *, color=INK, background=PANEL, bold=False):
        return tk.Label(parent, text=text, bg=background, fg=color,
                        font=("Helvetica", size, "bold" if bold else "normal"), justify="left")

    def load_example(self):
        for field, (_, _, initial) in zip(self.fields, FIELDS):
            field.set(initial)
        self.calculate()

    def calculate(self, event=None):
        try:
            parameters = read_parameters([field.get() for field in self.fields])
            level, spectrum, outside_point, outside_spectrum = predict_spectrum(parameters)
        except (ValueError, OSError) as error:
            self.level.configure(text="—")
            message = str(error) if isinstance(error, ValueError) else (
                "Не удалось прочитать веса модели. Запусти Train Airfoil Noise."
            )
            self.status.configure(text=message, fg="#a83232")
            self.draw_spectrum()
            return

        frequency = parameters[0]
        self.level.configure(text=f"{level:.2f} дБ".replace(".", ","))
        if outside_point:
            message = "Один или несколько параметров вне диапазона обучения. Точность может быть ниже."
        elif outside_spectrum:
            message = "Часть спектра выходит за диапазон частот обучения."
        else:
            message = f"Масштабированный уровень звукового давления на {frequency:g} Гц."
        self.status.configure(text=message, fg="#925000" if outside_point or outside_spectrum else MUTED)
        self.draw_spectrum(frequency, level, spectrum)

    def draw_spectrum(self, frequency=None, level=None, spectrum=None):
        self.ax.clear()
        self.ax.set(xscale="log", xlim=(200, 20_000), xlabel="Частота, Гц",
                    ylabel="Уровень, дБ", title="Прогноз спектра")
        ticks = (200, 500, 1000, 2000, 5000, 10_000, 20_000)
        self.ax.set_xticks(ticks)
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        self.ax.grid(alpha=0.2)
        self.ax.spines[["top", "right"]].set_visible(False)
        if spectrum is not None:
            self.ax.plot(FREQUENCIES, spectrum, color=ACCENT, linewidth=2.2)
            if FREQUENCIES[0] <= frequency <= FREQUENCIES[-1]:
                self.ax.scatter(frequency, level, s=49, color="#bb5a23", edgecolor=PANEL, zorder=3)
        self.canvas.draw_idle()


def main():
    torch.set_num_threads(1)
    root = tk.Tk()
    NoiseWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
