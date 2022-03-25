from numpy.typing import NDArray
from typing import Tuple, Union, Optional
from enum import Enum

import numpy as np
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


TOOLBAR_POSITION = Enum("TOOLBAR_POSITION", "NONE TOP LEFT RIGHT BOTTOM", start=-1)


class DataViewException(Exception):
    pass


class DataView:
    """ Base class creating container for data able to receive callbacks from commands when added as listener """
    def freq_change_callback(self, index_start : int, index_end : int) -> None:
        pass


class FigureDataView(DataView):
    """ Data Viewer based on pyplot's FigureCanvasTkAgg """

    def figure_clicked(self, event):
        """ Mouse click callback"""
        pass

    def __init__(self, root : tk.Tk, position : Union[tk.TOP, tk.BOTTOM, tk.LEFT, tk.RIGHT] = tk.TOP, figsize : Tuple[int, int] = (6,5), dpi : int = 100, toolbar_pos : TOOLBAR_POSITION = TOOLBAR_POSITION.NONE):
        self.root = root
        self.figsize = figsize
        self.dpi = dpi
        self.position = position
        self.frame = tk.Frame(master=self.root)
        self.figure = plt.Figure(figsize=figsize, dpi=dpi)
        self.ax = self.figure.add_subplot(111)

        # add plots
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.draw()
        self.canvas.mpl_connect("button_press_event", self.figure_clicked)

        # add toolbar and pack everything to the frame
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame, pack_toolbar=False)
        self.toolbar.update()

        if toolbar_pos == TOOLBAR_POSITION.TOP:
            self.toolbar.pack(side=tk.TOP, fill=tk.X)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        elif toolbar_pos == TOOLBAR_POSITION.BOTTOM:
            self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        elif toolbar_pos == TOOLBAR_POSITION.LEFT:
            self.toolbar.pack(side=tk.LEFT, fill=tk.Y)
            self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        elif toolbar_pos == TOOLBAR_POSITION.RIGHT:
            self.toolbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        elif toolbar_pos == TOOLBAR_POSITION.NONE:
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        else:
            raise DataViewException(f"Unexpected toolbar position: {toolbar_pos}")

        if position == tk.TOP:
            self.frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        elif position == tk.BOTTOM:
            self.frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        elif position == tk.LEFT:
            self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        elif position == tk.RIGHT:
            self.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        else:
            raise DataViewException(f"Unexpected figure position: {position}")


class LineFigureDataView(FigureDataView):
    """ Line Plot """

    def __init__(self, root, x, y,
                 label : Optional[str] = None,
                 color : Optional[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]] = None,
                 alpha : float = 1.0, *args, **kwargs):
        """ Initialize line plot """
        super().__init__(root, *args, **kwargs)
        self.line, = self.ax.plot(x, y, label=label, color=color, alpha=alpha)

    def set_data(self, x, y):
        """ change data """
        self.line.set_xdata(x)
        self.line.set_ydata(y)

        self.canvas.draw()


class ScatterFigureDataView(FigureDataView):
    """ Scatter plot """

    def __init__(self, root, x, y,
                 label : Optional[str] = None,
                 color : Optional[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]] = None,
                 alpha : float = 1.0, *args, **kwargs):
        """ Initialize scatter plot """
        super().__init__(root, *args, **kwargs)
        self.points = self.ax.scatter(x, y, label=label, color=color, alpha=alpha)

    def set_data(self, x, y):
        """ change data """
        data = np.vstack((x, y)).transpose()
        self.points.set_offsets(data)


class BarFigureDataView(FigureDataView):
    """ Bar plot """

    def __init__(self, root, x, y,
                 label : Optional[str] = None,
                 color : Optional[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]] = None,
                 alpha : float = 1.0, *args, **kwargs):
        """ Initialize bar plot """
        super().__init__(root, *args, **kwargs)
        self.bars = self.ax.bar(x, y, label=label, color=color, alpha=alpha)

    def set_data(self, y):
        """ change data """
        for j, b in enumerate(self.bars):
            b.set_height(y[j])
