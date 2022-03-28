from __future__ import annotations

from numpy.typing import NDArray
from typing import Tuple, Union, Optional, List, Any, Callable
from enum import Enum

import numpy as np
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .AudioData import AudioData, VersionControlArray
from .Commands import CommandManager, CommandException

TOOLBAR_POSITION = Enum("TOOLBAR_POSITION", "NONE TOP LEFT RIGHT BOTTOM", start=-1)


class DataViewException(Exception):
    pass


class DataView:
    """ Base class creating container for data able to receive callbacks from commands when added as listener """
    def freq_change_callback(self, data : AudioData, index_start : int, index_end : int) -> None:
        pass


class FigureDataView(DataView):
    """ Data Viewer based on pyplot's FigureCanvasTkAgg """

    def _figure_clicked(self, event):
        """ Mouse click callback"""
        pass

    def __init__(self, root: tk.Tk,
                 position: Union[tk.TOP, tk.BOTTOM, tk.LEFT, tk.RIGHT] = tk.TOP,
                 figsize: Tuple[int, int] = (6,5),
                 dpi: int = 100,
                 toolbar_pos: TOOLBAR_POSITION = TOOLBAR_POSITION.NONE,
                 command_manager: Optional[CommandManager] = None,
                 selection_window: Tuple[Union[float, int], Union[float, int]] = (-1, -1)):
        self.root = root
        self.figsize = figsize
        self.dpi = dpi
        self.position = position
        self.frame = tk.Frame(master=self.root)
        self.figure = plt.Figure(figsize=figsize, dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        self.command_manager = command_manager

        if selection_window[0] != -1 and selection_window[1] != -1:
            self.selection_line = self.ax.axvline(x=selection_window[0])
            self.selection_window = self.ax.axvspan(selection_window[0], selection_window[1] - selection_window[0], alpha=0.5)

        # add plots
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.draw()
        self.canvas.mpl_connect("button_press_event", self._figure_clicked)

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

    def set_selection_window(self, selection_window : Tuple[Union[float, int], Union[float, int]]):
        """ update the selection windows"""

        self.selection_line.set_data([selection_window[0], selection_window[0]], [0,1])
        window_points = self.selection_window.get_xy()
        window_points[:, 0] = [selection_window[0], selection_window[0],
                               selection_window[1], selection_window[1], selection_window[0]]
        self.selection_window.set_xy(window_points)

        self.canvas.draw()


class LineFigureDataView(FigureDataView):
    """ Line Plot """

    def __init__(self, root,
                 data: AudioData,
                 x: List[Callable[[AudioData], Union[NDArray[Any], VersionControlArray]]],
                 y: List[Callable[[AudioData], Union[NDArray[Any], VersionControlArray]]],
                 label: Optional[str] = None,
                 color: Optional[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]] = None,
                 alpha: float = 1.0, *args, **kwargs):
        """ Initialize line plot """
        super().__init__(root, *args, **kwargs)
        self._data = data
        self._x = x
        self._y = y

        xx = [x_tmp(self._data) for x_tmp in x]
        yy = [y_tmp(self._data) for y_tmp in y]

        for i, (x_tmp, y_tmp) in enumerate(zip(xx,yy)):
            setattr(self, f"line_{i}", self.ax.plot(x_tmp, y_tmp, label=label, color=color, alpha=alpha)[0])

    def set_data(self):
        """ change data """

        xx = [x(self._data) for x in self._x]
        yy = [y(self._data) for y in self._y]

        for i, (x_tmp, y_tmp) in enumerate(zip(xx,yy)):
            getattr(self, f"line_{i}").set_xdata(x_tmp)
            getattr(self, f"line_{i}").set_ydata(y_tmp)

        self.canvas.draw()


class ScatterFigureDataView(FigureDataView):
    """ Scatter plot """

    def __init__(self, root,
                 data: AudioData,
                 x: Callable[[AudioData], Union[VersionControlArray, NDArray[Any]]],
                 y: Callable[[AudioData], Union[VersionControlArray, NDArray[Any]]],
                 label: Optional[str] = None,
                 color: Optional[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]] = None,
                 alpha: float = 1.0, *args, **kwargs):
        """ Initialize scatter plot """
        super().__init__(root, *args, **kwargs)
        self._data = data
        self._x = x
        self._y = y
        self.points = self.ax.scatter(x(data), y(data), label=label, color=color, alpha=alpha)

    def set_data(self):
        """ change data """
        data = np.vstack((self._x(self._data), self._y(self._data))).transpose()
        self.points.set_offsets(data)


class BarFigureDataView(FigureDataView):
    """ Bar plot """

    def __init__(self, root,
                 data : AudioData,
                 x: Callable[[AudioData], Union[VersionControlArray, NDArray[Any]]],
                 y: Callable[[AudioData], Union[VersionControlArray, NDArray[Any]]],
                 label: Optional[str] = None,
                 color: Optional[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]] = None,
                 alpha: float = 1.0, *args, **kwargs):
        """ Initialize bar plot """
        super().__init__(root, *args, **kwargs)
        self._data = data
        self._x = x
        self._y = y
        self.bars = self.ax.bar(x(data), y(data), label=label, color=color, alpha=alpha)

    def set_data(self):
        """ change data """
        y = self._y(self._data)
        for j, b in enumerate(self.bars):
            b.set_height(y[j])


def x_data_freq(data: AudioData) -> NDArray[np.float32]:
    """ return frequency x grid"""
    return data.freq_x


def y_data_freq_abs(data: AudioData) -> NDArray[np.float32]:
    """ return absolute value of normalized frequency"""
    return np.abs(data.norm_freq())


def y_data_freq_real(data: AudioData) -> NDArray[np.float32]:
    """ returns real part of normalized frequencies """
    return np.real(data.norm_freq())


def y_data_freq_imag(data: AudioData) -> NDArray[np.float32]:
    """ returns imaginary part of normalized frequencies """
    return np.imag(data.norm_freq())


class EqualizerFigureDataView(LineFigureDataView):
    """ Equalizer plot """
    def __init__(self,
                 root2: tk.Toplevel,
                 position2: Union[tk.TOP, tk.BOTTOM, tk.LEFT, tk.RIGHT] = tk.TOP,
                 toolbar_pos2: TOOLBAR_POSITION = TOOLBAR_POSITION.NONE,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root2 = root2
        self.position2 = position2
        self.frame2 = tk.Frame(master=self.root2)
        self.figure2 = plt.Figure(figsize=self.figsize, dpi=self.dpi)
        self.ax2 = self.figure2.add_subplot(111)
        self.ax2.set_xlim((200,300))
        self.x_span = 100
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=self.frame2)

        xx = [x_tmp(self._data) for x_tmp in self._x]
        yy = [y_tmp(self._data) for y_tmp in self._y]

        for i, (x_tmp, y_tmp) in enumerate(zip(xx,yy)):
            setattr(self, f"line2_{i}", self.ax2.plot(x_tmp, y_tmp)[0])

        self.canvas2.draw()
        self.canvas2.mpl_connect("button_press_event", self._subfigure_clicked)
        self.canvas2.mpl_connect('scroll_event', self._subfigure_scrolled)
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.frame2, pack_toolbar=False)
        self.toolbar2.update()
        if toolbar_pos2 == TOOLBAR_POSITION.TOP:
            self.toolbar2.pack(side=tk.TOP, fill=tk.X)
            self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        elif toolbar_pos2 == TOOLBAR_POSITION.BOTTOM:
            self.toolbar2.pack(side=tk.BOTTOM, fill=tk.X)
            self.canvas2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        elif toolbar_pos2 == TOOLBAR_POSITION.LEFT:
            self.toolbar2.pack(side=tk.LEFT, fill=tk.Y)
            self.canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        elif toolbar_pos2 == TOOLBAR_POSITION.RIGHT:
            self.toolbar2.pack(side=tk.RIGHT, fill=tk.Y)
            self.canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        elif toolbar_pos2 == TOOLBAR_POSITION.NONE:
            self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        else:
            raise DataViewException(f"Unexpected toolbar position: {toolbar_pos}")

        if position2 == tk.TOP:
            self.frame2.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        elif position2 == tk.BOTTOM:
            self.frame2.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        elif position2 == tk.LEFT:
            self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        elif position2 == tk.RIGHT:
            self.frame2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        else:
            raise DataViewException(f"Unexpected figure position: {position2}")

    def set_data(self):
        """ change data """
        super().set_data()

        xx = [x(self._data) for x in self._x]
        yy = [y(self._data) for y in self._y]

        for i, (x_tmp, y_tmp) in enumerate(zip(xx, yy)):
            getattr(self, f"line2_{i}").set_xdata(x_tmp)
            getattr(self, f"line2_{i}").set_ydata(y_tmp)

        self.canvas2.draw()

    def freq_change_callback(self, data: AudioData, index_start: int, index_end: int) -> None:
        self.ax2.set_xlim((data.freq_x[index_start] - self.x_span, data.freq_x[index_end] + self.x_span))
        self.set_data()

    def _subfigure_clicked(self, event):
        """ click on zoomed window"""
        pass

    def _subfigure_scrolled(self, event):
        """ scrolled on zoom window """
        #TODO check if mouse over axis and change x or y limits
        print(event)
        x_lim = self.ax2.get_xlim()
        if event.button == "up":

            self.x_span += 10
            self.ax2.set_xlim(x_lim[0] - 10, x_lim[1] + 10)
        else:
            if self.x_span > 10:
                self.x_span -= 10
                self.ax2.set_xlim(x_lim[0] + 10, x_lim[1] - 10)

        self.canvas2.draw()

    def _figure_clicked(self, event):
        """ Mouse click callback"""
        if self.command_manager is None:
            raise CommandException("DataView does not know about the CommandManager. Callback can not be started")
        self.command_manager.call("equ_clicked", event, self)
