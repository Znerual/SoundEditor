from __future__ import annotations

import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

import time
import matplotlib

from numpy.typing import NDArray
from typing import Tuple, Union, Optional, List, Any, Callable
from enum import Enum
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from dataclasses import dataclass

from .AudioData import AudioData, VersionControlArray
from .Commands import CommandManager, CommandException
from .helper import bell_curve, DATA_MODE

TOOLBAR_POSITION = Enum("TOOLBAR_POSITION", "NONE TOP LEFT RIGHT BOTTOM", start=-1)


class DataViewException(Exception):
    pass


def zoom_invariant_view(data: NDArray[Any], limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> int:
    """ returns a portion of the data that stays invariant under zooming """
    all_visible = data.shape[0] / points_on_screen
    if limits == (-1, -1):
        return int(all_visible)

    step = int((limits[1] - limits[0]) / (max_limits[1] - max_limits[0]) * all_visible)

    return step if step > 0 else 1


def max_sample(x: NDArray[Any], f) -> NDArray[Any]:
    """ returns maximum of every f parts of the array"""
    xp = np.r_[x, np.nan + np.zeros((-len(x) % f,))]
    # reshape, so each chunk gets its own row, and then take mean
    return np.nanmax(xp.reshape(-1, f), axis=-1)


def min_sample(x: NDArray[Any], f) -> NDArray[Any]:
    """ returns minimum of every f parts of the array"""
    xp = np.r_[x, np.nan + np.zeros((-len(x) % f,))]
    # reshape, so each chunk gets its own row, and then take mean
    return np.nanmin(xp.reshape(-1, f), axis=-1)


def down_sample(x: NDArray[Any], f) -> NDArray[Any]:
    # https://stackoverflow.com/questions/41815361/average-every-x-numbers-in-numpy-array
    # pad to a multiple of f, so we can reshape
    # use nan for padding, so we needn't worry about denominator in
    # last chunk
    xp = np.r_[x + 1, np.nan + np.zeros((-len(x) % f,))]
    # reshape, so each chunk gets its own row, and then take mean
    return np.nanmean(xp.reshape(-1, f), axis=-1) - 1


def x_data_time(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 1024) -> NDArray[np.float32]:
    """ return time x grid"""

    return data.time_x[::zoom_invariant_view(data.time_x, limits, max_limits, points_on_screen)]


def y_data_time(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 1024) -> NDArray[np.float32]:
    """ return time """
    return down_sample(np.abs(data.time[:, 0]), zoom_invariant_view(data.time_x, limits, max_limits, points_on_screen))


def x_window_time(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ return x data of selected time window """
    return data.time_x[data.start_index:data.end_index:zoom_invariant_view(data.time_x[data.start_index:data.end_index], limits, max_limits, points_on_screen)]


def x_rel_window_time(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ return x data relative to current selected window"""
    return data.time_x[0:data.N:zoom_invariant_view(data.time_x[0:data.N], limits, max_limits, points_on_screen)]


def y_window_time_max(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ return maximum of selected window of time """
    return max_sample(data.time[data.start_index:data.end_index, 0], zoom_invariant_view(data.time_x[data.start_index:data.end_index], limits, max_limits, points_on_screen))


def y_window_time_min(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ returns minimum of selected window of time"""
    return min_sample(data.time[data.start_index:data.end_index, 0],
               zoom_invariant_view(data.time_x[data.start_index:data.end_index], limits, max_limits, points_on_screen))


def y_window_damping_min(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ return damping of time before fourier transformation """
    return min_sample(data.damping, zoom_invariant_view(data.time_x[data.start_index:data.end_index], limits, max_limits, points_on_screen))


def y_window_damping_max(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ return damping of time before fourier transformation """
    return max_sample(data.damping, zoom_invariant_view(data.time_x[data.start_index:data.end_index], limits, max_limits, points_on_screen))


def y_window_time_damped_min(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ returns damped audio data """

    return min_sample(data.time[data.start_index:data.end_index, 0] * data.damping,
                           zoom_invariant_view(data.time_x[data.start_index:data.end_index], limits, max_limits,
                                               points_on_screen))


def y_window_time_damped_max(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float],
                             points_on_screen=480) -> NDArray[np.float32]:
    """ returns damped audio data """

    return max_sample(data.time[data.start_index:data.end_index, 0] * data.damping,
                      zoom_invariant_view(data.time_x[data.start_index:data.end_index], limits, max_limits,
                                          points_on_screen))


def y_window_invdamping(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ returns the inverse damping """
    return data.inv_damping[::zoom_invariant_view(data.time_x[data.start_index:data.end_index], limits, max_limits, points_on_screen)]


def x_data_freq(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ return frequency x grid"""
    return data.freq_x[::zoom_invariant_view(data.freq_x, limits, max_limits, points_on_screen)]


def y_data_freq_abs(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ return absolute value of normalized frequency"""
    norm_freq = data.norm_freq()
    return down_sample(np.abs(norm_freq), zoom_invariant_view(norm_freq, limits, max_limits, points_on_screen))


def y_data_freq_real_min(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ returns real part of normalized frequencies """
    norm_freq = data.norm_freq()
    return min_sample(np.real(norm_freq),zoom_invariant_view(norm_freq, limits, max_limits, points_on_screen))


def y_data_freq_real_max(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ returns real part of normalized frequencies """
    norm_freq = data.norm_freq()
    return max_sample(np.real(norm_freq),zoom_invariant_view(norm_freq, limits, max_limits, points_on_screen))


def y_data_freq_imag_min(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ returns imaginary part of normalized frequencies """
    norm_freq = data.norm_freq()
    return min_sample(np.imag(norm_freq),zoom_invariant_view(norm_freq, limits, max_limits, points_on_screen))


def y_data_freq_imag_max(data: AudioData, limits: Tuple[float, float], max_limits: Tuple[float, float], points_on_screen = 480) -> NDArray[np.float32]:
    """ returns imaginary part of normalized frequencies """
    norm_freq = data.norm_freq()
    return max_sample(np.imag(norm_freq),zoom_invariant_view(norm_freq, limits, max_limits, points_on_screen))

@dataclass(frozen=True, eq=False)
class MouseEventData:
    time: float
    event: matplotlib.backend_bases.MouseEvent


class DataView:
    """ Base class creating container for data able to receive callbacks from commands when added as listener """
    def freq_change_callback(self, data: AudioData, index_start: int, index_end: int) -> None:
        pass

    def time_change_callback(self, data: AudioData, index_start: int, index_end: int) -> None:
        pass

    def play_timecode_callback(self, data: AudioData, current_frame) -> None:
        pass

    def reset_view_callback(self, data: AudioData) -> None:
        pass


class FigureDataView(DataView):
    """ Data Viewer based on pyplot's FigureCanvasTkAgg """

    def _figure_mouse_press(self, event):
        """ Mouse click callback"""
        self._mouse_press_data = MouseEventData(time=time.time(), event=event)

    def _figure_mouse_release(self, event):
        """ Mouse release callback"""
        mouse_event: MouseEventData = MouseEventData(time=time.time(), event=event)

        if self._mouse_click(self._mouse_press_data, mouse_event):
            self._figure_clicked(event)

    def _figure_clicked(self, event):
        """ Mouse click callback """
        pass

    def _mouse_click(self, mouse_press_data: MouseEventData, mouse_release_data: MouseEventData):
        """ Checks whether a mouse click was a click or a drag """
        start_time, start_event = mouse_press_data.time, mouse_press_data.event
        end_time, end_event = mouse_release_data.time, mouse_release_data.event
        delta_time = end_time - start_time
        delta_pos = np.linalg.norm(np.array([end_event.x, end_event.y]) - np.array([start_event.x, start_event.y]), ord=2)

        if delta_time <= 0.1 or (delta_time <= 0.25 and delta_pos <= 5):
            return True

        return False

    def _figure_mouse_moved(self, event: matplotlib.backend_bases.MouseEvent):
        """ pan if mouse is moved while holding the middle mouse button """
        if event.button == matplotlib.backend_bases.MouseButton.MIDDLE:
            self._pan(self.canvas, self.ax.transData, self._mouse_pos, (event.x, event.y), (self.ax.get_xlim, self.ax.get_ylim), (self.ax.set_xlim, self.ax.set_ylim))

        self._mouse_pos = (event.x, event.y)
        self._mouse_pos_data = (event.xdata, event.ydata)

    def _pan(self, canvas: FigureCanvasTkAgg,
             dataTrans: matplotlib.axes.Axes.transData,
             old_mouse_position: Tuple[float, float],
             current_mouse_position: Tuple[float, float],
             get_lim: Tuple[Callable[[FigureDataView], Any], Callable[[FigureDataView], Any]],
             set_lim: Tuple[Callable[[int, int], None], Callable[[int, int], None]]):
        """ pan the selection around """
        x_start, y_start = old_mouse_position
        x_end, y_end = current_mouse_position

        # get axis limits
        cur_lim1_data = np.array(get_lim[0]())
        cur_lim2_data = np.array(get_lim[1]())

        # transform to display coordinates
        cur_lim1 = np.empty((2,), dtype=int)
        cur_lim2 = np.empty((2,), dtype=int)
        cur_lim1[0], cur_lim2[0] = dataTrans.transform((cur_lim1_data[0], cur_lim2_data[0]))
        cur_lim1[1], cur_lim2[1] = dataTrans.transform((cur_lim1_data[1], cur_lim2_data[1]))

        # calculate scaling factor
        pix2x = (cur_lim1_data[1] - cur_lim1_data[0]) / (cur_lim1[1] - cur_lim1[0])
        pix2y = (cur_lim2_data[1] - cur_lim2_data[0]) / (cur_lim2[1] - cur_lim2[0])

        # set new limits
        set_lim[0](cur_lim1_data - (x_end - x_start) * pix2x)
        set_lim[1](cur_lim2_data - (y_end - y_start) * pix2y)

        canvas.draw()

    def zoom_1d(self, canvas: FigureCanvasTkAgg,
                event: matplotlib.backend_bases.MouseEvent, get_lim: Callable[[FigureDataView], Any],
                set_lim: Callable[[int, int], None], pos: float, base_scale: float = 1.05) -> None:
        """ zoom over one axis """
        cur_lim = get_lim()
        cur_range = (cur_lim[1] - cur_lim[0]) * 0.5
        midpoint = (cur_lim[1] + cur_lim[0]) * 0.5
        if event.button == "up":
            scale_factor = 1 / base_scale
        else:
            scale_factor = base_scale

        set_lim([midpoint - cur_range * scale_factor, midpoint + cur_range * scale_factor])
        canvas.draw()

    def zoom_2d(self, canvas: FigureCanvasTkAgg,
                event: matplotlib.backend_bases.MouseEvent,
                get_lim: Tuple[Callable[[FigureDataView], Any], Callable[[FigureDataView], Any]],
                set_lim: Tuple[Callable[[int, int], None], Callable[[int, int], None]], pos: Tuple[float, float],
                base_scale: float = 1.1) -> None:
        """ zoom over both axes """
        cur_lim1 = get_lim[0]()
        cur_lim2 = get_lim[1]()
        cur_range1 = (cur_lim1[1] - cur_lim1[0]) * 0.5
        cur_range2 = (cur_lim2[1] - cur_lim2[0]) * 0.5
        cur_midpoint1 = (cur_lim1[1] + cur_lim1[0]) * 0.5
        cur_midpoint2 = (cur_lim2[1] + cur_lim2[0]) * 0.5
        if event.button == "up":
            scale_factor = 1 / base_scale
        else:
            scale_factor = base_scale

        set_lim[0]([cur_midpoint1 + (pos[0] - cur_midpoint1) / 10 - cur_range1 * scale_factor, cur_midpoint1 + (pos[0] - cur_midpoint1) / 10 + cur_range1 * scale_factor])
        set_lim[1]([cur_midpoint2 + (pos[1] - cur_midpoint2) / 10 - cur_range2 * scale_factor, cur_midpoint2 + (pos[1] - cur_midpoint2) / 10 + cur_range2 * scale_factor])

        canvas.draw()

    def _figure_scrolled(self, event):
        """ scrolled on zoom window """
        x, y = self.ax.transAxes.inverted().transform((event.x, event.y))

        if x <= 0 < y:
            self.zoom_1d(self.canvas, event, self.ax.get_ylim, self.ax.set_ylim, event.ydata)
        elif 0 >= y < x:
            self.zoom_1d(self.canvas, event, self.ax.get_xlim, self.ax.set_xlim, event.xdata)
        elif 0 <= x <= 1 and 0 <= y <= 1:
            self.zoom_2d(self.canvas, event, (self.ax.get_xlim, self.ax.get_ylim), (self.ax.set_xlim, self.ax.set_ylim), (event.xdata, event.ydata))

    def __init__(self, root: tk.Tk,
                 position: Union[tk.TOP, tk.BOTTOM, tk.LEFT, tk.RIGHT] = tk.TOP,
                 figsize: Tuple[int, int] = (6,5),
                 dpi: int = 100,
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
        self._mouse_press_data: Optional[MouseEventData] = None
        self._mouse_pos: Tuple[float, float] = (-1.0, -1.0)
        self._mouse_pos_data: Tuple[float, float] = (-1.0, -1.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        if selection_window[0] != -1 and selection_window[1] != -1:
            self.selection_line = self.ax.axvline(x=selection_window[0], color="red", zorder=3)
            self.selection_window = self.ax.axvspan(selection_window[0], selection_window[1], alpha=0.5)

        # add plots
        self.canvas.draw()
        self.canvas.mpl_connect("button_press_event", self._figure_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._figure_mouse_release)
        self.canvas.mpl_connect("scroll_event", self._figure_scrolled)
        self.canvas.mpl_connect("motion_notify_event", self._figure_mouse_moved)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

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

    def set_selection_window(self, selection_window: Tuple[Union[float, int], Union[float, int]]):
        """ update the selection windows"""

        self.set_selection_line(selection_window[0])
        window_points = self.selection_window.get_xy()
        window_points[:, 0] = [selection_window[0], selection_window[0],
                               selection_window[1], selection_window[1], selection_window[0]]
        self.selection_window.set_xy(window_points)

        self.canvas.draw()

    def set_selection_line(self, selection_line: Union[float,int]):
        """ moved the selection line to the specified x position"""
        self.selection_line.set_data([selection_line, selection_line], [0, 1])
        self.canvas.draw()


class LineFigureDataView(FigureDataView):
    """ Line Plot """

    def __init__(self, root,
                 x: List[Callable[[AudioData, Tuple[float, float], Tuple[float, float], int], Union[NDArray[Any], VersionControlArray]]],
                 y: List[Callable[[AudioData, Tuple[float, float], Tuple[float, float], int], Union[NDArray[Any], VersionControlArray]]],
                 legend: List[str] = [],
                 color: Optional[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]] = None,
                 alpha: float = 1.0,
                 points_per_view:int = 480, *args, **kwargs):
        """ Initialize line plot """
        super().__init__(root, *args, **kwargs)
        self._x = x
        self._y = y
        self._points_per_view = points_per_view
        xx = [x_tmp(self.command_manager.data, (-1, -1), (-1, -1), self._points_per_view) for x_tmp in x]
        yy = [y_tmp(self.command_manager.data, (-1, -1), (-1, -1), self._points_per_view) for y_tmp in y]

        # create plots
        for i, (x_tmp, y_tmp) in enumerate(zip(xx, yy)):
            setattr(self, f"line_{i}", self.ax.plot(x_tmp, y_tmp, color=color, alpha=alpha)[0])

        # create legend
        if not legend == []:
            self.ax.legend(legend, loc='upper right')

        # set maximal xlim for down sampling
        self.xlim_max = self.ax.get_xlim()

    def set_data(self):
        """ change data """

        xx = [x(self.command_manager.data, self.ax.get_xlim(), self.xlim_max, self._points_per_view) for x in self._x]
        yy = [y(self.command_manager.data, self.ax.get_xlim(), self.xlim_max, self._points_per_view) for y in self._y]

        for i, (x_tmp, y_tmp) in enumerate(zip(xx, yy)):
            getattr(self, f"line_{i}").set_xdata(x_tmp)
            getattr(self, f"line_{i}").set_ydata(y_tmp)

        self.canvas.draw()

    def _figure_scrolled(self, event):
        super()._figure_scrolled(event)

        self.set_data()

'''
class ScatterFigureDataView(FigureDataView):
    """ Scatter plot """

    def __init__(self, root,
                 x: Callable[[AudioData], Union[VersionControlArray, NDArray[Any]]],
                 y: Callable[[AudioData], Union[VersionControlArray, NDArray[Any]]],
                 label: Optional[str] = None,
                 color: Optional[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]] = None,
                 alpha: float = 1.0, *args, **kwargs):
        """ Initialize scatter plot """
        super().__init__(root, *args, **kwargs)
        self._x = x
        self._y = y
        self.points = self.ax.scatter(x(self.command_manager.data), y(self.command_manager.data), label=label, color=color, alpha=alpha)

    def set_data(self):
        """ change data """
        data = np.vstack((self._x(self.command_manager.data), self._y(self.command_manager.data))).transpose()
        self.points.set_offsets(data)


class BarFigureDataView(FigureDataView):
    """ Bar plot """

    def __init__(self, root,
                 x: Callable[[AudioData], Union[VersionControlArray, NDArray[Any]]],
                 y: Callable[[AudioData], Union[VersionControlArray, NDArray[Any]]],
                 label: Optional[str] = None,
                 color: Optional[Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]] = None,
                 alpha: float = 1.0, *args, **kwargs):
        """ Initialize bar plot """
        super().__init__(root, *args, **kwargs)
        self._x = x
        self._y = y
        self.bars = self.ax.bar(x(self.command_manager.data), y(self.command_manager.data), label=label, color=color, alpha=alpha)

    def set_data(self):
        """ change data """
        y = self._y(self.command_manager.data)
        for j, b in enumerate(self.bars):
            b.set_height(y[j])

'''


class EqualizerFigureDataView(LineFigureDataView):
    """ Equalizer plot """
    def __init__(self, *args,toolbar=False, **kwargs):
        super().__init__(*args, **kwargs)

        # settings for gaussian curve
        self.bell, = self.ax.plot([self._x[0](self.command_manager.data, (-1,-1), (-1,-1), 20)[0]], [self._y[0](self.command_manager.data, (-1,-1), (-1,-1), 20)[0]], 'o', ms=4, alpha=0.8, color='red', visible=False)
        self.bell_halve_width = 2
        self.show_bell = False

        # data mode
        self.data_mode = DATA_MODE.REPLACE
        # custom toolbar
        if toolbar:
            self.toolbar_frame = tk.Frame(self.frame)
            self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            self.button_add = tk.Button(self.toolbar_frame, text="Add", command=self._add_button_clicked)
            self.button_subtract = tk.Button(self.toolbar_frame, text="Subtract", command=self._subtract_button_clicked)
            self.button_replace = tk.Button(self.toolbar_frame, text="Replace", state=tk.DISABLED, command=self._replace_button_clicked)
            self.button_add.pack(side=tk.LEFT)
            self.button_subtract.pack(side=tk.LEFT)
            self.button_replace.pack(side=tk.LEFT)

    def _enable_all_mode_buttons(self):
        self.button_add["state"] = tk.NORMAL
        self.button_subtract["state"] = tk.NORMAL
        self.button_replace["state"] = tk.NORMAL

    def _add_button_clicked(self):
        self._enable_all_mode_buttons()
        self.button_add["state"] = tk.DISABLED
        self.data_mode = DATA_MODE.ADD

    def _subtract_button_clicked(self):
        self._enable_all_mode_buttons()
        self.button_subtract["state"] = tk.DISABLED
        self.data_mode = DATA_MODE.SUBTRACT

    def _replace_button_clicked(self):
        self._enable_all_mode_buttons()
        self.button_replace["state"] = tk.DISABLED
        self.data_mode = DATA_MODE.REPLACE

    def key_press_callback(self, event: tk.Event):
        """ key press event """
        if event.char == "g":
            self.show_bell = not self.show_bell

        if self.show_bell:
            self._show_gauss()
        else:
            self._hide_gauss()

    def freq_change_callback(self, data: AudioData, index_start: int, index_end: int) -> None:
        self.set_data()

    def time_change_callback(self, data: AudioData, index_start: int, index_end: int) -> None:
        self.set_data()

    def reset_view_callback(self, data: AudioData) -> None:
        """ reset the axis to match the datas dimensions"""
        limits = [(min(self._y[i](data, (-1, -1), (-1, -1))), max(self._y[i](data, (-1, -1), (-1, -1)))) for i in range(len(self._y))]
        y_min, y_max = min(limits, key=lambda x: x[0])[0], max(limits, key=lambda x: x[1])[1]

        self.ax.set_xlim((0, data.seconds))
        self.ax.set_ylim((y_min, y_max))
        self.xlim_max = (0, data.seconds)
        self.canvas.draw()

    def _figure_clicked(self, event):
        """ Clicked on the figure """
        if self.command_manager is None:
            raise CommandException("DataView does not know about the CommandManager. Callback can not be started")

        payload = {"data_mode": self.data_mode}
        if self.show_bell:
            payload.update({"bell_halve_width": self.bell_halve_width})
        else:
            payload.update({"bell_halve_width": 0})

        self.command_manager.call("equ_clicked", event, payload=payload)

    def _figure_mouse_moved(self, event: matplotlib.backend_bases.MouseEvent):
        super()._figure_mouse_moved(event)
        if self.show_bell:
            self._show_gauss()

    def _hide_gauss(self):
        """ hide gaussian curve """
        self.bell.set_visible(False)
        self.canvas.draw()

    def _show_gauss(self):
        """ Show the gaussian curve that will be added on click"""
        if self._mouse_pos_data[0] is None or self._mouse_pos_data[1] is None:
            return

        if self.bell_halve_width == 0:
            self.bell.set_data([self._mouse_pos_data[0], self._mouse_pos_data[0]], [0, self._mouse_pos_data[1]])
            self.bell.set_visible(True)
            self.canvas.draw()
        else:
            d_x = self._x[0](self.command_manager.data, (-1, -1), (-1, -1))[1] - self._x[0](self.command_manager.data, (-1,-1), (-1,-1))[0]
            curve = bell_curve(halve_width=self.bell_halve_width) * self._mouse_pos_data[1]
            xx = np.arange(-self.bell_halve_width, self.bell_halve_width + 1, 1, dtype="float") * d_x + self._mouse_pos_data[0]

            # downsample the number of dots
            if xx.shape[0] < 20:
                self.bell.set_data(xx, curve)
            else:
                self.bell.set_data(xx[::xx.shape[0] // 20], curve[::xx.shape[0] // 20])

            self.bell.set_visible(True)
            self.canvas.draw()

    def _figure_scrolled(self, event):
        """ scrolled on window """
        if not self.show_bell:
            super()._figure_scrolled(event)
        else:
            if event.button == "up":
                if self.bell_halve_width > 8:
                    self.bell_halve_width = int(1.2 * self.bell_halve_width)
                else:
                    self.bell_halve_width += 1
            elif self.bell_halve_width > 0:
                if self.bell_halve_width > 8:
                    self.bell_halve_width = int(self.bell_halve_width // 1.2)
                else:
                    self.bell_halve_width -= 1

            self._show_gauss()


class TimeEqualizerFigureDataView(EqualizerFigureDataView):

    def freq_change_callback(self, data: AudioData, index_start: int, index_end: int) -> None:
        pass

    def time_change_callback(self, data: AudioData, index_start: int, index_end: int) -> None:
        self.ax.set_xlim((self.command_manager.data.time_x[data.start_index], self.command_manager.data.time_x[data.end_index - 1]))
        self.set_data()

    def reset_view_callback(self, data: AudioData) -> None:
        """ reset the axis to match the datas dimensions"""
        self.xlim_max = self.ax.get_xlim()

    def _figure_clicked(self, event):
        """ Clicked on the figure """
        if self.command_manager is None:
            raise CommandException("DataView does not know about the CommandManager. Callback can not be started")

        payload = {"data_mode": self.data_mode}
        if self.show_bell:
            payload.update({"bell_halve_width": self.bell_halve_width})
        else:
            payload.update({"bell_halve_width": 0})

        self.command_manager.call("timewindow_clicked", event, payload=payload)


class EqualizerZoomFigureDataView(EqualizerFigureDataView):
    """ Equalizer plot """
    def __init__(self, x_span, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_span = x_span
        self.ax.set_xlim((20, 20 + 2 * self.x_span))
        self.xlim_max = (0, max(self.command_manager.data.freq_x))
        self.set_data()

    def reset_view_callback(self, data: AudioData) -> None:
        """ reset the axis to match the datas dimensions"""
        self.ax.set_xlim((20, 20 + 2 * self.x_span))
        self.xlim_max = (0, max(self.command_manager.data.freq_x))

    def freq_change_callback(self, data: AudioData, index_start: int, index_end: int) -> None:
        #self.ax.set_xlim((data.freq_x[index_start] - self.x_span, data.freq_x[index_end] + self.x_span))
        self.set_data()


class TimeLineFigureDataView(LineFigureDataView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.xlim_max = (0, self.command_manager.data.seconds)

    # TODO: Add downsampler
    def time_change_callback(self, data:AudioData, index_start: int, index_end: int):
        """ Changes the current window interval """
        self.set_data()
        self.set_selection_window((data.start_index / self.command_manager.data.fs, data.end_index / self.command_manager.data.fs))

    def play_timecode_callback(self, data:AudioData, current_frame):
        """ gets called when the audio is played with the current frame of the audio data """
        self.set_selection_line(current_frame / data.fs)

    def reset_view_callback(self, data: AudioData) -> None:
        """ reset the axis to match the datas dimensions"""
        self.ax.set_xlim((0, data.seconds))
        self.ax.set_ylim((min(data.time[::data.time.shape[0] // 400,0]), max(data.time[::data.time.shape[0] // 400,0])))
        self.xlim_max = (0, data.seconds)
        self.canvas.draw()

    def _figure_clicked(self, event):
        """ Mouse click callback"""
        if self.command_manager is None:
            raise CommandException("DataView does not know about the CommandManager. Callback can not be started")
        self.command_manager.call("timeline_clicked", event)

    def key_press_callback(self, event):
        """ key press callback """
        pass
