import warnings
from types import NoneType

import matplotlib.backend_bases

import numpy as np
import sounddevice as sd
import tkinter as tk

from typing import List, Callable, Optional, Dict, Any, Union, Tuple

from .CommandManager import CommandManager
from .Commands import FreqToTime, SetFreq, SetTime, TimeToFreq, Play
from .helper import bell_curve, DATA_MODE
from .DataView import DataView


def equalizer_callback(command_manager: CommandManager, event: matplotlib.backend_bases.MouseEvent, callers: List[DataView], payload: Dict[str, Any]):
    """ Click callback for equalizer plot"""
    # catch clicks outside the figure
    if event.xdata is None:
        return

    # get x grid
    x_data = event.xdata
    delta_x = command_manager.data.freq_x[1] - command_manager.data.freq_x[0]

    # find index where event matches grid
    f_ind = np.where((command_manager.data.freq_x == x_data) | ((x_data - delta_x <= command_manager.data.freq_x) & (command_manager.data.freq_x <= x_data + delta_x)))[0]

    # reduce to one point if mouse between two events
    if f_ind.shape == 3:
        ind = f_ind[1]
    else:
        ind = f_ind[0]

    # change data and catch missing payload
    if not "bell_halve_width" in payload:
        warnings.warn("Missing bell_halve_width parameter in equalizer_callback's payload", RuntimeWarning)
        payload["bell_halve_width"] = 0

    if not "data_mode" in payload:
        warnings.warn("Missing data_mode parameter in equalizer_callback's payload", RuntimeWarning)
        payload["data_mode"] = DATA_MODE.REPLACE

    ind_start = ind - payload["bell_halve_width"]#
    ind_end = ind + payload["bell_halve_width"] + 1

    if payload["data_mode"] == DATA_MODE.REPLACE:
        curve = bell_curve(halve_width=payload["bell_halve_width"]) * event.ydata * command_manager.data.N / 2
    elif payload["data_mode"] == DATA_MODE.ADD:
        curve = command_manager.data.freq[ind_start:ind_end, 0] + bell_curve(halve_width=payload["bell_halve_width"]) * event.ydata * command_manager.data.N / 2
    elif payload["data_mode"] == DATA_MODE.SUBTRACT:
        curve = command_manager.data.freq[ind_start:ind_end, 0] - bell_curve(halve_width=payload["bell_halve_width"]) * event.ydata * command_manager.data.N / 2
    else:
        raise RuntimeError("Invalid choice of data_mode in the equalizer_callback payload")

    # set the frequency value and also update the time values
    command = SetFreq(ind_start, ind_end, curve, chanel=0, target=command_manager.data, listener=callers)
    command2 = FreqToTime(target=command_manager.data, listener=callers)
    command_manager.do(command)
    command_manager.do(command2)


def key_pressed_callback(command_manager: CommandManager, event: tk.Event, callers: List[DataView], payload: Dict[str, Any]):
    """ Key down callback"""
    # Application callbacks
    if event.char == "p":
        command = Play(target=command_manager.data, start_index=command_manager.data.start_index, end_index=command_manager.data.end_index, listener=callers)
        command_manager.do(command, no_undo=True)
        #command = FreqToTime(command_manager.data, listener=callers)
        #command_manager.do(command)
        #sd.play(command_manager.data.time[command_manager.data.start_index:command_manager.data.end_index], command_manager.data.fs)

    # DataView callbacks
    for caller in callers:
        if "key_press_callback" in dir(caller):
            caller.key_press_callback(event)


def update_data_callback(command_manager: CommandManager, event: NoneType, callers: List[DataView], payload: NoneType):
    for caller in callers:
        caller.time_change_callback(command_manager.data, 0, command_manager.data.time.shape[0])
        caller.freq_change_callback(command_manager.data, 0, command_manager.data.time.shape[0])
        caller.reset_view_callback(command_manager.data)


def timewindow_clicked(command_manager: CommandManager, event: matplotlib.backend_bases.MouseEvent, callers: List[DataView], payload: [Dict, Any]):
    """ clicked the time data equalizer view"""

    # catch clicks outside the figure
    if event.xdata is None:
        return

    # get x grid
    x_data = event.xdata
    delta_x = command_manager.data.time_x[1] - command_manager.data.time_x[0]

    # find index where event matches grid
    f_ind = np.where((command_manager.data.time_x == x_data) | ((x_data - delta_x <= command_manager.data.time_x) &
                                                                (command_manager.data.time_x <= x_data + delta_x)))[0]

    # reduce to one point if mouse between two events
    if f_ind.shape == 3:
        ind = f_ind[1]
    else:
        ind = f_ind[0]

    # change data and catch missing payload
    if not "bell_halve_width" in payload:
        warnings.warn("Missing bell_halve_width parameter in equalizer_callback's payload", RuntimeWarning)
        payload["bell_halve_width"] = 0

    if not "data_mode" in payload:
        warnings.warn("Missing data_mode parameter in equalizer_callback's payload", RuntimeWarning)
        payload["data_mode"] = DATA_MODE.REPLACE

    ind_start = ind - payload["bell_halve_width"]
    ind_end = ind + payload["bell_halve_width"] + 1

    if payload["data_mode"] == DATA_MODE.REPLACE:
        curve = bell_curve(halve_width=payload["bell_halve_width"]) * event.ydata
    elif payload["data_mode"] == DATA_MODE.ADD:
        curve = command_manager.data.time[ind_start:ind_end, 0] + bell_curve(
            halve_width=payload["bell_halve_width"]) * event.ydata
    elif payload["data_mode"] == DATA_MODE.SUBTRACT:
        curve = command_manager.data.time[ind_start:ind_end, 0] - bell_curve(
            halve_width=payload["bell_halve_width"]) * event.ydata
    else:
        raise RuntimeError("Invalid choice of data_mode in the equalizer_callback payload")

    command = SetTime(ind_start, ind_end, curve, chanel=0, target=command_manager.data, listener=callers)
    command2 = TimeToFreq(command_manager.data.start_index, command_manager.data.end_index, target=command_manager.data, listener=callers)

    # run the commands to set the time data and update the frequency data as well
    command_manager.do(command)
    command_manager.do(command2)


def timeline_callback(command_manager: CommandManager, event: matplotlib.backend_bases.MouseEvent, callers: List[DataView], payload: [Dict, Any]):
    """ Clicked on timeline plot """
    index_time = int(event.xdata * command_manager.data.fs)
    old_start = command_manager.data.start_index
    old_end = command_manager.data.end_index
    span = old_end - old_start

    if event.button == matplotlib.backend_bases.MouseButton.LEFT:
        if index_time < old_end:
            command = TimeToFreq(index_time, old_end, command_manager.data, listener=callers)
        else:
            command = TimeToFreq(old_end, index_time, command_manager.data, listener=callers)

    elif event.button == matplotlib.backend_bases.MouseButton.RIGHT:
        if index_time > old_start:
            command = TimeToFreq(old_start, index_time, command_manager.data, listener=callers)
        else:
            command = TimeToFreq(index_time, old_start, command_manager.data, listener=callers)

    elif event.button == matplotlib.backend_bases.MouseButton.MIDDLE:
        if index_time + span >= command_manager.data.time.shape[0]:
            command = TimeToFreq(index_time, command_manager.data.time.shape[0], command_manager.data, listener=callers)
        else:
            command = TimeToFreq(index_time, index_time + span, command_manager.data, listener=callers)

    command_manager.do(command)
