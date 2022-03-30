from __future__ import annotations

import tkinter
import copy
from abc import abstractmethod, ABCMeta
from typing import List, Callable, Optional, Dict, Any, Union, Tuple

import matplotlib.backend_bases
import numpy as np

from .AudioData import AudioData, VersionControlException

from numpy.typing import NDArray


class CommandException(Exception):
    pass


class Command(metaclass=ABCMeta):
    """ Abstract Base Class for commands"""

    @abstractmethod
    def do(self) -> None:
        pass

    @abstractmethod
    def undo(self) -> None:
        pass

    @abstractmethod
    def redo(self) -> None:
        pass


class CommandManager:
    """ Handles Command Objects """
    def __init__(self, data: AudioData):
        self._data: AudioData = data
        self._history: List[Command] = []
        self._redo: List[Command] = []
        self._callbacks: Dict[str,
                Tuple[DataView, Callable[[CommandManager, Union[matplotlib.backend_bases.MouseEvent, tkinter.Event], DataView], None]]] = {}

    def register_callback(self, name : str, callback : Callable[[CommandManager, Union[matplotlib.backend_bases.MouseEvent, tkinter.Event], DataView], None], callers: List[DataView]) -> None:
        """ registers a callback under the name name which can be called"""
        if name in self._callbacks:
            raise CommandException(f"Can't register callback. {name} already exists!")

        self._callbacks[name] = (callers, callback)

    def call(self, name : str, event : Union[matplotlib.backend_bases.MouseEvent, tkinter.Event]) -> None:
        """ call function used to trigger the callbacks"""
        if not name in self._callbacks:
            raise CommandException(f"No callback registered under the name {name}")

        callers, callback = self._callbacks[name]

        callback(self, event, callers)

    @property
    def data(self) -> AudioData:
        return self._data

    def do(self, command: Command) -> None:
        self._redo = []
        command.do()
        self._history.append(command)

    def undo(self) -> None:
        if len(self._history) == 0:
            raise VersionControlException("Can't undo, history is empty")
        last_action = self._history.pop()
        last_action.undo()
        self._redo.append(last_action)

    def redo(self) -> None:
        if len(self._redo) == 0:
            raise VersionControlException("Can't redo, history is empty")
        last_action = self._redo.pop()
        last_action.redo()
        self._history.append(last_action)


class SetFreq(Command):
    """ Set frequency values """

    def __init__(self, start_index: int, end_index: int, value: NDArray[np.csingle], chanel: int, target: AudioData, listener: List["DataView"] = []):
        self.start_index = start_index
        self.end_index = end_index
        self.value = value.copy()
        self.chanel = chanel
        self.target = target
        self.listener = listener

    def do(self) -> None:
        """ Change frequency and notify listeners """
        self.target.freq[self.start_index:self.end_index, self.chanel] = self.value
        for listener in self.listener:
            listener.freq_change_callback(self.target, self.start_index, self.end_index)

    def undo(self) -> None:
        """ undo a command. Check if nothing changed data inbetween"""
        if np.all(self.target.freq[:, self.chanel] != self.value):
            raise VersionControlException("Can't undo command because data was changed.")

        self.target.freq_undo()

        for listener in self.listener:
            listener.freq_change_callback(self.target, self.start_index, self.end_index)

    def redo(self) -> None:
        """ redos the command"""
        self.target.freq_redo()

        for listener in self.listener:
            listener.freq_change_callback(self.target, self.start_index, self.end_index)


class SetTimeFrame(Command):
    """ Changes the current selection of a time frame which is Fourier transformed and shown """

    def __init__(self, start_index: int, end_index: int, target: AudioData,
                 listener: List["DataView"] = []):
        self.start_index = start_index
        self.end_index = end_index
        self.target = target
        self.listener = listener
        self.history_freq: Optional[Tuple[int, int, VersionControlArray]] = None
        self.redo_freq: Optional[Tuple[int, int, VersionControlArray]] = None

    def do(self) -> None:
        """ changes the current time frame and does the fourier transformation """
        self.redo_freq = None
        self.history_freq = (self.target._freq_sel_start_ind, self.target._freq_sel_end_ind, copy.deepcopy(self.target.freq))
        self.target.ft(self.start_index, self.end_index)
        for listener in self.listener:
            listener.time_change_callback(self.start_index, self.end_index)

    def undo(self) -> None:
        """ undos the change of time frame"""
        if self.history_freq is None:
            raise VersionControlException("Can't undo command because of empty history.")
        self.redo_freq = (self.start_index, self.end_index, copy.deepcopy(self.target.freq))
        self.target._freq_sel_start_ind, self.target._freq_sel_end_ind, self.target._freq_data = self.history_freq
        self.history_freq = None
        for listener in self.listener:
            listener.time_change_callback(self.target._freq_sel_start_ind, self.target._freq_sel_end_ind)

    def redo(self) -> None:
        """ redos the change of time frame """
        if self.redo_freq is None:
            VersionControlException("Can't redo command because of empty history.")
        self.history_freq = (self.target._freq_sel_start_ind, self.target._freq_sel_end_ind,copy.deepcopy(self.target.freq))
        self.target._freq_sel_start_ind, self.target._freq_sel_end_ind, self.target._freq_data = self.redo_freq
        self.redo_freq = None
        for listener in self.listener:
            listener.time_change_callback(self.start_index, self.end_index)


def equalizer_callback(command_manager: CommandManager, event: matplotlib.backend_bases.MouseEvent, callers: List[DataView]):
    """ Click callback for equalizer plot"""
    print(event)
    x_data = int(round(event.xdata))
    delta_x = command_manager.data.freq_x[1] - command_manager.data.freq_x[0]
    #print(f"d_X: {delta_x}, x data: {x_data}, freq_x: {command_manager.data.freq_x}")
    f_ind = np.where((command_manager.data.freq_x == x_data) | ((x_data - delta_x <= command_manager.data.freq_x) & (command_manager.data.freq_x <= x_data + delta_x)))[0]

    if f_ind.shape == 3:
        ind = f_ind[1]
    else:
        ind = f_ind[0]

    command = SetFreq(ind, ind+3, event.ydata * command_manager.data.N / 2, chanel=0, target=command_manager.data, listener=callers)
    command_manager.do(command)


def timeline_callback(command_manager: CommandManager, event: matplotlib.backend_bases.MouseEvent, callers: List[DataView]):
    """ Clicked on timeline plot """
    index_time = int(event.xdata * command_manager.data.fs)
    old_start = command_manager.data.start_index
    old_end = command_manager.data.end_index
    span = old_end - old_start

    if event.button == matplotlib.backend_bases.MouseButton.LEFT:
        if index_time < old_end:
            command = SetTimeFrame(index_time, old_end, command_manager.data, listener=callers)
        else:
            command = SetTimeFrame(old_end, index_time, command_manager.data, listener=callers)

    elif event.button == matplotlib.backend_bases.MouseButton.RIGHT:
        if index_time > old_start:
            command = SetTimeFrame(old_start, index_time, command_manager.data, listener=callers)
        else:
            command = SetTimeFrame(index_time, old_start, command_manager.data, listener=callers)

    elif event.button == matplotlib.backend_bases.MouseButton.MIDDLE:
        if index_time + span >= command_manager.data.time.shape[0]:
            command = SetTimeFrame(index_time, command_manager.data.time.shape[0], command_manager.data, listener=callers)
        else:
            command = SetTimeFrame(index_time, index_time + span, command_manager.data, listener=callers)

    command_manager.do(command)

