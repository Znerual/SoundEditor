from __future__ import annotations

import copy

from typing import List, Callable, Optional, Dict, Any, Union, Tuple


import numpy as np

from .AudioData import AudioData, VersionControlException, VersionControlArray
from .CommandManager import Command, CommandException, CommandManager
from numpy.typing import NDArray


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
        self.history_freq = (self.target.start_index, self.target.end_index, copy.deepcopy(self.target.freq))
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
            listener.time_change_callback(self.target.start_index, self.target.end_index)

    def redo(self) -> None:
        """ redos the change of time frame """
        if self.redo_freq is None:
            VersionControlException("Can't redo command because of empty history.")
        self.history_freq = (self.target.start_index, self.end_index,copy.deepcopy(self.target.freq))
        self.target._freq_sel_start_ind, self.target._freq_sel_end_ind, self.target._freq_data = self.redo_freq
        self.redo_freq = None
        for listener in self.listener:
            listener.time_change_callback(self.start_index, self.end_index)


class FreqToTime(Command):
    """ converts the current frequency data back to time data """
    def __init__(self, target: AudioData, listener: List["DataView"] = []):
        self.target = target
        self.listener = listener

    def do(self) -> None:
        self.target.ift()
        for listener in self.listener:
            listener.time_change_callback(self.target.start_index, self.target.end_index)

    def undo(self) -> None:
        self.target.time_undo()
        self.target.time_undo()

    def redo(self) -> None:
        self.target.time_redo()
        self.target.time_redo()


