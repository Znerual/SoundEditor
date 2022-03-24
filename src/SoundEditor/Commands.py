from abc import abstractmethod, ABCMeta
from typing import List, Callable, Optional

import numpy as np

from .AudioData import AudioData
from .DataView import DataView
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
    def __init__(self):
        self._history: List[Command] = []
        self._redo: List[Command] = []

    def do(self, command: Command) -> None:
        self._redo = []
        command.do()
        self._history.append(command)

    def undo(self) -> None:
        if len(self._history) == 0:
            raise CommandException("Can't undo, history is empty")
        last_action = self._history.pop()
        last_action.undo()
        self._redo.append(last_action)

    def redo(self) -> None:
        if len(self._redo) == 0:
            raise CommandException("Can't redo, history is empty")
        last_action = self._redo.pop()
        last_action.redo()
        self._history.append(last_action)


class SetFreq(Command):
    """ Set frequency values """

    def __init__(self, start_index: int, end_index: int, value: NDArray[np.csingle], chanel: int, target: AudioData, listener: List[DataView] = []):
        self.start_index = start_index
        self.end_index = end_index
        self.value = value.copy()
        self.chanel = chanel
        self.target = target
        self.listener = listener

    def do(self) -> None:
        """ Change frequency and notify listeners """
        self.target.freq(start_index=self.start_index, end_index=self.end_index)[:, self.chanel] = self.value

        for listener in self.listener:
            listener.freq_change_callback(self.start_index, self.end_index)

    def undo(self) -> None:
        self.target.freq_undo()

        for listener in self.listener:
            listener.freq_change_callback(self.start_index, self.end_index)

    def redo(self) -> None:
        self.target.freq_redo()

        for listener in self.listener:
            listener.freq_change_callback(self.start_index, self.end_index)
