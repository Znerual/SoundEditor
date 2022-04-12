from __future__ import annotations

import matplotlib.backend_bases

from abc import abstractmethod, ABCMeta

from .AudioData import AudioData, VersionControlException
from typing import List, Callable, Optional, Dict, Any, Union, Tuple

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

    def call(self, name : str, event : Union[matplotlib.backend_bases.MouseEvent, tkinter.Event], payload: Dict[str: Any] = {}) -> None:
        """ call function used to trigger the callbacks"""
        if not name in self._callbacks:
            raise CommandException(f"No callback registered under the name {name}")

        callers, callback = self._callbacks[name]

        callback(self, event, callers, payload=payload)

    @property
    def data(self) -> AudioData:
        return self._data

    def do(self, command: Command, no_undo=False) -> None:
        command.do()
        if not no_undo:
            self._redo = []
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