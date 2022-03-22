import numpy as np
import typing

from typing import Optional, Any
#from numpy.typing import NDArray
from dataclasses import dataclass, field

class VersionControlException(IndexError):
    pass

class AudioDataException(Exception):
    pass

class VersionControlArray:
    """ Creates an array that saves changes to it"""

    def __init__(self, *args, **kwargs):
        self._history = []
        self._redo = []
        self._data = np.array(*args, **kwargs)

    @classmethod
    def empty(cls, shape, dtype=None, order='C'):
        instance = cls.__new__(cls)
        instance._history = []
        instance._data = np.empty(shape, dtype=dtype, order=order)
        return instance

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._history.append((key, self[key].copy()))
        self._redo = []
        self._data.__setitem__(key, value)

    def undo(self):
        """ undo the last change of the array. Throws an VersionControlException when no undo is available"""
        if len(self._history) == 0:
            raise VersionControlException("Can't undo, history is empty")

        last_action = self._history.pop()
        self._redo.append((last_action[0], self._data[last_action[0]].copy()))
        self._data[last_action[0]] = last_action[1]

    def redo(self):
        """ redo the last change of the array. Throws an VersionControlException when no redo is available"""
        if len(self._redo) == 0:
            raise VersionControlException("Can't redo, redo steps are empty")
        last_action = self._redo.pop()
        self._history.append((last_action[0], self._data[last_action[0]].copy()))
        self._data[last_action[0]] = last_action[1]

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def history(self):
        return self._history

@dataclass()
class AudioData:
    _audio_time_data : Optional[VersionControlArray] = None
    _audio_freq_data : Optional[VersionControlArray] = None
    _fs : Optional[int] = -1
    _seconds : Optional[float] = -1
    _freq_sel_start_ind : int = -1
    _freq_sel_end_ind : int = -1

    @classmethod
    def from_file(cls, filename : str) -> "AudioData":
        """ Initializer from a file"""
        import soundfile as sf

        # create new instance and call parent initializer
        instance = cls.__new__(cls)
        super(cls, instance).__init__()

        # fill audio data with loaded information
        instance._audio_time_data, instance._fs = sf.read(filename, dtype='float32')
        instance._seconds = instance._audio_time_data.shape[0] / instance.fs

        return instance

    @property
    def fs(self):
        """ fs property"""
        if self._fs == -1:
            raise AudioDataException("No fs information available. Check if AudioData was loaded correctly.")
        return self._fs

    @property
    def seconds(self):
        """ seconds property"""
        if self._seconds == -1:
            raise AudioDataException("No seconds information available. Check if AudioData was loaded correctly.")
        return self._seconds
