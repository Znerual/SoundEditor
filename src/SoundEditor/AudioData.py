import numpy as np
import typing

from typing import Optional, Any
from numpy.typing import NDArray
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
        """ set item indexed by [] notation with keeping track of history """
        self._history.append((key, self[key].copy()))
        self._redo = []
        self._data.__setitem__(key, value)

    def view(self, *args, **kwargs):
        return self._data.view(*args, **kwargs)

    def set_no_undo(self, key, value):
        """ set item indexed by key to value, without keeping track of change. This removes existing history"""
        self._history = []
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
    _time_data : Optional[VersionControlArray] = None
    _freq_data : Optional[VersionControlArray] = None
    _freq_x : Optional[NDArray[np.dtype["float64"]]] = None
    _time_x : Optional[NDArray[np.dtype["float64"]]] = None
    _freq_dom_ind : Optional[NDArray[np.dtype[int]]] = None
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
        instance._time_data, instance._fs = sf.read(filename, dtype='float32')
        instance._seconds = instance._time_data.shape[0] / instance.fs
        instance._time_x = np.arange(0, instance.seconds, 1 / instance.fs)
        return instance

    def _four_trans_seq(self):
        from scipy.fft import fft, fftfreq
        if self._freq_sel_start_ind == -1 or self._freq_sel_end_ind == -1:
            raise AudioDataException(f"Invalid frequency indices {self._freq_sel_start_ind}:{self._freq_sel_end_ind}")
        N = self._freq_sel_end_ind - self._freq_sel_start_ind
        T = 1 / self.fs
        self._freq_x = fftfreq(N, T)[:N // 2]  # positive frequencies

        # create version controlled array and fill without creating history
        self._freq_data = VersionControlArray.empty((N,2), dtype="complex64")
        self._freq_data.set_no_undo((slice(None, None, 1),0), fft(self._time_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 0]))
        self._freq_data.set_no_undo((slice(None, None, 1),1), fft(self._time_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 1]))


    def _find_peaks(self, n_dom=6):
        k_dom_ind = np.zeros((n_dom, 2), dtype="int32")
        yf0_tmp = np.abs(yf[:, 0])[:N // 2]
        yf1_tmp = np.abs(yf[:, 1])[:N // 2]
        # dominant frequencies
        # channel 1
        peaks1, _ = find_peaks(yf0_tmp, height=0, distance=int(40 / (xf[1] - xf[0])), prominence=0.5)
        if peaks1.shape[0] >= n_dom:
            idx = np.argpartition(yf0_tmp[peaks1], -n_dom)[-n_dom:]
            index = idx[np.argsort(-yf0_tmp[peaks1][idx])]
            k_dom_ind[:, 0] = peaks1[index]

        # channel 2
        peaks2, _ = find_peaks(yf1_tmp, height=0, distance=int(40 / (xf[1] - xf[0])), prominence=0.5)
        if peaks2.shape[0] >= n_dom:
            idx = np.argpartition(yf1_tmp[peaks2], -n_dom)[-n_dom:]
            index = idx[np.argsort(-yf1_tmp[peaks2][idx])]
            k_dom_ind[:, 1] = peaks2[index]

    def time(self, start_index=0, end_index=None, chanel=0):
        """ gets a view of the audio data in time domain """
        view = self._time_data.view()
        view.shape = (self._time_data.shape[0] * 2,)
        if end_index is None:
            return view[chanel + 2 * start_index::2]
        else:
            return view[chanel+2*start_index:chanel+2*end_index:2]

    def freq(self, start_index=0, end_index=None, chanel=0):
        """ gets the view of the audio data in frequency domain """
        if not (start_index == self._freq_sel_start_ind and end_index == self._freq_sel_end_ind):
            self._freq_sel_start_ind = start_index
            self._freq_sel_end_ind = self._time_data.shape[0] if end_index is None else end_index
            self._four_trans_seq()

        view = self._freq_data.view()
        view.shape = (self._freq_data.shape[0] * 2,)
        return view[chanel::2]

    def freq_to_time(self):
        """ converts the frequency data to audio data and write it to _time_data"""
        from scipy.fft import ifft
        if self._freq_sel_start_ind == -1 or self._freq_sel_end_ind == -1:
            raise AudioDataException(f"Invalid frequency indices {self._freq_sel_start_ind}:{self._freq_sel_end_ind}")

        self._time_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 0] = ifft(self._freq_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 0])
        self._time_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 1] = ifft(self._freq_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 1])

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
