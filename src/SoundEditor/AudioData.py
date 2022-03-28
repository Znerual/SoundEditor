from __future__ import annotations

import numpy as np

from typing import Optional
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

    def real(self):
        """ return a copy of the real part"""
        return np.real(self._data).copy()

    def imag(self):
        """ returns a copy of the imaginary part"""
        return np.image(self._data).copy()

    def abs(self):
        """ returns a copy of the array with only positive sign"""
        return np.abs(self._data).copy()

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        """ set item indexed by [] notation with keeping track of history """
        self._history.append((key, self[key].copy()))
        self._redo = []
        self._data.__setitem__(key, value)

    def view(self, *args, **kwargs) -> NDArray:
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


@dataclass
class AudioData:
    _time_data: VersionControlArray = field(init=False)
    _freq_data: VersionControlArray = field(init=False)
    _freq_x: NDArray[np.double] = field(init=False)
    _time_x: NDArray[np.double] = field(init=False)
    _fs: int = -1
    _seconds: float = -1
    _freq_sel_start_ind: int = -1
    _freq_sel_end_ind: int = -1

    @classmethod
    def from_file(cls, filename: str) -> AudioData:
        """ Initializer from a file"""
        import soundfile as sf

        # create new instance and call parent initializer
        instance: AudioData = cls.__new__(cls)
        super(cls, instance).__init__()

        # fill audio data with loaded information
        raw_time_data, instance._fs = sf.read(filename, dtype=np.double)
        instance._time_data = VersionControlArray(raw_time_data)
        instance._seconds = instance._time_data.shape[0] / instance.fs
        instance._time_x = np.arange(0, instance.seconds, 1 / instance.fs, dtype=np.float32)

        return instance

    def _four_trans_seq(self) -> None:
        """ transform a part of the audio data from time domain to frequency domain """
        from scipy.fft import fft, fftfreq
        if self._freq_sel_start_ind == -1 or self._freq_sel_end_ind == -1:
            raise AudioDataException(f"Invalid frequency indices {self._freq_sel_start_ind}:{self._freq_sel_end_ind}")
        N = self._freq_sel_end_ind - self._freq_sel_start_ind
        T = 1 / self.fs
        self._freq_x = fftfreq(N, T)[:N // 2]  # positive frequencies

        # create version controlled array and fill without creating history
        self._freq_data = VersionControlArray.empty((N, 2), dtype="complex64")
        self._freq_data.set_no_undo((slice(None, None, 1), 0),
                                    fft(self._time_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 0]))
        self._freq_data.set_no_undo((slice(None, None, 1), 1),
                                    fft(self._time_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 1]))

    def find_peaks(self, n_dom: int = 6, chanel: int = 0) -> NDArray[np.int32]:
        """ find peaks in the frequency spectrum """
        from scipy.signal import find_peaks

        k_dom_ind: NDArray[np.int32] = np.zeros((n_dom,), dtype="int32")
        delta_freq: float = self._freq_x[1] - self._freq_x[0]
        yf_tmp = np.abs(self._freq_data[:, chanel])[:self._freq_data.shape[0] // 2]

        # dominant frequencies
        peaks, _ = find_peaks(yf_tmp, height=0, distance=int(40 / delta_freq) + 1, prominence=0.5)

        if peaks.shape[0] >= n_dom:
            idx = np.argpartition(yf_tmp[peaks], -n_dom)[-n_dom:]
            index = idx[np.argsort(-yf_tmp[peaks][idx])]
            k_dom_ind[:] = peaks[index]

        return k_dom_ind

    @property
    def time(self) -> VersionControlArray:
        """ Get time data Version Control Array """
        return self._time_data

    @property
    def time_x(self) -> NDArray[np.float32]:
        return self._time_x

    @property
    def N(self) -> int:
        return self._freq_sel_end_ind - self._freq_sel_start_ind

    def ft(self, start_index: int, end_index: Optional[int] = None):
        """ Fourier transforms the given audio segment """
        if not (start_index == self._freq_sel_start_ind and end_index == self._freq_sel_end_ind):
            self._freq_sel_start_ind = start_index
            self._freq_sel_end_ind = self._time_data.shape[0] if end_index is None else end_index
            self._four_trans_seq()

    @property
    def freq(self) -> VersionControlArray:
        """ Get view of frequency data Version Control Array and generate Fourier Transform if not existing """
        return self._freq_data

    def norm_freq(self,  chanel: int = 0) -> NDArray[np.csingle]:
        """ Returns a copy of the normalized frequencies"""

        return self.freq[:self._freq_data.shape[0]//2, chanel] * 2.0 / self.N

    @property
    def freq_x(self) -> NDArray[np.float32]:
        if "_freq_x" not in dir(self) or self._freq_x is None:
            raise AudioDataException("No frequency data available! Need a call to freq() or norm_freq() "
                                     "before accessing freq_x data.")
        return self._freq_x

    def time_no_undo(self, start_index: int = 0, end_index: Optional[int] = None, chanel: int = 0) -> NDArray[
        np.float32]:
        """ gets a view of the audio data in time domain """
        view: NDArray[np.float32] = self._time_data.view()
        view.shape = (self._time_data.shape[0] * 2,)
        if end_index is None:
            return view[chanel + 2 * start_index::2]
        else:
            return view[chanel + 2 * start_index:chanel + 2 * end_index:2]

    def freq_no_undo(self, start_index: int = 0, end_index: Optional[int] = None, chanel: int = 0) -> NDArray[
        np.csingle]:
        """ gets the view of the audio data in frequency domain """
        if not (start_index == self._freq_sel_start_ind and end_index == self._freq_sel_end_ind):
            self._freq_sel_start_ind = start_index
            self._freq_sel_end_ind = self._time_data.shape[0] if end_index is None else end_index
            self._four_trans_seq()

        view: NDArray[np.csingle] = self._freq_data.view()
        view.shape = (self._freq_data.shape[0] * 2,)
        return view[chanel::2]

    def freq_undo(self) -> None:
        """ Undo changes made to the frequency data"""
        self._freq_data.undo()

    def time_undo(self) -> None:
        """ Undo changes made to the time data"""
        self._time_data.undo()

    def freq_redo(self) -> None:
        """ Redo changes to frequency data"""
        self._freq_data.redo()

    def time_redo(self) -> None:
        """ Redo changes to time data """
        self._time_data.redo()

    def freq_to_time(self) -> None:
        """ converts the frequency data to audio data and write it to _time_data"""
        from scipy.fft import ifft
        if self._freq_sel_start_ind == -1 or self._freq_sel_end_ind == -1:
            raise AudioDataException(f"Invalid frequency indices {self._freq_sel_start_ind}:{self._freq_sel_end_ind}")

        self._time_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 0] = ifft(
            self._freq_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 0])
        self._time_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 1] = ifft(
            self._freq_data[self._freq_sel_start_ind:self._freq_sel_end_ind, 1])

    @property
    def fs(self) -> int:
        """ fs property"""
        if self._fs == -1:
            raise AudioDataException("No fs information available. Check if AudioData was loaded correctly.")
        return self._fs

    @property
    def seconds(self) -> float:
        """ seconds property"""
        if self._seconds == -1:
            raise AudioDataException("No seconds information available. Check if AudioData was loaded correctly.")
        return self._seconds
