from __future__ import annotations

import subprocess
import pathlib

import numpy as np

from typing import Optional, Tuple, Union, Dict, List

import pydub
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


def ffmpeg_formats() -> List[Tuple[str, str]]:
    """
    finds available formats
    """

    formats = []
    process = subprocess.Popen(['ffmpeg', '-formats'], shell=True, stdout=subprocess.PIPE)

    # skip header information
    for _ in range(4):
        process.stdout.readline()

    while line := process.stdout.readline().decode('ascii'):
        parts = line[4:].rstrip("\r\n").split(" ")
        formats.append((" ".join(filter(lambda x: x != "",parts[1:])), parts[0]))

    return sorted(formats)


def pydub_to_np(audio: pydub.AudioSegment) -> (NDArray[np.float32], int):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate, audio.sample_width, audio.channels


def np_to_pydub(array: NDArray[np.float32], fs: int, sample_width: int = 4, channels: int = 2) -> pydub.AudioSegment:
    """
    Converts a given numpy array into a pydub AudioSegment
    """
    if sample_width == 4:
        int_array = np.int32(array.reshape(-1, ) * 2 ** 31)
    elif sample_width == 2:
        int_array = np.int16(array.reshape(-1, ) * 2 ** 15)
    elif sample_width == 8:
        int_array = np.int64(array.reshape(-1, ) * 2 ** 63)
    else:
        raise AudioDataException(f"Invalid sample width {sample_width} for the conversion to a numpy array")

    audio = pydub.AudioSegment(int_array.tobytes(), frame_rate=fs, sample_width=sample_width, channels=channels)

    return audio


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
    _a_coef: float = 0.25
    _channels: int = -1
    _sample_width: int = -1

    @classmethod
    def from_file(cls, filename: str) -> AudioData:
        """ Initializer from a file"""
        # import soundfile as sf

        # create new instance and call parent initializer
        instance: AudioData = cls.__new__(cls)
        super(cls, instance).__init__()

        # fill audio data with loaded information
        sound = pydub.AudioSegment.from_file(filename)
        raw_time_data, instance._fs, instance._sample_width, instance._channels = pydub_to_np(
            sound)  # sf.read(filename, dtype=np.double)
        instance._time_data = VersionControlArray(raw_time_data)
        instance._seconds = instance._time_data.shape[0] / instance.fs
        instance._time_x = np.arange(0, instance.seconds, 1 / instance.fs, dtype=np.float32)

        # do the initial fourier transform
        if instance.seconds > 10:
            instance.ft(0, 10 * instance.fs)
        else:
            instance.ft(0, instance.time.shape[0])

        return instance

    def save_to_file(self, filename: Union[pathlib.Path, str], file_format: str = "wav") -> None:
        """
        saves the curent audio data to a file
        All file endings supported by pydub (== all endings supported by ffmpeg)
        """
        audio = np_to_pydub(self.time.view(), fs=self.fs, sample_width=self._sample_width, channels=self._channels)
        audio.export(filename, format=file_format)

    def _four_trans_seq(self) -> None:
        # TODO: convolute with decaying funciton to avoid boundary effects
        # ft(1/(sqrt(2)* a) exp((-x^2)/(4(a^2)))) == exp(-f^2*a^2)
        """ transform a part of the audio data from time domain to frequency domain """
        from scipy.fft import fft, fftfreq
        if self._freq_sel_start_ind == -1 or self._freq_sel_end_ind == -1:
            raise AudioDataException(f"Invalid frequency indices {self._freq_sel_start_ind}:{self._freq_sel_end_ind}")
        N = self._freq_sel_end_ind - self._freq_sel_start_ind
        T = 1 / self.fs
        self._freq_x = fftfreq(N, T)[:N // 2]  # positive frequencies

        # decay folding function for suppression gipp's phenomenon
        damp_func = self.damping

        # create version controlled array and fill without creating history
        self._freq_data = VersionControlArray.empty((N, 2), dtype="complex64")
        self._freq_data.set_no_undo((slice(None, None, 1), 0),
                                    fft(damp_func * self._time_data[self._freq_sel_start_ind:self._freq_sel_end_ind,
                                                    0]))
        self._freq_data.set_no_undo((slice(None, None, 1), 1),
                                    fft(damp_func * self._time_data[self._freq_sel_start_ind:self._freq_sel_end_ind,
                                                    1]))

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
    def start_index(self) -> int:
        return self._freq_sel_start_ind

    @property
    def end_index(self) -> int:
        return self._freq_sel_end_ind

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

    @property
    def damping(self) -> NDArray[np.float32]:
        """ returns the damping function which is used on the time data before Fourier transformation"""
        a = self._a_coef
        x = np.linspace(0, 1, num=self.N)
        return np.exp((-(x - 0.5) ** 2) / (2 * a ** 2))  # self._time_x[0:self.N] - self._time_x[self.N//2]

    @property
    def inv_damping(self) -> NDArray[np.float32]:
        """ returns the inverse damping function which is used on the time data after the inverse FT"""
        a = self._a_coef
        x = np.linspace(0, 1, num=self.N)
        return np.exp(((x - 0.5) ** 2) / (2 * a ** 2))

    def ft(self, start_index: int, end_index: Optional[int] = None):
        """ Fourier transforms the given audio segment """
        if not (start_index == self._freq_sel_start_ind and end_index == self._freq_sel_end_ind):
            self._freq_sel_start_ind = start_index
            self._freq_sel_end_ind = self._time_data.shape[0] if end_index is None else end_index
            self._four_trans_seq()

    def ift(self):
        """ Inverse Fourier transform of the frequency signal back to audio signal """
        from scipy.fft import ifft

        inv_damp_func = self.inv_damping
        self._time_data[self.start_index:self.end_index, 0] = np.real(inv_damp_func * ifft(self.freq[:, 0]))
        self._time_data[self.start_index:self.end_index, 1] = np.real(inv_damp_func * ifft(self.freq[:, 1]))

    @property
    def freq(self) -> VersionControlArray:
        """ Get view of frequency data Version Control Array and generate Fourier Transform if not existing """
        return self._freq_data

    def norm_freq(self, chanel: int = 0) -> NDArray[np.csingle]:
        """ Returns a copy of the normalized frequencies"""

        return self.freq[:self._freq_data.shape[0] // 2, chanel] * 2.0 / self.N

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
