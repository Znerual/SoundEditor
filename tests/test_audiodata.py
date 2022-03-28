import numpy as np
import pytest

from SoundEditor.AudioData import AudioData, AudioDataException, VersionControlException


def test_init():
    ad = AudioData()
    with pytest.raises(AudioDataException):
        ad.fs


def test_readonly_attr():
    ad = AudioData()
    with pytest.raises(AttributeError):
        ad.fs = 2
    with pytest.raises(AttributeError):
        ad.seconds = 4.2


def test_load_file():
    ad = AudioData.from_file("test.wav")
    assert ad.fs == 44100
    assert ad.seconds == 5
    assert ad._time_data.shape[0] / ad.fs == ad.seconds
    with pytest.raises(AttributeError):
        ad.seconds = 4.2


def test_four_trans_seq():
    ad = AudioData.from_file("test.wav")
    with pytest.raises(AudioDataException):
        ad._four_trans_seq()
    ad._freq_sel_start_ind = 0
    ad._freq_sel_end_ind = 10
    ad._four_trans_seq()
    assert ad._freq_data.shape == (10, 2)

    ad._time_data[0:10, 0] = np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / 10))
    ad._four_trans_seq()
    assert abs(-2 / 10 * np.imag(ad._freq_data[0, 0])) <= 1e-6
    assert (-2 / 10 * np.imag(ad._freq_data[1, 0])) == 1
    for i in range(2, 10):
        assert (-2 / 10 * np.imag(ad._freq_data[i, 0])) <= 1e-6
    for i in range(10):
        assert (-2 / 10 * np.real(ad._freq_data[3, 0])) <= 1e-6


def test_find_peaks():
    ad = AudioData.from_file("test.wav")
    freq = ad.freq_no_undo(0, 56)
    freq[:] = 0.0

    peaks = ad.find_peaks()
    assert peaks[0] == 0.0
    assert np.all(peaks == np.zeros(6, ))

    freq[::4] = 12
    assert pytest.approx(np.real(ad._freq_data[0, 0])) == 12
    peaks = ad.find_peaks()
    print(peaks)
    assert peaks[0] == 4
    assert peaks[1] == 8


def test_time():
    ad = AudioData.from_file("test.wav")
    time = ad.time_no_undo(0, 10)
    time[2] = -1
    assert ad._time_data[2, 0] == -1
    time[1] = -2
    assert ad._time_data[1, 0] == -2
    time[0] = -3
    assert ad._time_data[0, 0] == -3

    time = ad.time_no_undo(0, 10, 1)
    assert time[2] == 0
    time[2] = -4
    assert ad._time_data[2, 1] == -4

    time = ad.time_no_undo(10, 15, 0)
    time[0] = -5
    assert ad._time_data[10, 0] == -5
    time[1] = -6
    assert ad._time_data[11, 0] == -6

    time = ad.time_no_undo(10, 15, 1)
    time[0] = -7
    assert ad._time_data[10, 1] == -7
    time[1] = -8
    assert ad._time_data[11, 1] == -8

    time = ad.time_no_undo(100, chanel=1)
    time[4] = -9
    assert ad._time_data[104, 1] == -9
    time[5] = -10
    assert ad._time_data[105, 1] == -10
    time[-1] = -10.5
    assert ad._time_data[-1, 1] == -10.5

    time = ad.time_no_undo(100)
    time[4] = -11
    assert ad._time_data[104, 0] == -11
    time[5] = -12
    assert ad._time_data[105, 0] == -12
    time[-1] = -13
    assert ad._time_data[-1, 0] == -13


def test_freq():
    ad = AudioData.from_file("test.wav")
    freq = ad.freq_no_undo(0, 10)
    freq[2] = -1
    assert ad._freq_data[2, 0] == -1
    freq[1] = -2
    assert ad._freq_data[1, 0] == -2
    freq[0] = -3
    assert ad._freq_data[0, 0] == -3

    freq = ad.freq_no_undo(0, 10, 1)
    freq[2] = -4
    assert ad._freq_data[2, 1] == -4

    freq = ad.freq_no_undo(10, 15, 0)
    assert ad._freq_sel_start_ind == 10
    assert ad._freq_sel_end_ind == 15
    assert ad._freq_data.shape == (5, 2)
    assert freq.shape == (5,)
    freq[0] = -5
    assert ad._freq_data[0, 0] == -5
    freq[1] = -6
    assert ad._freq_data[1, 0] == -6

    freq = ad.freq_no_undo(10, 15, 1)
    freq[0] = -7
    assert ad._freq_data[0, 1] == -7
    freq[1] = -8
    assert ad._freq_data[1, 1] == -8

    freq = ad.freq_no_undo(100, chanel=1)
    freq[4] = -9
    assert ad._freq_data[4, 1] == -9
    freq[5] = -10
    assert ad._freq_data[5, 1] == -10
    freq[-1] = -10.5
    assert ad._freq_data[-1, 1] == -10.5

    freq = ad.freq_no_undo(100)
    freq[4] = -11
    assert ad._freq_data[4, 0] == -11
    ad.freq[5,0] = -12
    assert ad._freq_data[5, 0] == -12
    freq[-1] = -13
    assert ad._freq_data[-1, 0] == -13

def test_N():
    ad: AudioData = AudioData.from_file("test.wav")
    ad.ft(0, 1024)
    assert ad.N == 1024
    assert ad.freq.shape[0] == ad.N

def test_time_undo():
    ad: AudioData = AudioData.from_file("test.wav")
    # time = ad.time[0:120, 0]
    ad.time[30, 0] = -13
    assert ad._time_data[30, 0] == -13
    ad.time[30, 0] = -26
    assert ad._time_data[30, 0] == -26
    ad.time_undo()
    assert ad._time_data[30, 0] == -13
    ad.time_redo()
    assert ad._time_data[30, 0] == -26


def test_time_redo():
    ad: AudioData = AudioData.from_file("test.wav")
    # time = ad.time[0:120, 0]
    ad.time[30, 0] = -13
    assert ad._time_data[30, 0] == -13
    ad.time[30, 0] = -26
    assert ad._time_data[30, 0] == -26
    ad.time_undo()
    assert ad._time_data[30, 0] == -13
    ad.time[30, 0] = -40
    with pytest.raises(VersionControlException):
        ad.time_redo()
    assert ad._time_data[30, 0] == -40


def test_freq_undo():
    ad: AudioData = AudioData.from_file("test.wav")
    ad.ft(0, 120)
    ad.freq[30, 0] = -13
    assert ad._freq_data[30, 0] == -13
    ad.freq[30, 0] = -26
    assert ad._freq_data[30, 0] == -26
    ad.freq_undo()
    assert ad._freq_data[30, 0] == -13
    ad.freq_redo()
    assert ad._freq_data[30, 0] == -26


def test_freq_redo():
    ad: AudioData = AudioData.from_file("test.wav")
    ad.ft(0, 120)
    ad.freq[30, 0] = -13
    assert ad._freq_data[30, 0] == -13
    ad.freq[30, 0] = -26
    assert ad._freq_data[30, 0] == -26
    ad.freq_undo()
    assert ad._freq_data[30, 0] == -13
    ad.freq[30, 0] = -40
    with pytest.raises(VersionControlException):
        ad.freq_redo()
    assert ad._freq_data[30, 0] == -40
