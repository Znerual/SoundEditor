import copy

import pytest
import numpy as np

from numpy.typing import NDArray
from typing import Any

from SoundEditor.Commands import SetFreq, CommandManager, TimeToFreq
from SoundEditor.DataView import DataView
from SoundEditor.AudioData import AudioData, VersionControlException


def test_set_freq():
    ad = AudioData.from_file("test.wav")
    ad.ft(0,2)
    ad.freq[:, 0] = np.array([0 + 0j, 0 + 0j])
    assert ad._freq_data[0, 0] == 0 + 0j
    assert ad.freq[1, 0] == 0 + 0j
    com = SetFreq(0, 2, np.array([0 + 1j, 1 + 2j]), chanel=0, target=ad)
    com.do()
    assert ad.freq[0, 0] == 0 + 1j
    assert ad.freq[1, 0] == 1 + 2j
    com.undo()
    assert ad._freq_data[0, 0] == 0 + 0j
    assert ad._freq_data[1, 0] == 0 + 0j

    with pytest.raises(VersionControlException):
        com.undo()
    with pytest.raises(VersionControlException):
        com.undo()

    com.redo()
    assert ad.freq[0, 0] == 0 + 1j
    assert ad.freq[1, 0] == 1 + 2j
    com.undo()
    assert ad._freq_data[0, 0] == 0 + 0j
    assert ad._freq_data[1, 0] == 0 + 0j
    com2 = SetFreq(0, 2, np.array([2 + 3j, 3 + 4j]), chanel=0, target=ad)
    com2.do()
    assert ad.freq[0, 0] == 2 + 3j
    assert ad.freq[1, 0] == 3 + 4j
    with pytest.raises(VersionControlException):
        com2.redo()


def test_settimeframe():
    ad = AudioData.from_file("test.wav")
    ad.ft(0, 10)
    assert ad.N == 10
    com = TimeToFreq(20,40, ad)
    com.do()
    assert ad.N == 20


def test_undo_settimeframe():
    ad = AudioData.from_file("test.wav")
    ad.ft(0, 10)
    freq = copy.deepcopy(ad.freq)
    com = TimeToFreq(20, 40, ad)
    com.do()
    assert ad.N == 20
    com.undo()
    assert ad.N == 10
    with pytest.raises(VersionControlException):
        com.undo()
    assert np.all(freq._data == ad.freq._data)
    assert freq._history == ad.freq._history
    assert freq._redo == ad.freq._redo

def test_redo_settimeframe():
    ad = AudioData.from_file("test.wav")
    ad.ft(0, 10)
    com = TimeToFreq(20, 40, ad)
    com.do()
    freq = copy.deepcopy(ad.freq)
    assert ad.N == 20
    com.undo()
    assert ad.N == 10
    com.redo()
    assert ad.N == 20
    for i in range(freq.shape[0]):
        assert freq[i,0] == ad.freq[i,0]

    assert np.all(freq._data == ad.freq._data)
    assert freq._history == ad.freq._history
    assert freq._redo == ad.freq._redo

def test_settimeframe_stack():
    ad = AudioData.from_file("test.wav")
    ad.ft(0, 10)
    freq0 = copy.deepcopy(ad.freq)
    com1 = TimeToFreq(20, 40, ad)
    com2 = TimeToFreq(60, 100, ad)
    com3 = TimeToFreq(200, 300, ad)
    com1.do()
    freq1 = copy.deepcopy(ad.freq)
    com2.do()
    freq2 = copy.deepcopy(ad.freq)
    com3.do()
    freq3 = copy.deepcopy(ad.freq)
    com3.undo()
    assert np.all(freq2._data == ad.freq._data)
    assert freq2._history == ad.freq._history
    assert freq2._redo == ad.freq._redo
    com3.redo()
    assert np.all(freq3._data == ad.freq._data)
    assert freq3._history == ad.freq._history
    assert freq3._redo == ad.freq._redo
    com3.undo()
    com2.undo()
    assert np.all(freq1._data == ad.freq._data)
    assert freq1._history == ad.freq._history
    assert freq1._redo == ad.freq._redo
    com1.undo()
    assert np.all(freq0._data == ad.freq._data)
    assert freq0._history == ad.freq._history
    assert freq0._redo == ad.freq._redo


def test_command_manager():
    ad = AudioData.from_file("test.wav")
    ad.ft(0, 2)
    ad.freq[:, 0] = np.array([0 + 0j, 0 + 0j])
    assert ad._freq_data[0, 0] == 0 + 0j
    assert ad._freq_data[1, 0] == 0 + 0j
    com1 = SetFreq(0, 2, np.array([0 + 1j, 1 + 2j]), chanel=0, target=ad)
    com2 = SetFreq(0, 2, np.array([3 + 4j, 4 + 5j]), chanel=0, target=ad)
    com3 = SetFreq(0, 2, np.array([5 + 6j, 6 + 7j]), chanel=0, target=ad)
    manager = CommandManager(data=ad)
    manager.do(com1)
    assert ad.freq[0, 0] == 0 + 1j
    assert ad.freq[1, 0] == 1 + 2j
    manager.do(com2)
    assert ad.freq[0, 0] == 3 + 4j
    assert ad.freq[1, 0] == 4 + 5j
    manager.do(com1)
    assert ad.freq[0, 0] == 0 + 1j
    assert ad.freq[1, 0] == 1 + 2j
    manager.undo()
    assert ad.freq[0, 0] == 3 + 4j
    assert ad.freq[1, 0] == 4 + 5j
    manager.undo()
    assert ad.freq[0, 0] == 0 + 1j
    assert ad.freq[1, 0] == 1 + 2j
    manager.redo()
    assert ad.freq[0, 0] == 3 + 4j
    assert ad.freq[1, 0] == 4 + 5j
    manager.undo()
    manager.do(com3)
    assert ad.freq[0, 0] == 5 + 6j
    assert ad.freq[1, 0] == 6 + 7j
    with pytest.raises(VersionControlException):
        manager.redo()
    manager.undo()
    manager.undo()
    assert ad.freq[0, 0] == 0 + 0j
    assert ad.freq[1, 0] == 0 + 0j


def test_listener():
    class TestDataView(DataView):
        def __init__(self):
            self.counter = 0

        def freq_change_callback(self, index_start: int, index_end: int, value: NDArray[Any]) -> None:
            self.counter += 1

    dv = TestDataView()
    ad = AudioData.from_file("test.wav")
    ad.ft(0, 2)
    ad.freq[:, 0] = np.array([0 + 0j, 0 + 0j])
    assert ad._freq_data[0, 0] == 0 + 0j
    assert ad._freq_data[1, 0] == 0 + 0j
    com1 = SetFreq(0, 2, np.array([0 + 1j, 1 + 2j]), chanel=0, target=ad, listener=[dv])
    com2 = SetFreq(0, 2, np.array([3 + 4j, 4 + 5j]), chanel=0, target=ad, listener=[dv])
    com3 = SetFreq(0, 2, np.array([5 + 6j, 6 + 7j]), chanel=0, target=ad, listener=[dv])
    manager = CommandManager(data=ad)
    assert dv.counter == 0
    manager.do(com1)
    assert ad.freq[0, 0] == 0 + 1j
    assert ad.freq[1, 0] == 1 + 2j
    assert dv.counter == 1
    manager.undo()
    assert dv.counter == 2
    manager.do(com1)
    assert dv.counter == 3
    manager.undo()
    assert dv.counter == 4
    manager.redo()
    assert dv.counter == 5
    manager.do(com2)
    assert dv.counter == 6
    manager.do(com3)
    assert dv.counter == 7
