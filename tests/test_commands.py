import pytest
import numpy as np

from SoundEditor.Commands import SetFreq, CommandManager
from SoundEditor.DataView import DataView
from SoundEditor.AudioData import AudioData, VersionControlException


def test_set_freq():
    ad = AudioData.from_file("test.wav")
    ad.freq(0, 2)[:, 0] = np.array([0 + 0j, 0 + 0j])
    assert ad._freq_data[0, 0] == 0 + 0j
    assert ad._freq_data[1, 0] == 0 + 0j
    com = SetFreq(0, 2, np.array([0 + 1j, 1 + 2j]), chanel=0, target=ad)
    com.do()
    assert ad.freq(0, 2)[0, 0] == 0 + 1j
    assert ad.freq(0, 2)[1, 0] == 1 + 2j
    com.undo()
    assert ad._freq_data[0, 0] == 0 + 0j
    assert ad._freq_data[1, 0] == 0 + 0j

    with pytest.raises(VersionControlException):
        com.undo()
    with pytest.raises(VersionControlException):
        com.undo()

    com.redo()
    assert ad.freq(0, 2)[0, 0] == 0 + 1j
    assert ad.freq(0, 2)[1, 0] == 1 + 2j
    com.undo()
    assert ad._freq_data[0, 0] == 0 + 0j
    assert ad._freq_data[1, 0] == 0 + 0j
    com2 = SetFreq(0, 2, np.array([2 + 3j, 3 + 4j]), chanel=0, target=ad)
    com2.do()
    assert ad.freq(0, 2)[0, 0] == 2 + 3j
    assert ad.freq(0, 2)[1, 0] == 3 + 4j
    with pytest.raises(VersionControlException):
        com2.redo()


def test_command_manager():
    ad = AudioData.from_file("test.wav")
    ad.freq(0, 2)[:, 0] = np.array([0 + 0j, 0 + 0j])
    assert ad._freq_data[0, 0] == 0 + 0j
    assert ad._freq_data[1, 0] == 0 + 0j
    com1 = SetFreq(0, 2, np.array([0 + 1j, 1 + 2j]), chanel=0, target=ad)
    com2 = SetFreq(0, 2, np.array([3 + 4j, 4 + 5j]), chanel=0, target=ad)
    com3 = SetFreq(0, 2, np.array([5 + 6j, 6 + 7j]), chanel=0, target=ad)
    manager = CommandManager()
    manager.do(com1)
    assert ad.freq(0, 2)[0, 0] == 0 + 1j
    assert ad.freq(0, 2)[1, 0] == 1 + 2j
    manager.do(com2)
    assert ad.freq(0, 2)[0, 0] == 3 + 4j
    assert ad.freq(0, 2)[1, 0] == 4 + 5j
    manager.do(com1)
    assert ad.freq(0, 2)[0, 0] == 0 + 1j
    assert ad.freq(0, 2)[1, 0] == 1 + 2j
    manager.undo()
    assert ad.freq(0, 2)[0, 0] == 3 + 4j
    assert ad.freq(0, 2)[1, 0] == 4 + 5j
    manager.undo()
    assert ad.freq(0, 2)[0, 0] == 0 + 1j
    assert ad.freq(0, 2)[1, 0] == 1 + 2j
    manager.redo()
    assert ad.freq(0, 2)[0, 0] == 3 + 4j
    assert ad.freq(0, 2)[1, 0] == 4 + 5j
    manager.undo()
    manager.do(com3)
    assert ad.freq(0, 2)[0, 0] == 5 + 6j
    assert ad.freq(0, 2)[1, 0] == 6 + 7j
    with pytest.raises(VersionControlException):
        manager.redo()
    manager.undo()
    manager.undo()
    assert ad.freq(0, 2)[0, 0] == 0 + 0j
    assert ad.freq(0, 2)[1, 0] == 0 + 0j


def test_listener():
    class TestDataView(DataView):
        def __init__(self):
            self.counter = 0

        def freq_change_callback(self, index_start: int, index_end: int) -> None:
            self.counter += 1

    dv = TestDataView()
    ad = AudioData.from_file("test.wav")
    ad.freq(0, 2)[:, 0] = np.array([0 + 0j, 0 + 0j])
    assert ad._freq_data[0, 0] == 0 + 0j
    assert ad._freq_data[1, 0] == 0 + 0j
    com1 = SetFreq(0, 2, np.array([0 + 1j, 1 + 2j]), chanel=0, target=ad, listener=[dv])
    com2 = SetFreq(0, 2, np.array([3 + 4j, 4 + 5j]), chanel=0, target=ad, listener=[dv])
    com3 = SetFreq(0, 2, np.array([5 + 6j, 6 + 7j]), chanel=0, target=ad, listener=[dv])
    manager = CommandManager()
    assert dv.counter == 0
    manager.do(com1)
    assert ad.freq(0, 2)[0, 0] == 0 + 1j
    assert ad.freq(0, 2)[1, 0] == 1 + 2j
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