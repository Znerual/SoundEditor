import numpy as np
import pytest

from SoundEditor.AudioData import VersionControlArray, VersionControlException

def test_VersionControlArray_Creation():
    array = VersionControlArray([1,2,3,4,5,6,7,8,9], dtype=int)
    assert array.history == []
    assert array.shape[0] == 9

def test_VersionControlArray_Creation_Empty():
    array = VersionControlArray.empty((10,1), dtype=int)
    assert array.history == []
    assert array.shape[0] == 10

def test_VersionControlArray_history():
    array = VersionControlArray([1,2,3,4], dtype=int)
    assert array[2] == 3
    array[2] = 10
    assert array.history == [(2, 3)]
    assert array[2] == 10
    array[2] = 3
    assert array[2] == 3
    assert array.history == [(2,3), (2,10)]
    array[1:3] = [-1,0]
    assert array[1] == -1
    assert array[2] == 0
    assert array.history[0] == (2,3)
    assert array.history[1] == (2,10)
    assert array.history[2][0] == slice(1,3)
    assert np.all(array.history[2][1] == np.array([2,3]))

def test_VersionControlArray_undo():
    array = VersionControlArray([1, 2, 3, 4], dtype=int)
    array[0] = -1
    assert array[0] == -1
    assert array.history == [(0,1)]
    array.undo()
    assert array[0] == 1
    assert array.history == []
    assert array._redo == [(0,-1)]
    with pytest.raises(VersionControlException):
        array.undo()

def test_VersionControlArray_redo():
    array = VersionControlArray(["Ha", "llo", " world", "!"])
    assert array[0] == "Ha"
    array[0] = 0
    assert array[0] == '0'
    array[0] = -12.3
    assert array[0] == '-12.3'
    array.undo()
    assert array[0] == '0'
    array.undo()
    assert array[0] == "Ha"
    array.redo()
    assert array[0] == '0'
    array.redo()
    assert array[0] == '-12.3'
    with pytest.raises(VersionControlException):
        array.redo()
    array.undo()
    assert array[0] == '0'
    array[0] = "Ab"
    with pytest.raises(VersionControlException):
        array.redo()
    array.undo()
    assert array[0] == '0'
    array.undo()
    assert array[0] == "Ha"
    array.redo()
    assert array[0] == '0'
    array.redo()
    assert array[0] == "Ab"