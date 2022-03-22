import pytest

from SoundEditor.AudioData import AudioData, AudioDataException

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
    assert ad._audio_time_data.shape[0] / ad.fs == ad.seconds
    with pytest.raises(AttributeError):
        ad.seconds = 4.2