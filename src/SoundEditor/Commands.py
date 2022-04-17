from __future__ import annotations

import copy
import queue
import sys
import threading
import numpy as np
import sounddevice as sd

from .AudioData import AudioData, VersionControlException, VersionControlArray
from .CommandManager import Command, CommandException, CommandManager

from numpy.typing import NDArray
from typing import List, Callable, Optional, Dict, Any, Union, Tuple


class SetFreq(Command):
    """ Set frequency values """

    def __init__(self, start_index: int, end_index: int, value: NDArray[np.csingle], chanel: int, target: AudioData, listener: List["DataView"] = []):
        self.start_index = start_index
        self.end_index = end_index
        self.value = value.copy()
        self.chanel = chanel
        self.target = target
        self.listener = listener

    def do(self) -> None:
        """ Change frequency and notify listeners """
        self.target.freq[self.start_index:self.end_index, self.chanel] = self.value
        for listener in self.listener:
            listener.freq_change_callback(self.target, self.start_index, self.end_index)

    def undo(self) -> None:
        """ undo a command. Check if nothing changed data inbetween"""
        if np.all(self.target.freq[:, self.chanel] != self.value):
            raise VersionControlException("Can't undo command because data was changed.")

        self.target.freq_undo()

        for listener in self.listener:
            listener.freq_change_callback(self.target, self.start_index, self.end_index)

    def redo(self) -> None:
        """ redos the command"""
        self.target.freq_redo()

        for listener in self.listener:
            listener.freq_change_callback(self.target, self.start_index, self.end_index)


class TimeToFreq(Command):
    """ Changes the current selection of a time frame which is Fourier transformed and shown """

    def __init__(self, start_index: int, end_index: int, target: AudioData,
                 listener: List["DataView"] = []):
        self.start_index = start_index
        self.end_index = end_index
        self.target = target
        self.listener = listener
        self._history_indices = (-1, -1)

    def do(self) -> None:
        """ changes the current time frame and does the fourier transformation """
        history_freq = copy.deepcopy(self.target.freq.history)
        self._history_indices = (self.target.start_index, self.target.end_index)
        self.target.ft(self.start_index, self.end_index)
        self.target.freq.history.extend(history_freq)

        for listener in self.listener:
            listener.freq_change_callback(self.target, self.start_index, self.end_index)
            listener.time_change_callback(self.target, self.start_index, self.end_index)

    def undo(self) -> None:
        """ undos the change of time frame"""
        if self._history_indices == (-1, -1) or (self.target.start_index, self.target.end_index) == (self._history_indices[0], self._history_indices[1]):
            raise VersionControlException("Can't undo command because of empty history.")
        history_freq = copy.deepcopy(self.target.freq.history)
        self.target.ft(self._history_indices[0], self._history_indices[1])
        self.target.freq.history.extend(history_freq)

        for listener in self.listener:
            listener.freq_change_callback(self.target, self.target.start_index, self.target.end_index)
            listener.time_change_callback(self.target, self.start_index, self.end_index)

    def redo(self) -> None:
        """ redos the change of time frame """
        if (self.target.start_index, self.target.end_index) == (self.start_index, self.end_index):
            VersionControlException("Can't redo command because of empty history.")
        history_freq = copy.deepcopy(self.target.freq.history)
        self._history_indices = (self.target.start_index, self.target.end_index)
        self.target.ft(self.start_index, self.end_index)
        self.target.freq.history.extend(history_freq)

        for listener in self.listener:
            listener.freq_change_callback(self.target, self.start_index, self.end_index)
            listener.time_change_callback(self.target, self.start_index, self.end_index)


class SetTime(Command):
    """ changes the audio data in the time domain """
    def __init__(self, start_index: int, end_index: int, value: NDArray[np.csingle], chanel: int, target: AudioData, listener: List["DataView"] = []):
        self.start_index = start_index
        self.end_index = end_index
        self.value = value.copy()
        self.chanel = chanel
        self.target = target
        self.listener = listener

    def do(self) -> None:
        """ Change time data and notify listeners """
        #print("set time")
        self.target.time[self.start_index:self.end_index, self.chanel] = self.value
        #for listener in self.listener:
        #    listener.time_change_callback(self.target, self.start_index, self.end_index)

    def undo(self) -> None:
        """ undo a command. Check if nothing changed data inbetween"""
        if np.all(self.target.time[:, self.chanel] != self.value):
            raise VersionControlException("Can't undo command because data was changed.")

        self.target.time_undo()

        for listener in self.listener:
            listener.time_change_callback(self.target, self.start_index, self.end_index)

    def redo(self) -> None:
        """ redos the command"""
        self.target.time_redo()

        for listener in self.listener:
            listener.time_change_callback(self.target, self.start_index, self.end_index)


class FreqToTime(Command):
    """ converts the current frequency data back to time data """
    def __init__(self, target: AudioData, listener: List["DataView"] = []):
        self.target = target
        self.listener = listener

    def do(self) -> None:
        self.target.ift()
        for listener in self.listener:
            listener.time_change_callback(self.target, self.target.start_index, self.target.end_index)

    def undo(self) -> None:
        self.target.time_undo()
        self.target.time_undo()
        for listener in self.listener:
            listener.time_change_callback(self.target, self.target.start_index, self.target.end_index)

    def redo(self) -> None:
        self.target.time_redo()
        self.target.time_redo()
        for listener in self.listener:
            listener.time_change_callback(self.target, self.target.start_index, self.target.end_index)


class PlaySegment(threading.Thread):

    def __init__(self, target: AudioData, start_index: int, end_index: int, listener: List["DataView"] = []):
        threading.Thread.__init__(self)
        self.start_index = start_index
        self.end_index = end_index
        self.target = target
        self.listener = listener

        self.current_index = start_index
        self.block_size = self.target.fs // 15

    def callback(self, outdata, frames, time, status):
        # assert frames == args.blocksize
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            #raise sd.CallbackAbort
        #assert not status

        end_index = self.current_index + self.block_size if self.current_index + self.block_size <= self.end_index else self.end_index
        data = self.target.time[self.current_index:end_index]
        self.current_index += self.block_size

        if len(data) < len(outdata):
            outdata[:len(data)] = data
            outdata[len(data):].fill(0)
            for listener in self.listener:
                listener.play_timecode_callback(self.target, self.end_index)
            raise sd.CallbackStop
        else:
            for listener in self.listener:
                listener.play_timecode_callback(self.target, self.current_index)
            outdata[:] = data

    def run(self) -> None:
        event = threading.Event()
        stream = sd.OutputStream(
            samplerate=self.target.fs, blocksize=self.block_size,
            device=sd.default.device[1], channels=self.target.channels,
            callback=self.callback, finished_callback=event.set)

        with stream:
            event.wait()  # Wait until playback is fi

        for listener in self.listener:
            listener.play_timecode_callback(self.target, self.start_index)


class Play(Command):
    """ plays a given chunk of the audio data and updates the time selection"""

    def __init__(self, target: AudioData, start_index: int, end_index: int, listener: List["DataView"] = []):
        self.audio_thread = PlaySegment(target=target, start_index=start_index, end_index=end_index, listener=listener)

    def do(self) -> None:
        self.audio_thread.start()

    def undo(self) -> None:
        pass

    def redo(self) -> None:
        pass
    