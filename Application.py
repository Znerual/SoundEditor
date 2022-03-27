import tkinter as tk
import numpy as np

from SoundEditor.AudioData import AudioData, VersionControlArray
from SoundEditor.Commands import CommandManager, SetFreq, equalizer_callback
from SoundEditor.DataView import LineFigureDataView,  BarFigureDataView,  EqualizerFigureDataView, TOOLBAR_POSITION
from SoundEditor.DataView import x_data_freq, y_data_freq_imag, y_data_freq_real, y_data_freq_abs



def main():
    """ run the application"""
    root = tk.Tk()

    root.bind("<Key>", lambda x : print(x, type(x)))
    frequency_frame = tk.Frame(master=root)
    time_frame = tk.Frame(master=root)

    frequency_frame.pack(side=tk.TOP, fill=tk.BOTH)
    time_frame.pack(side=tk.TOP, fill=tk.BOTH)

    audio_data = AudioData.from_file("test.wav")
    manager = CommandManager(audio_data)
    manager.register_callback("equ_clicked", equalizer_callback)
    start_index = 80000
    end_index = 81024
    audio_data.freq(start_index, end_index)

    equ_dv = EqualizerFigureDataView(frequency_frame,
                                     data= audio_data,
                                     x=[x_data_freq, x_data_freq, x_data_freq],
                                     y=[y_data_freq_abs, y_data_freq_real, y_data_freq_imag],
                                     command_manager=manager,
                                     position=tk.LEFT,
                                     toolbar_pos=TOOLBAR_POSITION.TOP)


    #bar_dv = BarFigureDataView(frequency_frame,
    #                           x=np.arange(1,7,1, dtype=int),
    #                           y=audio_data.freq_x[audio_data.find_peaks()],
    #                           position=tk.LEFT,
    #                           toolbar_pos=TOOLBAR_POSITION.TOP)

    root.mainloop()

if __name__ == "__main__":
    main()