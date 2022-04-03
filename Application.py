import tkinter as tk
import numpy as np

from SoundEditor.AudioData import AudioData, VersionControlArray
from SoundEditor.Commands import CommandManager, SetFreq, equalizer_callback, timeline_callback
from SoundEditor.DataView import TimeLineFigureDataView,  BarFigureDataView,  EqualizerFigureDataView, TOOLBAR_POSITION, EqualizerZoomFigureDataView
from SoundEditor.DataView import x_data_freq, y_data_freq_imag, y_data_freq_real, y_data_freq_abs, x_data_time, y_data_time



def main():
    """ run the application"""
    root = tk.Tk()
    root2 = tk.Toplevel(root)
    root.bind("<Key>", lambda x : print(x, type(x)))
    frequency_frame = tk.Frame(master=root)
    time_frame = tk.Frame(master=root)

    frequency_frame.pack(side=tk.TOP, fill=tk.BOTH)
    time_frame.pack(side=tk.TOP, fill=tk.BOTH)

    audio_data = AudioData.from_file("test.wav")
    manager = CommandManager(audio_data)

    start_index = 80000
    end_index = 81024
    audio_data.ft(start_index, end_index)

    equ_dv = EqualizerFigureDataView(root=frequency_frame,
                                     data=audio_data,
                                     x=[x_data_freq, x_data_freq, x_data_freq],
                                     y=[y_data_freq_abs, y_data_freq_real, y_data_freq_imag],
                                     command_manager=manager,
                                     position=tk.LEFT)
    equ_zoom_dv = EqualizerZoomFigureDataView(x_span=200,
                                              root=root2,
                                              position=tk.TOP,
                                              data=audio_data,
                                              x=[x_data_freq, x_data_freq, x_data_freq],
                                              y=[y_data_freq_abs, y_data_freq_real, y_data_freq_imag],
                                              command_manager=manager)

    manager.register_callback("equ_clicked", equalizer_callback, [equ_dv, equ_zoom_dv])

    time_dv = TimeLineFigureDataView(root=time_frame,
                                     data=audio_data,
                                     x=[x_data_time],
                                     y=[y_data_time],
                                     selection_window=(start_index / audio_data.fs, end_index / audio_data.fs),
                                     command_manager=manager,
                                     position=tk.TOP,
                                     dpi=100,
                                     figsize=(6,2))

    manager.register_callback("timeline_clicked", timeline_callback, [equ_dv, equ_zoom_dv, time_dv])

    #bar_dv = BarFigureDataView(frequency_frame,
    #                           x=np.arange(1,7,1, dtype=int),
    #                           y=audio_data.freq_x[audio_data.find_peaks()],
    #                           position=tk.LEFT,
    #                           toolbar_pos=TOOLBAR_POSITION.TOP)

    root.mainloop()

if __name__ == "__main__":
    main()