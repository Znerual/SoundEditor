import tkinter as tk
import numpy as np

from SoundEditor.AudioData import AudioData, VersionControlArray
from SoundEditor.CommandManager import CommandManager
from SoundEditor.Callbacks import equalizer_callback, timeline_callback, key_pressed_callback, timewindow_clicked
from SoundEditor.Commands import SetFreq
from SoundEditor.DataView import TimeLineFigureDataView,  BarFigureDataView,  EqualizerFigureDataView, TOOLBAR_POSITION, EqualizerZoomFigureDataView, TimeEqualizerFigureDataView
from SoundEditor.DataView import x_data_freq, y_data_freq_imag, y_data_freq_real, y_data_freq_abs, x_data_time, y_data_time, x_window_time, y_window_time_damped, y_window_time, y_window_invdamping, y_window_damping, x_rel_window_time



def main():
    """ run the application"""
    root = tk.Tk()
    root2 = tk.Toplevel(root)
    root3 = tk.Toplevel(root)

    frequency_frame = tk.Frame(master=root)
    time_frame = tk.Frame(master=root)

    frequency_frame.pack(side=tk.TOP, fill=tk.BOTH)
    time_frame.pack(side=tk.TOP, fill=tk.BOTH)

    audio_data = AudioData.from_file("test.wav")
    manager = CommandManager(audio_data)

    start_index = 80000
    end_index = 81024
    audio_data.ft(start_index, end_index)

    # create main figures
    equ_dv = EqualizerFigureDataView(root=frequency_frame,
                                     data=audio_data,
                                     x=[x_data_freq, x_data_freq, x_data_freq],
                                     y=[y_data_freq_abs, y_data_freq_real, y_data_freq_imag],
                                     command_manager=manager,
                                     position=tk.LEFT,
                                     toolbar=True)

    equ_zoom_dv = EqualizerZoomFigureDataView(x_span=200,
                                              root=root2,
                                              position=tk.TOP,
                                              data=audio_data,
                                              x=[x_data_freq, x_data_freq, x_data_freq],
                                              y=[y_data_freq_abs, y_data_freq_real, y_data_freq_imag],
                                              command_manager=manager,
                                              toolbar=True)

    time_equ_dv = TimeEqualizerFigureDataView(root=root3,
                                              data=audio_data,
                                              x=[x_window_time, x_window_time, x_window_time, x_rel_window_time],
                                              y=[y_window_time, y_window_time_damped, y_window_damping],
                                              command_manager=manager,
                                              position=tk.BOTTOM,
                                              toolbar=True)

    # register click callback for frequency equalizer and zoom of frequency equalizer
    manager.register_callback("equ_clicked", equalizer_callback, [equ_dv, equ_zoom_dv, time_equ_dv])
    manager.register_callback("timewindow_clicked", timewindow_clicked, [time_equ_dv, equ_dv, equ_zoom_dv])

    # link tkinter key press to the command manager callback of a keypress
    root.bind("<Key>", lambda event: manager.call("key_pressed", event))
    root2.bind("<Key>", lambda event: manager.call("key_pressed", event))
    root3.bind("<Key>", lambda event: manager.call("key_pressed", event))

    time_dv = TimeLineFigureDataView(root=time_frame,
                                     data=audio_data,
                                     x=[x_data_time],
                                     y=[y_data_time],
                                     selection_window=(start_index / audio_data.fs, end_index  / audio_data.fs),
                                     command_manager=manager,
                                     position=tk.TOP,
                                     dpi=100,
                                     figsize=(6,2))

    manager.register_callback("timeline_clicked", timeline_callback, [equ_dv, equ_zoom_dv, time_dv, time_equ_dv])

    # register pressed key callback listeners
    manager.register_callback("key_pressed", key_pressed_callback, [equ_dv, equ_zoom_dv, time_dv, time_equ_dv])


    #bar_dv = BarFigureDataView(frequency_frame,
    #                           x=np.arange(1,7,1, dtype=int),
    #                           y=audio_data.freq_x[audio_data.find_peaks()],
    #                           position=tk.LEFT,
    #                           toolbar_pos=TOOLBAR_POSITION.TOP)

    root.mainloop()

if __name__ == "__main__":
    main()