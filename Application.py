import tkinter as tk
import numpy as np


from pathlib import Path

from SoundEditor.AudioData import AudioData, VersionControlArray, ffmpeg_formats
from SoundEditor.CommandManager import CommandManager
from SoundEditor.Callbacks import equalizer_callback, timeline_callback, key_pressed_callback, timewindow_clicked, \
    update_data_callback
from SoundEditor.Commands import SetFreq
from SoundEditor.Configuration import Configuration
from SoundEditor.DataView import TimeLineFigureDataView, EqualizerFigureDataView, TOOLBAR_POSITION, \
    EqualizerZoomFigureDataView, TimeEqualizerFigureDataView
from SoundEditor.DataView import x_data_freq, y_data_freq_imag_min, y_data_freq_imag_max, y_data_freq_real_min, y_data_freq_real_max, y_data_freq_abs, x_data_time, \
    y_data_time, x_window_time, y_window_time_damped_min, y_window_time_damped_max, y_window_time_min, y_window_time_max, y_window_invdamping, y_window_damping_min, y_window_damping_max, \
    x_rel_window_time


class Application:

    def on_open(self):
        """ opens a file dialog and load the selected file"""

        initial_dir = self.config.get("last_open_path", default="/")
        formats = tuple([("All Files", "*.*")] + ffmpeg_formats())

        filepath = tk.filedialog.askopenfilename(initialdir=initial_dir, title="Open file",
                                            filetypes=formats)

        # catch the case when no file was selected
        if filepath == '':
            return

        # save open path
        self.config["last_open_path"] = Path(filepath).resolve().parent
        self.config["current_file_saved_as"] = ""

        # load and set audio data
        self.manager._data = AudioData.from_file(filepath)
        self.manager.call("data_update", None, None)

    def on_save_as(self):
        initial_dir = self.config.get("last_saved_paths", ["/"])[0]
        formats = tuple([("All Files", "*.*")] + ffmpeg_formats())

        filepath = tk.filedialog.asksaveasfilename(initialdir=initial_dir, title="Save as",
                                              filetypes=formats)

        if filepath == "":
            return

        if "last_saved_paths" in self.config:
            self.config["last_saved_paths"].append(Path(filepath).resolve().parent)
            self.config.save()
        else:
            self.config["last_saved_paths"] = [Path(filepath).resolve().parent]

        self.config["current_file_saved_as"] = Path(filepath).resolve()

        self.manager.data.save_to_file(filepath, filepath.split(".")[1])

    def on_save(self):
        if self.config.get("current_file_saved_as", default="") == "":
            self.on_save_as()
        else:
            print(f"Saved to {self.config['current_file_saved_as']}")
            self.manager.data.save_to_file(self.config["current_file_saved_as"], self.config["current_file_saved_as"].split(".")[1])

    def main_application(self):
        """ run the application"""
        self.root = tk.Tk()
        self.root.title("Sound Editor")

        self.menubar = tk.Menu(self.root)

        filemenu = tk.Menu(self.menubar)
        filemenu.add_command(label="New")
        filemenu.add_command(label="Open", command=self.on_open)
        filemenu.add_command(label="Save")
        filemenu.add_command(label="Save As", command=self.on_save_as)

        self.menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=self.menubar)

        self.root2 = tk.Toplevel(self.root)
        self.root2.title("Frequency Window")

        self.root3 = tk.Toplevel(self.root)
        self.root3.title("Time Window")

        self.frequency_frame = tk.Frame(master=self.root)
        self.time_frame = tk.Frame(master=self.root)

        self.frequency_frame.pack(side=tk.TOP, fill=tk.BOTH)
        self.time_frame.pack(side=tk.TOP, fill=tk.BOTH)

        audio_data = AudioData.from_file("test.wav")
        self.manager = CommandManager(audio_data)

        # create main figures
        self.equ_dv = EqualizerFigureDataView(root=self.frequency_frame,
                                              x=[x_data_freq, x_data_freq, x_data_freq],
                                              y=[y_data_freq_abs, y_data_freq_real_max, y_data_freq_imag_max],
                                              legend=["Abs", "Real", "Imag"],
                                              command_manager=self.manager,
                                              position=tk.LEFT,
                                              alpha=0.75,
                                              toolbar=True)

        self.equ_zoom_dv = EqualizerZoomFigureDataView(x_span=200,
                                                       root=self.root2,
                                                       position=tk.TOP,
                                                       x=[x_data_freq, x_data_freq, x_data_freq, x_data_freq,
                                                          x_data_freq],
                                                       y=[y_data_freq_abs, y_data_freq_real_min, y_data_freq_real_max,
                                                          y_data_freq_imag_min, y_data_freq_imag_max],
                                                       alpha=0.6,
                                                       legend=["Abs", "Min Real", "Max Real", "Min Imag", "Max Imag"],
                                                       command_manager=self.manager,
                                                       toolbar=True)

        self.time_equ_dv = TimeEqualizerFigureDataView(root=self.root3,
                                                       x=[x_window_time, x_window_time, x_window_time, x_window_time],
                                                       y=[y_window_time_max, y_window_time_min, y_window_time_damped_max, y_window_time_damped_min],
                                                       command_manager=self.manager,
                                                       position=tk.BOTTOM,
                                                       toolbar=True)

        # register click callback for frequency equalizer and zoom of frequency equalizer
        self.manager.register_callback("equ_clicked", equalizer_callback,
                                       [self.equ_dv, self.equ_zoom_dv, self.time_equ_dv])
        self.manager.register_callback("timewindow_clicked", timewindow_clicked,
                                       [self.time_equ_dv, self.equ_dv, self.equ_zoom_dv])

        # link tkinter key press to the command manager callback of a keypress
        self.root.bind("<Key>", lambda event: self.manager.call("key_pressed", event))
        self.root2.bind("<Key>", lambda event: self.manager.call("key_pressed", event))
        self.root3.bind("<Key>", lambda event: self.manager.call("key_pressed", event))

        self.time_dv = TimeLineFigureDataView(root=self.time_frame,
                                              x=[x_data_time],
                                              y=[y_data_time],
                                              selection_window=(self.manager.data.start_index / self.manager.data.fs,
                                                                self.manager.data.end_index / self.manager.data.fs),
                                              command_manager=self.manager,
                                              position=tk.TOP,
                                              dpi=100,
                                              figsize=(6, 2))

        self.manager.register_callback("timeline_clicked", timeline_callback,
                                       [self.equ_dv, self.equ_zoom_dv, self.time_dv, self.time_equ_dv])

        # register pressed key callback listeners
        self.manager.register_callback("key_pressed", key_pressed_callback,
                                       [self.equ_dv, self.equ_zoom_dv, self.time_dv, self.time_equ_dv])

        # register data update callback
        self.manager.register_callback("data_update", update_data_callback, [self.equ_dv, self.equ_zoom_dv, self.time_dv, self.time_equ_dv])
        # bar_dv = BarFigureDataView(frequency_frame,
        #                           x=np.arange(1,7,1, dtype=int),
        #                           y=audio_data.freq_x[audio_data.find_peaks()],
        #                           position=tk.LEFT,
        #                           toolbar_pos=TOOLBAR_POSITION.TOP)

        self.root.mainloop()

    def __init__(self):
        self.config = Configuration()
        self.main_application()

if __name__ == "__main__":
    app = Application()

