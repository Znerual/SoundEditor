import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import sounddevice as sd
import soundfile as sf
import numpy as np

from scipy.io.wavfile import write
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks

fs = 44100
interval = 0.2
seconds = 5

interval_step_duration = 1 # in ms
interval_step = int(interval_step_duration / 1000  * fs) # in frames

n_dom = 6
# 50 ms intervals

N = int(fs * interval)
N_tot = int(fs * seconds)
T = 1 / fs
t = np.linspace(0.0, interval, N, endpoint=False)

index = 0
playing = False

def four_trans_seq(audio, n_dom=6, fs=fs, seconds=seconds, N=N, T=T, interval_step=interval_step):
    n_dom = 6
    xf = np.zeros(((fs * seconds - N) // interval_step, N // 2), dtype="float64")
    xft = np.zeros(((fs * seconds - N) // interval_step, N), dtype="float64")
    yf = np.zeros(((fs * seconds - N) // interval_step, N, 2), dtype="complex64")
    k_dom_ind = np.zeros(((fs * seconds - N) // interval_step, n_dom, 2), dtype="int32")

    for i in range((int(fs * seconds) - N) // interval_step):
        # get axis
        xf[i, :] = fftfreq(N, T)[:N // 2]  # positive frequencies
        xft[i, :] = fftfreq(N, T)  # all frequencies (redundant)

        yf[i, :, 0] = fft(audio[i * interval_step:i * interval_step + N, 0])  # fft of channel 1
        yf[i, :, 1] = fft(audio[i * interval_step:i * interval_step + N, 1])  # fft of channel 2

        yf0_tmp = np.abs(yf[i, :, 0])[:N // 2]
        yf1_tmp = np.abs(yf[i, :, 1])[:N // 2]
        # dominant frequencies
        # channel 1
        peaks1, _ = find_peaks(yf0_tmp, height=0, distance=int(40 / (xf[0, 1] - xf[0, 0])), prominence=0.5)
        if peaks1.shape[0] >= n_dom:
            idx = np.argpartition(yf0_tmp[peaks1], -n_dom)[-n_dom:]
            index = idx[np.argsort(-yf0_tmp[peaks1][idx])]
            k_dom_ind[i, :, 0] = peaks1[index]
            # k_dom[i,:, 0]= xf[i, k]
            # k_ind = peaks1[index]
            # if i > 1000:
            #    print(xf[i, peaks1], yf0_tmp[peaks1])
            ##    #print(xf[idx], yf0_tmp[idx])
            #    print(xf[i, peaks1][indices], yf0_tmp[peaks1][indices])
            #    plt.plot(xf[i,:], yf0_tmp)
            #    plt.scatter(k_dom[i,:,0],yf0_tmp[k_ind], color = "red")
            #    print(k_dom[i, :,0], )
            #    break
        # channel 2
        peaks2, _ = find_peaks(yf1_tmp, height=0, distance=int(40 / (xf[0, 1] - xf[0, 0])), prominence=0.5)
        if peaks2.shape[0] >= n_dom:
            idx = np.argpartition(yf1_tmp[peaks2], -n_dom)[-n_dom:]
            index = idx[np.argsort(-yf1_tmp[peaks2][idx])]
            k_dom_ind[i, :, 1] = peaks2[index]

    return xf, yf, k_dom_ind


def four_trans_tot(audio, N=N_tot, T=T):
    yf_tot = fft(audio[:, 0])
    xf_tot = fftfreq(N, T)[:N // 2]
    xft_tot = fftfreq(N, T)

    return xf_tot, yf_tot

def record(seconds):
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    return myrecording

def load(filename):
    myrecording, fs = sf.read(filename, dtype='float32')
    return myrecording, fs


def shift(arr, num, fill_value=0.0):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:N_tot // 2] = arr[:(N_tot // 2) - num]

        # move negative freq. as well
        result[-num:] = fill_value
        result[N_tot // 2:-num] = arr[(N_tot // 2) + num:]
    elif num < 0:
        raise ValueError("Not implemented")
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def shiftHz(arr, hz):
    num = int(hz / (xf_tot[1] - xf_tot[0]))
    return shift(arr, num)

def key_pressed(event):
    global playing

    if event.keycode == 32:
        playing = not playing

    print(event.char)
    print(event)

def update_vlines(*, h, x, ymin=None, ymax=None):
    seg_old = h.get_segments()
    if ymin is None:
        ymin = seg_old[0][0, 1]
    if ymax is None:
        ymax = seg_old[0][1, 1]

    seg_new = [np.array([[xx, ymin],
                         [xx, ymax]]) for xx in x]

    h.set_segments(seg_new)

def main_window():
    root = tk.Tk()

    root.bind("<Key>", key_pressed)
    toolframe = tk.Frame(master=root)
    toolframe.pack(side=tk.TOP, fill=tk.X)
    graphsframe = tk.Frame(master=root)
    graphsframe.pack(side=tk.TOP, fill=tk.X)
    wavframe = tk.Frame(master=root)
    wavframe.pack(side=tk.TOP, fill=tk.X)
    uiframe = tk.Frame(master=root)
    uiframe.pack(side=tk.TOP, fill=tk.X)

    myrecording, fs = load("test.wav")
    t = np.arange(0, myrecording.shape[0] / fs, 1/fs)
    xf, yf, k_dom_ind = four_trans_seq(myrecording)

    figure1 = plt.Figure(figsize=(6,5), dpi=100)
    ax1 = figure1.add_subplot(111)
    line, = ax1.plot(xf[index, :], 2 / N* np.abs(yf[index, :,0][:N//2]))
    scat = ax1.scatter(xf[index, k_dom_ind[index, :, 0]], 2.0/N * np.abs(yf[index,:,0])[:N//2][k_dom_ind[index, :, 0]], color="red")
    ax1.set_xlim(1, max(xf[0,:]))
    ax1.set_ylim(0, 0.1)
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Absolute Value")

    canvas1 = FigureCanvasTkAgg(figure1, master=graphsframe)
    canvas1.draw()

    figure2 = plt.Figure(figsize=(6,5), dpi=100)
    ax2 = figure2.add_subplot(111)
    bar = ax2.bar(np.arange(1,n_dom + 1), xf[index, k_dom_ind[index,:,0]]) #np.zeros((n_dom))
    ax2.set_ylim(1, max(xf[0,:]))

    canvas2 = FigureCanvasTkAgg(figure2, master=graphsframe)
    canvas2.draw()

    figure3 = plt.Figure(figsize=(6,2), dpi=50)
    ax3 = figure3.add_subplot(111)
    ax3.plot(t, myrecording)
    timeline = ax3.vlines(0, 0, 0.2)
    canvas3 = FigureCanvasTkAgg(figure3, master=wavframe)
    canvas3.draw()

    toolbar = NavigationToolbar2Tk(canvas1, toolframe, pack_toolbar=False)
    toolbar.update()

    toolbar2 = NavigationToolbar2Tk(canvas2, toolframe, pack_toolbar=False)
    toolbar2.update()

    canvas1.mpl_connect(
        "button_press_event", lambda event: print(f"{event} you clicked at {event.xdata} ({event.x}) {event.ydata} ({event.y}) with {event.button}, "))
    canvas1.mpl_connect("key_press_event", key_press_handler)

    button_quit = tk.Button(master=uiframe, text="Quit", command=root.quit)

    def play():
        if playing:
            update_frequency(index+10)
            slider_update.set(index)

    def update_frequency(new_val):
        # retrieve frequency
        global index
        index = int(new_val)

        line.set_xdata(xf[index, :])
        line.set_ydata(2.0 / N * np.abs(yf[index, :, 0])[:N // 2])

        x_tmp = xf[index, k_dom_ind[index, :, 0]]
        y_tmp = 2.0 / N * np.abs(yf[index, :, 0])[:N // 2][k_dom_ind[index, :, 0]]
        data = np.vstack((x_tmp, y_tmp)).transpose()

        # draw points for maxima
        scat.set_offsets(data)

        for j, b in enumerate(bar):
            b.set_height(xf[index, k_dom_ind[index, j, 0]])

        #update_vlines(timeline ,index * interval_step_duration / 1000)
        # required to update canvas and attached toolbar!
        canvas1.draw()
        canvas2.draw()

    slider_update = tk.Scale(uiframe, from_=0, to=xf.shape[0]-1, orient=tk.HORIZONTAL,
                                  command=update_frequency, label="Timewindow")

    # Packing order is important. Widgets are processed sequentially and if there
    # is no space left, because the window is too small, they are not displayed.
    # The canvas is rather flexible in its size, so we pack it last which makes
    # sure the UI controls are displayed as long as possible.



    toolbar.pack(side=tk.LEFT, fill=tk.X)
    toolbar2.pack(side=tk.RIGHT, fill=tk.X)
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    canvas3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    button_quit.pack(side=tk.RIGHT)
    slider_update.pack(side=tk.LEFT)

    while True:
        play()
        root.update()
    #root.after(17, play)
    root.mainloop()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_window()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
