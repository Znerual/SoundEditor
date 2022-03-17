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
peak_width = 1

N = int(fs * interval)
N_tot = int(fs * seconds)
T = 1 / fs
t = np.linspace(0.0, interval, N, endpoint=False)

index = 0
playing = False
play_speed = int(fs / 15)

def four_trans_seq(audio, index_start, index_stop, n_dom=6, fs=fs):
    N = index_stop - index_start
    xf = np.zeros((N // 2,), dtype="float64")
    yf = np.zeros((N, 2), dtype="complex64")

    k_dom_ind = np.zeros((n_dom, 2), dtype="int32")


    # get axis
    xf[:] = fftfreq(N, T)[:N // 2]  # positive frequencies


    yf[:, 0] = fft(audio[index_start:index_stop, 0])  # fft of channel 1
    yf[:, 1] = fft(audio[index_start:index_stop, 1])  # fft of channel 2

    yf0_tmp = np.abs(yf[:, 0])[:N // 2]
    yf1_tmp = np.abs(yf[:, 1])[:N // 2]

    # dominant frequencies
    # channel 1
    peaks1, _ = find_peaks(yf0_tmp, height=0, distance=int(40 / (xf[1] - xf[0])), prominence=0.5)
    if peaks1.shape[0] >= n_dom:
        idx = np.argpartition(yf0_tmp[peaks1], -n_dom)[-n_dom:]
        index = idx[np.argsort(-yf0_tmp[peaks1][idx])]
        k_dom_ind[:, 0] = peaks1[index]

    # channel 2
    peaks2, _ = find_peaks(yf1_tmp, height=0, distance=int(40 / (xf[1] - xf[0])), prominence=0.5)
    if peaks2.shape[0] >= n_dom:
        idx = np.argpartition(yf1_tmp[peaks2], -n_dom)[-n_dom:]
        index = idx[np.argsort(-yf1_tmp[peaks2][idx])]
        k_dom_ind[:, 1] = peaks2[index]

    return xf, yf, k_dom_ind

def four_trans_seq_all(audio, n_dom=6, fs=fs, seconds=seconds, N=N, T=T, interval_step=interval_step):
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



def update_vlines(*, h, x, ymin=None, ymax=None):
    seg_old = h.get_segments()
    if ymin is None:
        ymin = seg_old[0][0, 1]
    if ymax is None:
        ymax = seg_old[0][1, 1]

    seg_new = [np.array([[xx, ymin],
                         [xx, ymax]]) for xx in x]

    h.set_segments(seg_new)

def set_vspan(polygon, x0, x1):
    _ndarray = polygon.get_xy()
    _ndarray[:, 0] = [x0, x0, x1, x1, x0]
    polygon.set_xy(_ndarray)



def main_window():
    global xf,yf, myrecording
    # create master window
    root = tk.Tk()

    # create layout frames
    toolframe = tk.Frame(master=root)
    toolframe.pack(side=tk.TOP, fill=tk.X)
    graphsframe = tk.Frame(master=root)
    graphsframe.pack(side=tk.TOP, fill=tk.X)
    wavframe = tk.Frame(master=root)
    wavframe.pack(side=tk.TOP, fill=tk.X)
    uiframe = tk.Frame(master=root)
    uiframe.pack(side=tk.TOP, fill=tk.X)

    # load audio data
    myrecording, fs = load("test.wav")

    # fourier transform and time axis
    t = np.arange(0, myrecording.shape[0] / fs, 1/fs)
    #xf, yf, k_dom_ind = four_trans_seq_all(myrecording)
    xf, yf, k_dom_ind = four_trans_seq(myrecording, index, int(interval*fs+index))

    # add equilizer plot
    figure1 = plt.Figure(figsize=(6,5), dpi=100)
    ax1 = figure1.add_subplot(111)
    line, = ax1.plot(xf, 2 / N* np.abs(yf[:,0][:N//2]), label="2/N |y(f)|")
    line_im, = ax1.plot(xf, 2 / N* np.abs(np.imag(yf[:,0][:N//2])), alpha=0.5, label="2/N |Im(y(f))|")
    line_re, = ax1.plot(xf, 2 / N* np.abs(np.real(yf[:,0][:N//2])), alpha=0.5, label="2/N |Re(y(f))|")
    scat = ax1.scatter(xf[k_dom_ind[:, 0]], 2.0/N * np.abs(yf[:,0])[:N//2][k_dom_ind[:, 0]], color="red")
    ax1.set_xlim(1, max(xf))
    ax1.set_ylim(0, 0.1)
    ax1.set_xlabel("Frequency f")
    ax1.set_ylabel("Absolute Value")
    ax1.legend()
    canvas1 = FigureCanvasTkAgg(figure1, master=graphsframe)
    canvas1.draw()

    toolbar = NavigationToolbar2Tk(canvas1, toolframe, pack_toolbar=False)
    toolbar.update()

    # add bar plot
    figure2 = plt.Figure(figsize=(6,5), dpi=100)
    ax2 = figure2.add_subplot(111)
    bar = ax2.bar(np.arange(1,n_dom + 1), xf[k_dom_ind[:,0]]) #np.zeros((n_dom))
    ax2.set_ylim(1, max(xf))

    canvas2 = FigureCanvasTkAgg(figure2, master=graphsframe)
    canvas2.draw()

    toolbar2 = NavigationToolbar2Tk(canvas2, toolframe, pack_toolbar=False)
    toolbar2.update()

    # add audio plot
    figure3 = plt.Figure(figsize=(6,2), dpi=50)
    ax3 = figure3.add_subplot(111)
    ax3.plot(t, myrecording)
    timeline = ax3.axvline(x=0)
    timewindow = ax3.axvspan(0, interval, alpha=0.5)
    ax3.set_xlim(0,myrecording.shape[0] / fs)
    canvas3 = FigureCanvasTkAgg(figure3, master=wavframe)
    canvas3.draw()


    def timeline_clicked(event):
        global index, interval

        x_index = event.xdata * fs
        if x_index < index:
            index = x_index
        else:
            interval = (x_index - index) / fs

        update_frequency(index)
    canvas3.mpl_connect("button_press_event", timeline_clicked)



    def freq_clicked(event):
        global xf, yf, myrecording

        def bell_curve(halve_width : int):
            if halve_width == 0:
                return np.array([1.0])
            else:
                x = np.arange(-halve_width, halve_width+1,1)
                return  np.exp(-0.5 *(4.0 * x / halve_width)**2) # 2.0 / (halve_width * np.sqrt(2.0 * np.pi)) *

        N = xf.shape[0]
        print(yf.shape, xf.shape)
        #nonlocal myrecording

        #xf, yf, _ = four_trans_seq(myrecording, index, int(interval * fs + index))
        # get selected frequency
        x_data = int(round(event.xdata))
        delta_x = xf[1] - xf[0]
        f_ind = np.where((xf == x_data) | ((x_data - delta_x <= xf) & (xf <= x_data + delta_x)))[0]
        if f_ind.shape == 3:
            ind = f_ind[1]
        else:
            ind = f_ind[0]

        peak_width = slider_peak_width.get()
        curve = bell_curve(peak_width)
        if event.button == 2: # middle mouse
            yf[ind-peak_width:ind+peak_width+1,0].real = event.ydata * N * curve
            yf[ind + N  : ind + N + 1,0].real = event.ydata * N
            yf[ind : ind + 1, 1].real = event.ydata * N
            yf[ind + N  : ind + N  + 1, 1].real = event.ydata * N
        elif event.button == 3: # right mouse
            yf[ind:ind+1,0].imag = event.ydata * 2.0 / N
            yf[ind + N :ind+1 + N ,0].imag = event.ydata * 2.0 / N
            yf[ind:ind+1, 1].imag = event.ydata * 2.0 / N
            yf[ind + N :ind + N + 1, 1].imag = event.ydata * 2.0 / N
        myrecording[index:int(index+interval*fs),0] = np.abs(ifft(yf[:,0]))
        myrecording[index:int(index+interval*fs),1] = np.abs(ifft(yf[:,1]))
        update_view()

    canvas1.mpl_connect(
        "button_press_event",freq_clicked)



    def key_pressed(event):
        global playing

        if event.keycode == 32:
            playing = not playing

        if event.char == "p":
            sd.play(myrecording[index:int(index+interval*fs)],fs)
        print(event.char)
        print(event)

    root.bind("<Key>", key_pressed)
    def play():
        if playing:
            update_frequency(index+play_speed)
            slider_update.set(index)

    def update_view():
        N = int(interval * fs + index) - index
        ax1.set_xlim(xf[0], xf[-1])
        line.set_xdata(xf)
        line_im.set_xdata(xf)
        line_re.set_xdata(xf)
        line.set_ydata(2.0 / N * np.abs(yf[:, 0])[:N // 2])
        line_im.set_ydata(2.0 / N * np.imag(yf[:, 0])[:N // 2])
        line_re.set_ydata(2.0 / N * np.real(yf[:, 0])[:N // 2])

        x_tmp = xf[k_dom_ind[:, 0]]
        y_tmp = 2.0 / N * np.abs(yf[:, 0])[:N // 2][k_dom_ind[:, 0]]
        data = np.vstack((x_tmp, y_tmp)).transpose()

        # draw points for maxima
        scat.set_offsets(data)

        for j, b in enumerate(bar):
            b.set_height(xf[k_dom_ind[j, 0]])

        canvas1.draw()
        canvas2.draw()

        timeline.set_data([index / fs, index / fs], [0, 1])
        set_vspan(timewindow, index / fs, index / fs + interval)

        canvas3.draw()

    def update_frequency(new_val):
        # retrieve frequency
        global index, interval, xf, yf
        index = int(new_val)

        if index + interval * fs >= myrecording.shape[0]:
            interval = (myrecording.shape[0] - index) / fs

        xf, yf, k_dom_ind = four_trans_seq(myrecording, index, int(interval * fs + index))

        update_view()

    slider_update = tk.Scale(uiframe, from_=0, to=myrecording.shape[0]-1, orient=tk.HORIZONTAL,
                             command=update_frequency, label="Timewindow")


    slider_peak_width = tk.Scale(uiframe, from_=0, to=xf.shape[0]-1, orient=tk.HORIZONTAL, label="Peak Width")

    def freq_scrolled(event):
        val = slider_peak_width.get()
        if event.button == "up":
            if val < xf.shape[0]:
                slider_peak_width.set(val+1)
        elif event.button == "down":
            if val > 1:
                slider_peak_width.set(val-1)


    canvas1.mpl_connect("scroll_event", freq_scrolled)

    button_quit = tk.Button(master=uiframe, text="Quit", command=root.quit)

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
    slider_peak_width.pack(side=tk.LEFT)

    while True:
        play()
        root.update()
    #root.after(17, play)
    root.mainloop()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_window()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
