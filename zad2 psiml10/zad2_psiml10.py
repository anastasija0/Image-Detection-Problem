import imageio
import numpy as np
import time
from scipy_filter import butterworth_bandpass, lfilter
import math
import matplotlib.pyplot as plt
from typing import Tuple

# Assuming you have your numpy array containing data
# Example array (replace this with your actual array):
def butterworth_bandpass(N: int, Wn: [float, float]) -> [np.ndarray, np.ndarray]:
    Wn = np.array(Wn)
    assert np.size(Wn) == 2, "Must specify a single critical frequency Wn for lowpass or highpass filter"
    assert np.all(Wn > 0) and np.all(Wn < 1), "Digital filter critical frequencies must be 0 < Wn < 1"

    z, p, k = buttap(N)
    warped = 4 * np.tan(np.pi * Wn / 2)  # digital

    bw = warped[1] - warped[0]
    wo = np.sqrt(warped[0] * warped[1])
    z, p, k = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
    z, p, k = bilinear_zpk(z, p, k, fs=2)
    b, a = zpk2tf(z, p, k)

    return b, a
def buttap(N):

    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    z = np.array([])
    m = np.arange(-N+1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -np.exp(1j * np.pi * m / (2 * N))
    k = 1
    return z, p, k
def lp2bp_zpk(z, p, k, wo=1.0, bw=1.0):

    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Scale poles and zeros to desired bandwidth
    z_lp = z * bw/2
    p_lp = p * bw/2

    # Square root needs to produce complex result, not NaN
    z_lp = z_lp.astype(complex)
    p_lp = p_lp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = np.concatenate((z_lp + np.sqrt(z_lp**2 - wo**2),
                           z_lp - np.sqrt(z_lp**2 - wo**2)))
    p_bp = np.concatenate((p_lp + np.sqrt(p_lp**2 - wo**2),
                           p_lp - np.sqrt(p_lp**2 - wo**2)))

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = np.append(z_bp, np.zeros(degree))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree

    return z_bp, p_bp, k_bp
def bilinear_zpk(z, p, k, fs):

    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    fs = float(fs)

    degree = _relative_degree(z, p)

    fs2 = 2.0*fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = np.append(z_z, -np.ones(degree))

    # Compensate for gain change
    k_z = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))

    return z_z, p_z, k_z
def zpk2tf(z, p, k):

    z = np.atleast_1d(z)
    k = np.atleast_1d(k)
    if len(z.shape) > 1:
        temp = np.poly(z[0])
        b = np.empty((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = k[i] * np.poly(z[i])
    else:
        b = k * np.poly(z)
    a = np.atleast_1d(np.poly(p))

    # Use real output if possible. Copied from np.poly, since
    # we can't depend on a specific version of np.
    if issubclass(b.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(z, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) == np.sort_complex(pos_roots)):
                b = b.real.copy()

    if issubclass(a.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(p, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) ==
                         np.sort_complex(pos_roots)):
                a = a.real.copy()

    return b, a
def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree
def lfilter(b, a, x):
    """A simple implementation of a 1-D linear filter."""
    a = np.array(a)
    b = np.array(b)
    y = np.zeros_like(x)
    a0 = a[0]
    if a0 != 1:
        a = a / a0
        b = b / a0
    for i in range(len(x)):
        for j in range(len(b)):
            if i - j >= 0:
                y[i] += b[j] * x[i - j]
        for j in range(1, len(a)):
            if i - j >= 0:
                y[i] -= a[j] * y[i - j]
    return y
def detect_circles(image,frame_idx):
    shapes = []
    gray = np.mean(image, axis=2)

    threshold = 60
    binary_image = np.where(gray > threshold, 1, 0)

    visited = []
    height, width = gray.shape
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 1 and (i, j) not in visited:
                shape = dfs(binary_image, i, j) #vraca obojene delove oblika
                shapes.append(shape)
                visited.extend(shape)
    #print(shapes)
    #print(visited)
    #shape_centers = []
    circles = []
    rectangles = []
    center_shapes = []
    for shape in shapes:
        #num_points = len(shape)
        x_min = 5000
        y_min = 5000
        x_max = 0
        y_max = 0
        for point in shape:
            #print(point)
            x = point[0]
            y = point[1]
            if (x < x_min):
                x_min = x
            if (x > x_max):
                x_max = x
            if (y < y_min):
                y_min = y
            if (y > y_max):
                y_max = y
        center_x = (x_min + x_max)/2
        center_y = (y_min + y_max)/2
        if binary_image[int(center_x), int(center_y)] == 1:
            radius = (x_max - x_min)/2
            circles.append((center_x, center_y, radius,0))
            center_shapes.append((center_x, center_y, radius,0))
        elif binary_image[int(center_x), int(center_y)] == 0:
            height = (x_max-x_min)/2
            length = (y_max-y_min)/2
            rectangles.append((center_x, center_y,height,length))
            center_shapes.append((center_x, center_y,height,length))

    if frame_idx == 0:
        leftmost_circle = min(circles, key=lambda c: c[1] - c[2])
        print(int(leftmost_circle[0]), int(leftmost_circle[1]))
        #print(circles)
        for r in rectangles:
            print(int(r[0]), int(r[1]))
    return center_shapes

def dfs(binary_image, i, j):
    stack = [(i, j)]
    colored = set()
    while stack:
        i, j = stack.pop()
        if (i, j) in colored or binary_image[i, j] == 0:
            continue
        colored.add((i, j))
        stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
    return colored


def find_leftmost_circle(video_path):
    leftmost_circle = None
    frame_rate = 30
    vid = imageio.get_reader(video_path)
    frame_duration = 1.0 / frame_rate
    frame_idx = 0
    while not frame_idx:
        frame = vid.get_next_data()
        shapes = detect_circles(frame,frame_idx)
        time.sleep(frame_duration)
        frame_idx += 1
    vid.close()
def return_rectangle(video_path):
    frame_rate = 30
    vid = imageio.get_reader(video_path)
    frame_duration = 1.0 / frame_rate
    frame_idx = 0
    while not frame_idx:
        frame = vid.get_next_data()
        shapes = detect_circles(frame, 1)
        time.sleep(frame_duration)
        frame_idx += 1
    for shape in shapes:
        if shape[3]!=0:
            return shape
    vid.close()
# Assuming sound_data is your numpy array containing the sound data
def check_collision(shape, rect):
    rect_area=[]
    x_r, y_r, height_r, length_r = rect[0], rect[1], rect[2], rect[3]
    for i in range(int(x_r - height_r/2), int(x_r + height_r/2)):
        rect_area.append((i, int(y_r -length_r/2)))
        rect_area.append((i, int(y_r + length_r / 2)))
    for j in range(int(y_r -length_r/2), int(y_r + length_r/2)):
        rect_area.append((int(x_r - height_r / 2),j))
        rect_area.append((int(x_r + height_r / 2), j))
    circ_area = []
    x,y,r=shape[0],shape[1],shape[2]
    for i in range (0,360):
        deg = i*math.pi/180
        circ_area.append((x+math.cos(deg)*r,y + math.sin(deg)*r))
    #print(rect_area)
    #print(circ_area)
    for dot in circ_area:
        if dot in rect_area:
            return True
    return False

def find_bounces_video(video_path):
    frame_rate = 30
    vid = imageio.get_reader(video_path)
    frame_duration = 1.0 / frame_rate
    bounce_count = 0
    rect= return_rectangle(video_path)
    x_r, y_r, height_r, length_r = rect[0], rect[1], rect[2], rect[3]
    #print(x_r, y_r, height_r, length_r)
    first_frame = 0
    try:
        for frame_idx, frame in enumerate(vid):
            gray = np.mean(frame, axis=2)
            height, length = gray.shape
            #if frame_idx==0:
                #print(height, length)
            shapes = detect_circles(frame, 1)
            #print(shapes)
            for shape in shapes:
                if shape[3] == 0:
                    x = shape[0]
                    y = shape[1]
                    r = shape[2]
                    #print(shape)
                    if ((x - r <= 1) or (x + r >= height-1) or (y - r <= 1) or (y + r >= length-1)):
                        bounce_count += 1
                        if (bounce_count == 1):
                            first_frame = frame_idx+1
                        #print(bounce_count,1)
                    elif check_collision(shape,rect):
                        bounce_count += 1
                        if(bounce_count == 1):
                            first_frame = frame_idx + 1
                        #print(bounce_count,3)
    except:
        print(bounce_count)
    vid.close()
    return first_frame

def estimate_threshold(audio_data, num_std_dev):
    mean = np.mean(audio_data)
    std_dev = np.std(audio_data)
    threshold = mean + num_std_dev * std_dev
    return threshold
'''
def find_peaks_custom(audio_data, threshold):
    peaks = []
    
    return peaks

def count_bounces_custom(audio_data, num_std_dev, min_distance):
    threshold = estimate_threshold(audio_data, num_std_dev)
    peaks = find_peaks_custom(audio_data, threshold)

    # Filter out overlapping peaks
    filtered_peaks = []
    for peak in peaks:
        if not any(abs(peak - other_peak) <= min_distance for other_peak in filtered_peaks):
            filtered_peaks.append(peak)

    # Count the number of remaining peaks
    num_bounces = len(filtered_peaks)
    return num_bounces
'''


def plot_spectrum(signal, sampling_rate):
    # Perform Fourier Transformation
    fft_result = np.fft.fft(sound_data)

    # Calculate the frequencies corresponding to the FFT result
    frequencies = np.fft.fftfreq(len(sound_data))
    #spectrum = np.fft.fft(signal)

    # Calculate Frequency Bins
    #freq_bins = np.fft.fftfreq(len(signal), d = 1 / sampling_rate)

    # Plot the Spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(fft_result))
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
# Detect peaks in the data


# Filter out noise from the data
def find_peaks(sound_data):
    order = 4
    f_sample = 44100
    #f_pass =
    #f_stop = 0.2
    #wp = f_pass / (f_sample / 2)
    #ws = f_stop / (f_sample / 2)
    wp = 0.1
    ws=0.2
    critical_frequencies = (wp, ws)

    # Apply bandpass filter
    b, a = butterworth_bandpass(order, critical_frequencies)
    filtered_signal = lfilter(b, a, sound_data)
    threshold = estimate_threshold(filtered_signal, 3)
    peaks = []
    first_bounce_sound = 0
    for i in range(1, len(sound_data) - 2):
        if (sound_data[i] > sound_data[i-1] and sound_data[i] > sound_data[i+1] and sound_data[i] > threshold) or (sound_data[i] < sound_data[i-1] and sound_data[i] < sound_data[i+1] and sound_data[i] < -1*threshold):
            peaks.append(i)
            if first_bounce_sound==0:
                first_bounce_sound = i
        i+=10000
    print(len(peaks))
    return(first_bounce_sound)

if __name__ == "__main__":
  video_path = input()
  sound_path = input()
  #find_leftmost_circle(video_path)
  sound_data = np.load(sound_path)
  f_sample =  44100
  #plot_spectrum(sound_data, f_sample)
  first_bounce_sound = find_peaks(sound_data)

  #filtered_data = filter_noise(sound_data, N, Wn)

  #peaks_indices = detect_peaks(filtered_data)
  #num_peaks = len(peaks_indices)
  #print(num_peaks)
  #print(audio_data)
  #first_bounce_sound = find_peaks(sound_data)
  #first_bounce_frame = find_bounces_video(video_path)

  #print(first_bounce_frame)
  #start_time = float(first_bounce_frame/30) - float(first_bounce_sound/44100)
  #start_time = start_time*30
  #print(int(start_time))

