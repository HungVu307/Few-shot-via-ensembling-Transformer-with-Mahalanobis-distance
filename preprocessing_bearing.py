import numpy as np
import cv2

def gaussian_filter(data, sigma):
    kernel_size = int(6 * sigma + 1)
    kernel_size += 1 if kernel_size % 2 == 0 else 0  # Ensure kernel size is odd
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-kernel_size//2)**2 + (y-kernel_size//2)**2)/(2*sigma**2)),
        (kernel_size, kernel_size)
    )
    kernel /= np.sum(kernel)
    return cv2.filter2D(data, -1, kernel)

def stft(signal, n_fft, hop_length):
    window = np.hanning(n_fft)
    spectrogram = np.abs(np.array([np.fft.fft(window * signal[j:j+n_fft]) for j in range(0, len(signal)-n_fft+1, hop_length)]))
    spectrogram = spectrogram[:, :n_fft//2]  
    return spectrogram

def power_to_db(spectrogram):
    return 10 * np.log10(spectrogram)

def resize(data, target_height, target_width):
    return cv2.resize(data, (target_height, target_width))

def to_spectrum(data, height=64, width=64, sigma=0.6):
    spectrograms = []

    for i in range(data.shape[0]):
        signal = data[i, :]
        signal = np.array(signal)
        n_fft = 512
        hop_length = 512
        spectrogram = stft(signal, n_fft, hop_length)
        spectrogram = spectrogram ** 2
        log_spectrogram = power_to_db(spectrogram)
        log_spectrogram = resize(log_spectrogram, height, width)
        smoothed_spectrogram = gaussian_filter(log_spectrogram, sigma=sigma)

        spectrograms.append(smoothed_spectrogram)

    data = np.stack(spectrograms).astype(np.float32)
    data = np.expand_dims(data, axis=1)
    return data