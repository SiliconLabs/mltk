"""Audio utilities"""

import librosa
import numpy as np



def melspectrogram_transform(x, window_length, window_stride, n_bins, sample_rate, fmin=0, fmax=None, center=False, **kwargs):
    if not fmax:
        fmax = sample_rate // 2 - 1
    
    n_fft = 1 
    while n_fft < window_length:
        n_fft <<= 1
    
    S = librosa.feature.melspectrogram(y=x, 
                                       sr=sample_rate, 
                                       n_fft=n_fft, 
                                       win_length=window_length,
                                       hop_length=window_stride, 
                                       n_mels=n_bins, 
                                       fmin=fmin, 
                                       fmax=fmax,
                                       center=center,
                                       **kwargs)
    x = librosa.power_to_db(S, ref=np.max)
    x = x.transpose()
    return x 