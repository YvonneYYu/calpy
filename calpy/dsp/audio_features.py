import numpy
from scipy.fftpack import dct
from .yin import *
from ..utilities import utilities

def _silence_or_sounding(signal, eps=1e-8):
    """
       Determine silence and sounding of a given (usually a relatively long period of time) audio
       Reference:
       @inproceedings{Sakhnov,
            title={Dynamical energy-based speech/silence detector for speech enhancement applications},
            author={Sakhnov, Kirill and Verteletskaya, Ekaterina and Simak, Boris},
            booktitle={Proceedings of the World Congress on Engineering},
            volume={1},  pages={2},  year={2009},  organization={Citeseer}
            }
        input:
            signal :: sound signal in time domain :: 1D numpy float array or a 1D float list
            eps  :: The minimum threshold :: a float number, default 1e-8. 
        output:
            marker: a 0-1 list marking silence (0) and sounding (1)
    """
    
    N = len(signal)
    signal = signal ** 2
    
    e_max, e_min = signal[0] if signal[0]>eps else 2*eps, eps

    marker = list()
    lamda  = (e_max - e_min)/e_max
    threshold  = (1 - lamda) * e_max + lamda * e_min

    marker.append(1 if signal[0]>threshold else 0)

    for i in range(1, N):
        #YYY modified the original equation from engy = signal[i] to the following.
        #YYY doesn't know why, bu it works more acurately than the original.
        #Happy accident
        engy = signal[i] ** 2
        if engy > e_max:
            e_max = engy
        elif engy < e_min:
            e_min = eps if engy <= eps else engy
        #thresholding
        lamda = (e_max - e_min) / e_max
        threshold = (1 - lamda) * e_max + lamda * e_min
        marker.append(1 if engy > threshold else 0)
        
    return marker

def pause_profile(signal, sampling_rate, min_silence_duration=0.01, time_step = 0.01, frame_window = 0.025):
    """
        Determine pause sin a long audio (usually an entire covnersation)
        input:
            signal: audio signal :: 1D numpy float array or a 1D float list
            sampling_rate: sampling frequency in Hz :: a float number
            t: in second, the minimum time duration to be considered pause :: a float, Default 0.01
        output:
            ans: 0-1 1D numpy integer array with 1s marking pause
    """
    
    signal = signal / max(abs(signal))
    signal = _silence_or_sounding(signal)
    signal = numpy.array(signal)
    
    N = len(signal)
    ans = numpy.zeros(N)

    i, count, start, end = 0, 0, 0, 0

    T = min_silence_duration * sampling_rate
    for i in range(N):
        if signal[i] == 0:
            if i == N-1 and signal[i-1] == 0 and count >= T:
                ans[start:end] = 1
            elif count==0:
                start, end, count = i, i+1, count+1
            else:
                end, count = end+1, count+1
        elif i > 0 and signal[i-1] == 0:
            if count >= T:  
                ans[start:end] = 1
            count = 0
    return utilities.compress_pause_to_time(ans, sampling_rate, time_step=time_step, frame_window=frame_window)

def dB_profile(signal, sampling_rate, time_step = 0.01, frame_window = 0.025):
    """
        Computes decible of signal amplitude of an entire conversation
        input:
            signal: the padded audio signal :: 1D numpy float array or a float 1D list
            sampling_rate: sampling frequency in Hz :: a float
            time_step: the temporal step :: a float number, Default is 0.01 second
            frame_window: the frame window :: a float number, Default is 0.025 second
        output:
            dB: the decible :: 1D float list
    """
    signal = numpy.abs(signal)
    
    T  = int(sampling_rate * time_step)
    Fr = int(sampling_rate * frame_window)
    Fr += (sampling_rate * frame_window - Fr) > 0
    
    N = (len(signal) - Fr) // T + 1

    dB = numpy.empty(N)
    
    #use mean over the entire converstaion as reference signal
    #should avoid using square opertion on row signal in case of overflow
    
    ref = 20*numpy.log(numpy.mean(signal[signal>0]))
    for i in range(N):
        dB[i] = sum(signal[i*Fr:(i+1)*Fr])/(Fr-1)

    vfunc = numpy.vectorize(lambda x: -float('inf') if not x else numpy.log(x) - numpy.log(ref))
    return vfunc(dB)

def pitch_profile(signal, sampling_rate, time_step = 0.01, frame_window = 0.025, lower_threshold = 75, upper_threshold = 255):
    """
        Compute pitch for a long (usually over an entire conversation) sound signal
        input:
            signal: padded audio signal :: 1D numpy float array or 1D float list
            sampling_rate: sampling frequency in Hz :: a float number
            time_step: temporal step in second :: a float, Default 0.01 second
            frame_window: frame window :: a float, Default 0.025 second
            lower: lower limit of pitch in Hz. Pitchs below this limit are not recruited :: a float, Default 75
            upper: upper limit of pitch in Hz. Pitchs above this limit are not recruited :: a float, Default 250
        output:
            p: estimated pitch in Hz :: a 1D float list
    """
    
    T  = int(sampling_rate * time_step)
    Fr = int(sampling_rate * frame_window)
    Fr += sampling_rate * frame_window - Fr > 0
    
    N = (len(signal) - Fr) // T + 1

    if not N:
        print("Warning: not enough signal, pitch profile will be empty.")
    
    p = numpy.empty( N )
    for i in range(N):
        p[i] = instantaneous_pitch(signal[i*T:i*T+Fr], sampling_rate)
    p[numpy.where( (p > upper_threshold) | (p < lower_threshold) )] = 0
    
    return p

def mfcc_profile(signal, sampling_rate, time_step = 0.01, frame_window = 0.025, NFFT = 512, nfilt = 40, ceps = 12):
    """
        Author: YYY
        Compute MFCC for a long (usually over an entire conversation) sound signal
        reference: http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
        input:
            signal: audio signal :: 1D numpy float array or 1D float list
            sampling_rate: sampling frequency in Hz :: a float number
            time_step: temporal step in second :: a float, Default 0.01 second
            frame_window: frame window :: a float, Default 0.025 second
            NFFT: NFFT-point FFT, default value is 512
            nfilt: number of frequency bands in Mel-scaling, default value is 40
            ceps: number of mel frequency ceptral coefficients to be retained, default value is 12
        output:
            mfccs: calculated Mel-Frequecy Cepstral Coefficients :: a 2D numpy array
    """
    
    T  = int(sampling_rate * time_step)
    Fr = int(sampling_rate * frame_window)
    Fr += sampling_rate * frame_window - Fr > 0
    N = (len(signal) - Fr) // T + 1
    res = numpy.empty((ceps, N))
     
    if not N:
        print("Warning: not enough signal, mfcc profile will be empty.")
 
    #pre-calculate routine filter bank array
    low_mel, high_mel = 0, 2595 * numpy.log10(1 + sampling_rate / 1400)
    # Mel points
    mel_pts = numpy.linspace(low_mel, high_mel, nfilt + 2)
    # Corresponding Hz points
    hz_pts = (10 ** (mel_pts / 2595) - 1) * 700
    hz_pts = numpy.floor((NFFT + 1) * hz_pts / sampling_rate)
    # filter bank array
    fltBank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for i in range(1, nfilt + 1):
        left, mid, right = int(hz_pts[i - 1]), int(hz_pts[i]), int(hz_pts[i + 1])
        for k in range(left, mid):
            fltBank[i - 1, k] = (k - hz_pts[i - 1]) / (hz_pts[i] - hz_pts[i - 1])
        for k in range(mid, right):
            fltBank[i - 1, k] = (hz_pts[i + 1] - k) / (hz_pts[i + 1] - hz_pts[i])
    fltBank = fltBank.T
 
    for i in range(N):
        frame = signal[i * T : i * T + Fr]
        # applies a hamming window before STFT, optional but highly recommended
        frame *= numpy.hamming(Fr)
        # power spectrum of FFT
        pow_frame = (1.0 / NFFT) * (numpy.absolute(numpy.fft.rfft(frame, NFFT)) ** 2)
        # filter bank it
        filter_bank = numpy.dot(pow_frame, fltBank)
        # special process of 0 points
        filter_bank = numpy.where(filter_bank == 0, numpy.finfo(float).eps, filter_bank)
        #scale to dB
        filter_bank = 20 * numpy.log10(filter_bank)
        #mfcc
        mfcc = dct(filter_bank, norm='ortho')
        res[:, i] = mfcc[1 : ceps + 1]
 
    return res