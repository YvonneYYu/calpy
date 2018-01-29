import numpy

def difference_function(signal):
    """
        Calculate difference function of the signal.
        Step 1 and 2 of YIN
        Input:
            signal: a small piece of audio signal :: 1D numpy float array
        Output:
            d: Equation (6) of YIN. The difference function d(t, tau) :: 1D float list
    """
    N = len(signal)
    d = numpy.zeros(N//2+1)
    y = signal
    
    for tau in range(N//2):
        y = numpy.roll(y,-1)
        z = signal-y
        d[tau+1] = numpy.dot(z, z)
    
    return d

def normalisation(signal):
    """
        Normalise the difference function by the cumulative mean.
        Step 3 of YIN
        Input:
            signal: a small piece of self correlated audio signal d(t, tau) processed by difFunction() :: 1D float list
            N: length of audio signals to be analysed
        Output:
            signal: Equation (8) of YIN. Normalised difference function d'(t, tau), 1D list
    """
    N, signal[0] = len(signal), 1
    
    tmp = 0
    for tau in range(1,N):
        tmp += signal[tau]
        signal[tau] *= tau/tmp if tmp else 0

    return signal

def absolute_threshold(signal, threshold):
    """
        Absolute thresholdeshold
        Step 4 in YIN
        Input:
            signal: a small piece normalised self correlated audio d'(t, tau) processed by normalisation(). 1D array like
            threshold: thresholdeshold value
        Output:
            tau: the index
    """
    #since signal[0] == 1, which is definitely greater than threshold, we start the search from the second element
    
    N   = len(signal)
    MIN = numpy.min(signal[1:])
    threshold = MIN + threshold*(numpy.max(signal[1:])-MIN)
    
    for tau in range(1, int(N)):
        if signal[tau] < threshold:
            while tau + 1 < N and signal[tau+1] < signal[tau]:
                tau += 1
            break
    
    if tau == N or signal[tau] >= threshold:
        tau = -1
    
    return tau

def parabolic_interpolation(signal, tau):
    """
        Parabolic Interpolation on tau
        Step 5 in YIN
        Input:
            signal: a small piece normalised self correlated audio d'(t, tau) processed by normalisation(). 1D array like
            N: length of audio signal to be analysed
            tau: estimated tau in thresholdeshold(). An integer
        Output:
            estTau: better estimation of tau
    """
    
    N, tau = len(signal), int(tau)

    x1 = tau if tau < 1 else tau-1
    x2 = tau if tau+1 >= N/2 else tau+1
    
    if x1 == tau:
        return tau if signal[tau] <= signal[x2] else x2
    elif x2 == tau:
        return tau if signal[tau] <= signal[x1] else x1
    else:
        s0, s1, s2 = signal[x1], signal[tau], signal[x2]
        return tau if 2 * s1 - s2 - s0 == 0 else tau + (s2 - s0) / (2 * (2 * s1 - s2 - s0))

def instantaneous_pitch(signal, sampling_frequency, threshold=0.1):
    """
        Computes fundamental frequency (based on YIN) as pitch of a given
        (usually a very short) time interval.
        Signal Processing Reference:
        @article{YIN,
            title={YIN, a fundamental frequency estimator for speech and music},
            author={De Cheveign{\'e}, Alain and Kawahara, Hideki},
            journal={The Journal of the Acoustical Society of America},
            volume={111},
            number={4},
            pages={1917--1930},
            year={2002},
            publisher={ASA}
            }
        C++ Code Reference (with significant changes):
        https://github.com/ashokfernandez/Yin-Pitch-Tracking
        Input:
            signal: audio signal, 1D array like
            fs: sampling frequency in Hz
            threshold: absolute thresholdeshold value as defined in Step 4 of YIN. Default 0.1
        Output:
            f0: fundamental frequency in Hz (estimated speech pitch), a float
            p: probability
    """
    
    N, tau = len(signal), -1

    signal = difference_function(signal)
    signal = normalisation(signal)

    tau = absolute_threshold(signal, threshold)
    if tau != -1:
        ans = sampling_frequency / parabolic_interpolation(signal, tau)
    else:
        ans = 0

    return ans
