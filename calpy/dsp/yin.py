import numpy

def difference_function(signal):
    """Calculate difference function of the signal.  Step 1 and 2 of `YIN`_.
        
        Args:
            signal (:obj:`numpy.array(float)`):  A short audio signal. 1D array.
        
        Returns:
            :obj:`numpy.array(float)`: Equation (6) of `YIN`_. The difference function d(t, tau).  1D array.

        .. _YIN:
            https://www.ncbi.nlm.nih.gov/pubmed/12002874
    """
    N = len(signal)
    d = numpy.zeros(N//2+1)
    y = numpy.copy(signal)

#    OLD CODE
#    for tau in range(N//2):
#        y = numpy.roll(y,-1)
#        z = signal-y
#        d[tau+1] = numpy.dot(z, z)
#    return d
    
    x2 = numpy.dot(signal,signal)
    for tau in range(N//2):
        tmp = y[0]
        y[:-1] = y[1:]
        y[-1] = tmp
        d[tau+1] = 2*x2 - 2*numpy.dot(signal, y)

    return d

def normalisation(signal):
    """Normalise the difference function by the cumulative mean.  Step 3 of `YIN`_.
        
        Args:
            signal (:obj:`numpy.array(float)`): A small piece of self correlated audio signal d(t, tau) processed by difFunction(). 1D array.

        Returns:
            :obj:`numpy.array(float)`: Equation (8) of `YIN`_. Normalised difference function d'(t, tau). 1D array.
    """
    N, signal[0] = len(signal), 1
    
    tmp = 0
    for tau in range(1,N):
        tmp += signal[tau]
        signal[tau] *= tau/tmp if tmp else 0

    return signal

def absolute_threshold(signal, threshold):
    """ Absolute thresholdeshold. Step 4 in `YIN`_.
        
        Args:
            signal (:obj:`numpy.array(float)`): A small piece normalised self correlated audio d'(t, tau) processed by normalisation(). 1D array like.
            threshold (float): Threshold value.
        
        Returns:
            float: The index tau.
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
    """Parabolic Interpolation on tau.  Step 5 in `YIN`_.
        
        Args:
            signal (:obj:`numpy.array(float)`): A small piece normalised self correlated audio d'(t, tau) processed by normalisation(). 1D array.
            tau (int): Estimated thresholdeshold.
        
        Returns:
            float: A better estimation of tau.
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

#@profile
def instantaneous_pitch(signal, sampling_frequency, threshold=0.1):
    """Computes fundamental frequency (based on `YIN`_) as pitch of a given (usually a very short) time interval.
        
        Code is an adpation of  https://github.com/ashokfernandez/Yin-Pitch-Tracking.
        
        Args:
            signal (:obj:`numpy.array(float)`): Audio signal. 1D array.
            sampling_frequency (int): Sampling frequency in Hz.
            threshold (float,optional): Absolute thresholdeshold value as defined in Step 4 of `YIN`_. Default 0.1
        
        Returns:
            f0: fundamental frequency in Hz (estimated speech pitch), a float
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
