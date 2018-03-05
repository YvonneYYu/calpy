import numpy
from scipy.stats.mstats import zscore

def entropy_profile( symbols, window_size = 100, window_overlap = 0 ):
    """Calculate the entropy profile of a list of symbols.

    Args:
        symbols (numpy.array or list (int)):  A list of symbols.
        window_size (int, optional):  Number of symbols per entropy window.  Defaults to 100.
        window_overlap (int, optional):  How much the entropy windows should overlap.  Defaults to 0.
    
    Returns:
        numpy.array(float): The entropy profile.
    """
    N = len(symbols)
    ent_prof  = []
    time_step = window_size - window_overlap

    for k in range((N - window_size) // time_step + 1):
        window = symbols[k * time_step : k * time_step + window_size]
        key, cnts = numpy.unique(window, return_counts=True)
        cnts = cnts / numpy.sum(cnts)
        probs = dict(zip(key,cnts))
        ent_prof.append(-sum(probs[s] * numpy.log(probs[s]) for s in window))
    return numpy.array(ent_prof)

def entropy_profile_2D(symbols, window_size = 100, window_overlap = 0):
    """Calculate the 2D entropy profile of symbols (typically mfcc with axis 1 as time).

    Args:
        symbols (2D numpy.array (int)): Symbols of 2 dimensions.
        window_size (int, optional):  Number of symbols per entropy window.  Defaults to 100.
        window_overlap (int, optional):  How much the entropy windows should overlap.  Defaults to 0.
    
    Returns:
        2D numpy.array(float): The 2D entropy profile.
    """
    vfun = lambda x : entropy_profile(x, window_size, window_overlap)
    ent_prof = numpy.apply_along_axis(vfun, 1, symbols)
    
    return ent_prof

def estimate_Gaussian(X):
    """Estimate the parametres of a Gaussian distribution using X

    Args:
        X (numpy.array (float)): Training dataset with features along axis 0, and examples along axis 1.

    Returns:
        Mu (numpy.array (float)): mean of the X (n by 1 dimension).
        Sigma2 (numpy.array (float)): variance of X (n by 1 dimension).
    """

    Mu = numpy.mean(X, axis=1).reshape(X.shape[0], 1)
    Sigma2 = numpy.var(X, axis=1).reshape(X.shape[0], 1)
    return (Mu, Sigma2)


def multivariate_Gaussion(X, Mu, Sigma2):
    """Computes the probability density function of multivariate Gaussian distribution.

    Args:
        X (1D numpy.array (float)): n by 1 feature vector.
        Mu (1D numpy.array (float)): n by 1 mean vector.
        Sigma2 (1D numpy.array (float)): n by 1 variance vector.

    Returns:
        p (float): probability of input X.
    """
    assert X.shape == Mu.shape, "Input X and Mu must be the same shape"
    assert Mu.shape == Sigma2.shape, "Input Mu and Sigma2 must be the same shape"
    Sigma2 = numpy.diagflat(Sigma2)
    Sigma2_inv = numpy.linalg.inv(Sigma2)
    k = X.shape[0]
    p = 1 / numpy.sqrt( (2 * numpy.pi) ** k * numpy.linalg.det(Sigma2) )
    exp_power = -0.5 * numpy.dot( numpy.dot( (X - Mu).T, Sigma2_inv ), (X - Mu) )
    p *= numpy.exp(exp_power)

    return p

def symbolise( pitches, eps=8e-2 ):
    """Symbolose a small piece of speech segment according to pitch slopes.

    Args:
        pitches (numpy.array or list (float)):  A list of pitches.
        eps (float, optional):  Treshold of pitch slopes to be considered level.  Defaults to tan(5 degrees).

    Returns:
        int: one symbol of the small piece of speech segment.
    """
    #if input pitches are all 0.0, categorise it as silent (3)
    if not pitches.any():
        return 3

    N  = len(pitches)
    xs = numpy.arange(N)
    ys = pitches

    AA = numpy.vstack([xs, numpy.ones(N)]).T
    m, c = numpy.linalg.lstsq(AA, ys)[0]

    #return m

    if m>eps:
        return 0 #rising
    elif -eps <= m <= eps:
        return 2 # level
    else:
        return 1 # falling

def symbolise_speech(pitches, pauses, thre=250):
    """Symbolise a small speech segment according to pitch and pause.

    Args:
        pitches (numpy.array(float)):  A list of pitches.
        pauses (numpy.array(int)):  A list of pauses.
        thre (float, optional): Threshold of high pitch.

    Returns:
        int: one symbol of the small speech segment.
    """
    #if all silent
    if pauses.all():
        return 1
    #high pitch
    if numpy.average(pitches) > thre:
        return 2
    #normal pitch
    return 0

def symbolise_mfcc(mfcc):
    """Symbolise speech according to mfcc.

    Args:
        mfcc (2D numpy.array (float)): A list of mfcc, axis 1 is time and axis 0 is mfcc

    Returns:
        symbols (numpy.array (float)): A list of symbols.
    """
    symbols = numpy.zeros(mfcc.shape[1])
    mu = numpy.average(mfcc, axis=1)
    low_mu, med_mu, high_mu = numpy.average(mu[:4]), numpy.average(mu[4:8]), numpy.average(mu[8:12])
    for i, mf in enumerate(mfcc.T):
        symbols[i] = symbols[i] * 2 +  int(numpy.average(mf[8:]) > high_mu)
        symbols[i] = symbols[i] * 2 + int(numpy.average(mf[4:8]) > med_mu)
        symbols[i] = symbols[i] * 2 +int(numpy.average(mf[:4]) > low_mu)
    return symbols

def symbolise_mfcc_multidimension(mfcc):
    """Symbolise speech in multi-dimensional scale according to mfcc.

    Args:
        mfcc (2D numpy.array (float)): A list of mfcc, axis 1 is time and axis 0 is mfcc

    Returns:
        symbols (2D numpy.array(int)): Multi-dimensional symbols. A 2D numpy.array with the same shape as input mfcc
    """
    mfcc = zscore(mfcc, axis=1)
    symbols = numpy.zeros(mfcc.shape)
    symbols[numpy.where( numpy.abs(mfcc) <= 1 )] = 1
    symbols[numpy.where( mfcc > 1)] = 2

    return symbols