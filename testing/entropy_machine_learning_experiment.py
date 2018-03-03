import numpy
from calpy.dsp import mfcc_profile
from calpy.utilities import read_wavfile
from calpy.entropy import entropy_profile
from scipy.stats.mstats import zscore

import os

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

def entropys(symbols, entropy_durations, overlap_factors):
    for dur in entropy_durations:
        for overlap_factor in overlap_factors:
            overlap = int(overlap_factor * dur)
            vfun = lambda x : entropy_profile(x, window_size=dur, window_overlap=overlap)
            ent_prof = numpy.apply_along_axis(vfun, 1, symbols)
            t = numpy.arange(ent_prof.shape[1]) * dur * 0.01 * (1 - overlap_factor)
            t = t.reshape(1, ent_prof.shape[1])
            ent_prof = numpy.append(t, ent_prof, axis=0)
            numpy.save(path + "entropy_with_time_info{}_{:.1f}".format(dur, overlap_factor), ent_prof)

def estimate_Gaussian(X):
    """Estimate the parametres of a Gaussian distribution using X

    Args:
        X (numpy.array (float)): Training dataset with features along axis 0, and examples along axis 1.

    Returns:
        Mu (numpy.array (float)): mean of the X (n by 1 dimension).
        Sigma2 (numpy.array (float)): variance of X (n by 1 dimension).
    """

    Mu = numpy.mean(X, axis=1).reshape(X.shape[0], 1)
    Sigma2 = numpy.var(X. axis=1).reshape(X.shape[0], 1)
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
    
path = './entropy_machine_learning_results/'
if not os.path.isdir(path):
    os.makedirs(path)

fname = 'audiodump_8000'

fs, sound = read_wavfile(fname + '.wav')
mfcc = mfcc_profile(sound, fs)
#numpy.save(path + 'mfcc', mfcc)
symbols = symbolise_mfcc_multidimension(mfcc)
#numpy.save( path + 'symbols', symbols)

entropy_durations = [300, 400, 500, 1000]
overlap = [0, 0.5, 0.9]
entropys(symbols, entropy_durations, overlap)