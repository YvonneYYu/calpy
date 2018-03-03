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