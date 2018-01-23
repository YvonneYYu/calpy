"""
Experiments on single frequency sinusoidal signals
"""

import numpy
import matplotlib.pyplot as plt
import calpy
from calpy.testing.experiments import random_symbols, symbolise
from calpy.entropy.entropy import entropy_profile
#from sklearn import cluster

##########################generate artificial signals###########################
'''
Pipeline 0
'''
fs, time_step = 8000, 0.025
lower, upper, N = 75, 255, 3

"""
pitches = list(range(lower, upper, (upper - lower) // N))
sin_waves = list()
t = numpy.arange(int(fs * time_step))
for f in pitches:
    y = numpy.sin(2 * numpy.pi * f * t / fs)
    sin_waves.append(list(y))

sig_a = numpy.array([])
sig_b = numpy.array([])
sig_c = numpy.array([])
for i in range(N):
    sig_a = numpy.append(sig_a, sin_waves[i]) #rising tone
    sig_b = numpy.append(sig_b, sin_waves[N - 1 - i]) #falling tone
    sig_c = numpy.append(sig_c, sin_waves[N // 2]) #monotonicity

sig_d = numpy.array([0] * len(sig_a)) #silence/pause

#generate signal according to a distribution symbols [0, 1, 2, 3] rising, falling, level, silent
p1 = [0.2, 0.2, 0.5, 0.1] # assumed as norm
p2 = [0.4, 0.15, 0.375, 0.075] # increase rising
p3 = [0.15, 0.4, 0.375, 0.075] # increase falling
p4 = [0.13, 0.13, 0.34, 0.4] # increase pause
l_norm = 4000 # length of symbols list of normal speech
l_abnorm = 2000 # length of symbols list of abnormal speech
ps = [p1, p2, p1, p3, p1, p4, p1]
ls = [l_norm, l_abnorm, l_norm, l_abnorm, l_norm, l_abnorm, l_norm]

symbols = numpy.array([])
for i in range(len(ps)):
    symbols = numpy.append(symbols, random_symbols(ps[i], ls[i]))
symbols = symbols.astype(int)
sig = numpy.array([])
for symbol in symbols:
    if symbol == 0:
        sig = numpy.append(sig, sig_a)
    elif symbol == 1:
        sig = numpy.append(sig, sig_b)
    elif symbol == 2:
        sig = numpy.append(sig, sig_c)
    else:
        sig = numpy.append(sig, sig_d)
numpy.save("signal", sig)
numpy.save("symbols", symbols)
"""
symbols = numpy.load("symbols.npy")
sig = numpy.load("signal.npy")

#################################generate pitch#################################
'''
pipe line 1
'''
#set time_step == frame_size, no overlapping
"""
pitch_prof = calpy.dsp.pitch_profile(sig, fs, time_step = 0.025)
numpy.save("pitch_prof", pitch_prof)
"""

pitch_prof = numpy.load("pitch_prof.npy")

#mfcc_prof = calpy.dsp.mfcc_profile(sig, fs, time_step = 0.025)
#mfcc_prof = mfcc_prof.T

#############################symbolisation######################################
'''
pipe line 2

def hist_N_median_profile( xs, num_bins=3, window_size=10 ):
    N, res = len(xs), []
    edges = (xs.min(),xs.max())
    for k in range(N-window_size):
        window = xs[k : k+window_size]
        hist   = numpy.histogram(window,range=edges,bins=num_bins)
        median = numpy.median(window)
        res.append(list(hist[0]) + [median])
    return numpy.array(res)

feature_pitch = hist_N_median_profile(pitch_prof)
db_pitch = cluster.DBSCAN(eps=0.1, min_samples=20).fit(feature_pitch)
symbols = db_pitch.labels_
'''


symbols_hat =numpy.array([])
for pitch in numpy.array_split(pitch_prof, len(pitch_prof) // N):
    symbols_hat = numpy.append( symbols_hat, symbolise(pitch) )

numpy.save("symbols_hat", symbols_hat)
################################entropy#########################################
'''
pipe line 3
'''
entropy_durations = [300, 500, 700, 900, 1000, 1100, 1200, 2000]
for dur in entropy_durations:
    ent_prof = entropy_profile(symbols, window_size=dur)
    ent_prof_hat = entropy_profile(symbols_hat, window_size=dur)

    fig = plt.figure()
    t = list(range(len(ent_prof)))
    plt.plot(t, ent_prof, '+-', label="true entropy")
    plt.plot(t, ent_prof_hat, '*-', label="model calculated entropy")
    plt.title("model calibration with entropy duration of {}s ({} symbols per entropy)".format(int(dur * N * 0.01), dur))
    plt.legend()
    fig.savefig("entropy_{}.png".format(dur))
    numpy.save("entropy_{}".format(dur), ent_prof)
    numpy.save("entropy_hat_{}".format(dur), ent_prof_hat)