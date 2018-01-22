import numpy
import matplotlib.pyplot as plt
import calpy
from calpy.testing.experiments import *
from calpy.entropy.entropy import entropy_profile
#from sklearn import cluster

##########################generate artificial signals###########################
'''
Pipeline 0
'''
lower, upper, N = 75, 255, 3
pitches = list(range(lower, upper, (upper - lower) // N))
sin_waves = list()
fs, time_step = 8000, 0.025

t = numpy.arange(int(fs * time_step))
for f in pitches:
    y = numpy.sin(2 * numpy.pi * f * t / fs)
    sin_waves.append(list(y))



sig_a = []
sig_b = []
sig_c = []

for i in range(N):
    sig_a += sin_waves[i] #rising tone
    sig_b += sin_waves[N - 1 - i] #falling tone
    sig_c += sin_waves[N // 2] #monotonicity

#generate signal according to a distribution

#################################generate pitch#################################
'''
pipe line 1
'''
#set time_step == frame_size, no overlapping
pitch_prof = calpy.dsp.pitch_profile(sig, fs, time_step = 0.025)


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
symbols = []
    for pitch in numpy.array_split(pitch_prof, len(pitches) // N):
        symbols.append( symbolise(pitch) )

################################entropy#########################################
'''
pipe line 3
'''
entropy_durations = [300, 500, 700, 900, 1000, 1100, 1200, 1400]
for dur in entropy_durations:
    ent_prof = entropy_profile(symbols, window_size=dur)

    fig = plt.figure()
    plt.plot(ent_prof)
    plt.title("entropy profile with entropy duration of {}s ({} symbols per entropy)".format(int(dur * N * 0.01), dur))
    fig.savefig("entropy_{}.png".format(dur))