import numpy
import matplotlib.pyplot as plt
import calpy
from calpy.testing.experiments import random_symbols
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
sig_d = []
for i in range(N):
    sig_a += sin_waves[i] #rising tone
    sig_b += sin_waves[N - 1 - i] #falling tone
    sig_c += sin_waves[N // 2] #monotonicity

sig_d = [0] * len(sig_a) #silence/pause

#generate signal according to a distribution symbols [0, 1, 2, 3] rising, falling, level, silent
p1 = [0.2, 0.2, 0.5, 0.1] # assumed as norm
p2 = [0.4, 0.15, 0.375, 0.075] # increase rising
p2 = [0.15, 0.4, 0.375, 0.075] # increase falling
p3 = [0.18, 0.18, 0.44, 0.2] # increase pause
l_norm = 6000 # length of symbols list of normal speech
l_abnorm = 2400 # length of symbols list of abnormal speech
ps = [p1, p2, p1, p3, p1, p4, p1]
ls = [l_norm, l_abnorm, l_norm, l_abnorm, l_norm, l_abnorm, l_norm]

symbols = []
for i in range(len(ps)):
    symbols += random_symbols(ps[i], ls[i])
print(type(symbols), len(symbols))
keyboard
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
symbols_hat = []
for pitch in numpy.array_split(pitch_prof, len(pitches) // N):
    symbols_hat.append( symbolise(pitch) )

################################entropy#########################################
'''
pipe line 3
'''
entropy_durations = [300, 500, 700, 900, 1000, 1100, 1200]
for dur in entropy_durations:
    ent_prof = entropy_profile(symbols, window_size=dur)
    ent_prof_hat = entropy_profile(symbols_hat, window_size=dur)

    fig = plt.figure()
    plt.plot(ent_prof, label="true entropy")
    plt.plot(ent_prof_hat, label="model calculated entropy")
    plt.title("model calibration with entropy duration of {}s ({} symbols per entropy)".format(int(dur * N * 0.01), dur))
    fig.legend()
    fig.savefig("entropy_{}.png".format(dur))