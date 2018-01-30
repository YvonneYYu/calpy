import os

import numpy
import matplotlib.pyplot as plt
import calpy
from experiments import random_symbols, symbolise
from calpy.entropy import entropy_profile
#from sklearn import cluster

path = './entropy_experiment0_result/'
if not os.path.isdir(path):
    os.makedirs(path)

##########################generate artificial signals###########################
#Pipeline 0

fs, time_step = 8000, 0.025
lower, upper, N = 75, 255, 3

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
numpy.save(path + "signal", sig)
numpy.save(path + "symbols", symbols)

#symbols = numpy.load(path + "symbols.npy")
#sig = numpy.load(path + "signal.npy")

#################################generate pitch#################################
#pipe line 1

#set time_step == frame_size, no overlapping

pitch_prof = calpy.dsp.pitch_profile(sig, fs, time_step = 0.025)
numpy.save(path + "pitch_prof", pitch_prof)

#pitch_prof = numpy.load(path + "pitch_prof.npy")

#############################symbolisation######################################
#pipe line 2

##########################OLD Machine Learning Staff############################
#def hist_N_median_profile( xs, num_bins=3, window_size=10 ):                  #
#    N, res = len(xs), []                                                      #
#    edges = (xs.min(),xs.max())                                               #
#    for k in range(N-window_size):                                            #
#        window = xs[k : k+window_size]                                        #
#        hist   = numpy.histogram(window,range=edges,bins=num_bins)            #
#        median = numpy.median(window)                                         #
#        res.append(list(hist[0]) + [median])                                  #
#    return numpy.array(res)                                                   #
#                                                                              #
#feature_pitch = hist_N_median_profile(pitch_prof)                             #
#db_pitch = cluster.DBSCAN(eps=0.1, min_samples=20).fit(feature_pitch)         #
#symbols = db_pitch.labels_                                                    #
################################################################################

symbols_hat =numpy.array([])
for pitch in numpy.array_split(pitch_prof, len(pitch_prof) // N):
    symbols_hat = numpy.append( symbols_hat, symbolise(pitch) )

numpy.save(path + "symbols_hat", symbols_hat)

#symbols_hat = numpy.load(path + "symbols_hat.npy")
################################entropy#########################################
#pipe line 3

entropy_durations = [200, 500, 1000, 2000, 3000]

#H1 = -sum(x * numpy.log(x) for x in p1)
#H2 = -sum(x * numpy.log(x) for x in p2)
#H3 = -sum(x * numpy.log(x) for x in p3)
#H4 = -sum(x * numpy.log(x) for x in p4)
#H = [H1, H2, H1, H3, H1, H4, H1]

for dur in entropy_durations:
    overlap = 2 * dur // 3
    #ent_prof = []
    #for i in range(len(ls)):
    #    ent_prof += [ H[i] ] * (ls[i] // dur)
    ent_prof_hat = entropy_profile(symbols_hat, window_size=dur,window_overlap=overlap)

    fig = plt.figure(num=None, figsize=(14, 6), dpi=150)
    t = list(range(len(ent_prof_hat)))
    #plt.plot(t, ent_prof, '+-', label="true entropy")
    plt.plot(t, ent_prof_hat, '*-', label="model calculated entropy")
    plt.title("model calibration with entropy duration of {}s ({} symbols per entropy) and overlap window of {} symbols".format(int(dur * N * 0.01), dur, overlap))
    #plt.legend()
    fig.savefig(path + "entropy_{}_{}.png".format(dur, overlap))