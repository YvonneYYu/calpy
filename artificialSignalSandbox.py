import numpy
import matplotlib.pyplot as plt
import calpy
from sklearn import cluster
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


def RandSymbols( probs, length ):
    if sum(probs) != 1:
        print("Warning: probabilites must sum to 1")
        return
 
    return numpy.random.choice( len(probs), length, p=probs )
 

 
def RandomRun( length, distributions, min_run=100, max_more=100 ):

    s1 = [0.8, 0.1, 0.1]
    s2 = [0.4, 0.3, 0.3]
    ss = [ s1, s2 ]
    ans  = list()
    k, N, M = 0, length, len(ss)
     
    while True:
        ext_length = min_run + numpy.random.randint(0,max_more)
        ext_length = min( ext_length, N )
         
        ans.extend( RandSymbols( distributions[k % M], ext_length ) )
        k += 1 % M
 
        N -= ext_length
        if N <= 0: return ans

############generate pitch##############################################################################################
'''
pipe line 1
'''
#set time_step == frame_size, no overlapping
pitch_prof_a = calpy.dsp.pitch_profile(sig_a, fs, time_step = 0.025)
pitch_prof_b = calpy.dsp.pitch_profile(sig_b, fs, time_step = 0.025)
pitch_prof_c = calpy.dsp.pitch_profile(sig_c, fs, time_step = 0.025)

t = range(len(pitch_prof_a))
fig0 = plt.figure()
plt.scatter(t, pitch_prof_a, label = 'symbols a rising')
plt.legend()
fig0.show()

fig1 = plt.figure()
plt.scatter(t, pitch_prof_b, label = 'symbols b falling')
plt.legend()
fig1.show()

fig2 = plt.figure()
plt.scatter(t, pitch_prof_c, label = 'symbols c level')
plt.legend()
fig2.show()


#mfcc_prof = calpy.dsp.mfcc_profile(sig, fs, time_step = 0.025)
#mfcc_prof = mfcc_prof.T

############symbolisation###############################################################################################
'''
pipe line 2
'''
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


############entropy#####################################################################################################
'''
pipe line 3
'''
def entropy_profile( symbols, window_size = 100, time_step = 100 ):
    N = len(symbols)
    ent_prof = []
    for k in range((N - window_size) // time_step):
        window = symbols[k * time_step : k * time_step + window_size]
        key, cnts = numpy.unique(window, return_counts=True)
        cnts = cnts / numpy.sum(cnts)
        probs = dict(zip(key,cnts))
        ent_prof.append(-sum(probs[s] * numpy.log(probs[s]) for s in window))
    return numpy.array(ent_prof)

ent_prof = entropy_profile(symbols)
 
############visualisation###############################################################################################
fig = plt.figure()
plt.plot(ent_prof)
fig.show()