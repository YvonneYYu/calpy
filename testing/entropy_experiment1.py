"""
Experiment on simple speech signals
"""
import numpy
from scipy.stats import zscore
import matplotlib.pyplot as plt
from experiments import symbolise_speech as symbolise
from calpy.dsp import pitch_profile, pause_profile
from calpy.entropy import entropy_profile
from calpy.utilities import read_wavfile
import os

path = "./entropy_experiment1_result/"
if not os.path.isdir(path):
    os.makedirs(path)

def getPauseThred(pauses):
    pause_length = numpy.array([])
    cnt = 0
    for pause in pauses:
        if pause:
            cnt += 1
        elif cnt > 0:
            pause_length = numpy.append(pause_length, cnt)
            cnt = 0
    pause_z = zscore(pause_length)
    thred = numpy.min(pause_length[numpy.where(pause_z >= 0.7)]).astype(int)
    return thred

def getPitchThred(pitches):
    pitches_diff = numpy.diff(pitches)
    pitches_diff = numpy.abs(pitches_diff)
    pithces_diff_z = zscore(pitches_diff)
    thred = numpy.min(pitches_diff[numpy.where(pithces_diff_z >= 0.1)])
    return thred

def symbolisation(feature, pause_thred, pitch_thred):
    N = numpy.ceil(feature.shape[1] / pause_thred).astype(int)
    symbols =numpy.empty(N)
    for idx, data in enumerate(numpy.array_split(feature, N, axis = 1)):
        symbols[idx] = symbolise(data[0], data[1], eps=pitch_thred)
    return symbols

def entropys(symbols, entropy_durations, overlap_factor, N):
    for dur in entropy_durations:
        overlap = int(overlap_factor * dur)
        ent_prof = entropy_profile(symbols, window_size=dur, window_overlap=overlap)

        fig = plt.figure()
        plt.plot( ent_prof, '+-')
        plt.title("model calibration with entropy duration of {}s ({} symbols per entropy) and  factor of {}".format(int(dur * N * 0.01), dur, overlap_factor))
        fig.savefig(path + "entropy_{}_{}.png".format(dur, overlap_factor))
        numpy.save(path + "entropy_{}_{}".format(dur, overlap_factor), ent_prof)
        return ent_prof

'''
fs, sound = read_wavfile('PaulTroubled_8000.wav')
print('sound read in')
pitches = pitch_profile(sound, fs)
print('got pitches')
pauses = pause_profile(sound, fs)
print('got pauses')
feature = numpy.vstack((pitches, pauses))
pause_thred = getPauseThred(pauses)
print('pause_thred is: {}'.format(pause_thred))
pitch_thred = getPitchThred(pitches)
print('pitch_thred is: {}'.format(pitch_thred))
symbols = symbolisation(feature, pause_thred=pause_thred, pitch_thred=pitch_thred)
print('got symbols')

numpy.save(path + 'pitches', pitches)
numpy.save(path + 'pauses', pauses)
numpy.save(path + 'symbols', symbols)
'''
################################entropy#########################################
pitches = numpy.load(path + 'pitches.npy')
pauses = numpy.load(path + 'pauses.npy')
pause_thred = 85
pitch_thred = 40
feature = numpy.vstack((pitches, pauses))
symbols = symbolisation(feature, pause_thred=pause_thred, pitch_thred=pitch_thred)
print(set(symbols))
Keyboard
#######some manipulation on symbols to make enough amount of symbols for entropy calculation
N = 30


entropy_durations = [300, 500, 700, 900, 1000]
overlap = 0.
ent_prof = entropys(symbols, entropy_durations, overlap, pause_thred)