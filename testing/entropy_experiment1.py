"""
Experiment on simple speech signals
"""
import numpy
from scipy.stats import zscore
import matplotlib.pyplot as plt
from calpy.dsp import pitch_profile, pause_profile
from calpy.entropy import entropy_profile
from calpy.entropy import symbolise_speech as symbolise
from calpy.utilities import read_wavfile
import os

path = "./entropy_experiment1_result/"

path_data = './entropy_experiment1_data/'
if not os.path.isdir(path):
    os.makedirs(path)

def symbolisation(feature, pause_thred, pitch_thred):
    N = numpy.ceil(feature.shape[1] / pause_thred).astype(int)
    symbols =numpy.empty(N, dtype=int)
    for idx, data in enumerate(numpy.array_split(feature, N, axis = 1)):
        symbols[idx] = symbolise(data[0], data[1], pitch_thred)
    return symbols

def entropys(symbols, entropy_durations, overlap_factors, N):
    for dur in entropy_durations:
        for overlap_factor in overlap_factors:
            overlap = int(overlap_factor * dur)
            ent_prof = entropy_profile(symbols, window_size=dur, window_overlap=overlap)
            t = numpy.arange(ent_prof.shape[0]) * int(dur * N * 0.01 * (1 - overlap_factor))
            fig = plt.figure(figsize=(20, 9))
            plt.plot(t, ent_prof, '+-')
            plt.title("model calibration with entropy duration of {}s ({} symbols per entropy) and  factor of {:.1f}".format(int(dur * N * 0.01), dur, overlap_factor))
            plt.ylabel('entropy')
            plt.xlabel('time in seconds')
            fig.savefig(path + "entropy_{}_{:.1f}.png".format(dur, overlap_factor))
            numpy.save(path + "entropy_{}_{:.1f}".format(dur, overlap_factor), ent_prof)


fs, sound = read_wavfile(path_data + 'test_audio.wav')
pitches = pitch_profile(sound, fs, lower_threshold = 30, upper_threshold = 1000)
pauses = pause_profile(sound, fs)

numpy.save(path + 'pitches', pitches)
numpy.save(path + 'pauses', pauses)
numpy.save(path + 'symbols', symbols)
################################entropy#########################################
'''
pitches = numpy.load(path + 'pitches.npy')
pauses = numpy.load(path + 'pauses.npy')
'''
pause_thred = 10
pitch_thred = 260
feature = numpy.vstack((pitches, pauses))
symbols = symbolisation(feature, pause_thred=pause_thred, pitch_thred=pitch_thred)


entropy_durations = [300, 400, 500]
overlap = numpy.arange(0, 1, 0.1)
entropys(symbols, entropy_durations, overlap, pause_thred)