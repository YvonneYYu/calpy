"""
Experiment on simple speech signals
"""
import numpy
import matplotlib.pyplot as plt
from experiments import symbolise_speech as symbolise
from calpy.dsp import pitch_profile, pause_profile
from calpy.entropy import entropy_profile
from calpy.utilities import read_wavfile


def getPauseThred(pauses):
    pause_length = []
    cnt = 0
    for pause in pauses:
        if pause:
            cnt += 1
        elif cnt > 0:
            pause_length.append(cnt)
            cnt = 0
    
    return thred

def getPitchThred(pitches):
    pitches_diff = numpy.pitches_diff(pitches)
    
    return thred

def symbolisation(feature, pause_thred, pitch_thred):
    N = numpy.ceil(feature.shape[1] / pause_thred).astype(int)
    symbols =numpy.empty(N)
    for idx, data in numpy.array_split(feature, N):
        symbols[idx] = symbolise(data[0], data[1], eps=pitch_thred)
    return symbols

def entropys(symbols, entropy_durations, overlap_factor, N=N):
    for dur in entropy_durations:
        overlap = int(overlap_factor * dur)
        ent_prof = entropy_profile(symbols, window_size=dur, window_overlap=overlap)

        fig = plt.figure()
        plt.plot( ent_prof, '+-')
        plt.title("model calibration with entropy duration of {}s ({} symbols per entropy) and  factor of {}".format(int(dur * N * 0.01), dur, overlap_factor))
        fig.savefig(path + "entropy_{}_{}.png".format(dur, overlap_factor))
        numpy.save(path + "entropy_{}_{}".format(dur, overlap_factor), ent_prof)
        return ent_prof

path = "./experiment1/"
fs, sound = read_wavfile(path + 'data.wav')
pitches = pitch_profile(sound, fs)
pauses = pause_profile(sound, fs)
feature = zip(pitches, pauses)


thred = getPauseThred(pauses)
symbols = symbolisation(feature, eps=thred)
numpy.save(path + 'pitches', pitches)
numpy.save(path + 'pauses', pauses)
numpy.save(path + 'symbols', symbols)

################################entropy#########################################



entropy_durations = [300, 500, 700, 900, 1000]
overlap = 0.7
ent_prof = entropys(symbols, entropy_durations, overlap)