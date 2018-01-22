import matplotlib.pyplot as plt
import calpy
import numpy
from calpy.testing.experiments import *
from calpy.entropy.entropy import entropy_profile

def artificial_production( frequencys, symbol_duration, entropy_duration ):
    """
    :symbol_duration: in seconds, duration of time per symbol, float
    :entropy_duration: in seconds, duration of time per entropy, ASSERT(symbol_duration *much smaller than* entropy_duration) float
    """
    sampling_frequency = 8000
    # because the duration of every uni-frequency signal is 0.025 second, there
    # is no need for overlap in pitch detection.

    time_step = 0.025 #second

    #generate an artificial signal
    signal = artificial_signal( frequencys, sampling_frequency )

    #obtain pitches (this is slow)
    pitches = calpy.dsp.pitch_profile(signal, sampling_frequency, time_step=time_step)

    #symbolise the signal
    symbols = []
    for pitch in numpy.array_split(pitches, len(pitches) // int(symbol_duration / time_step)):
        symbols.append( symbolise(pitch) )

    #obtain its entropy profile
    entropys = entropy_profile(symbols, int(entropy_duration / symbol_duration), window_overlap=0 )

    return entropys

#(fs, )
# one frequency --> 25ms
# len(freqs) = 6s / 0.025s / 3

sym_durs = 0.03
ent_durs = [3,6,9,]

pause  = [0.0]*3  # 0.1s
#medium_pause = [0.0]*10 # 0.25s
#long_pause   = [0.0]*40 # 1s
#sound = list(numpy.random.randint(255-75, size=200)+75) # 5s

frequencys = sound + short_pause + sound + long_pause + sound + medium_pause + sound # 213.5 seconds
frequencys *= 10 #2135 seconds = 35.5 min

numpy.save("./Experiment1/frequency", numpy.array(frequencys) )

for sym_dur in sym_durs:
    for ent_dur in ent_durs:
        if int(ent_dur / sym_dur) >= 10:
            EP = artificial_production( frequencys, sym_dur, ent_dur)
#            numpy.save( "./Experiment1/EP_{}_{}".format(int(sym_dur*100), int(ent_dur*100)), EP)

