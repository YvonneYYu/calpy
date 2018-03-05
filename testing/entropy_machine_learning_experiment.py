"""
Provided mfcc and pitch data of Life Story Part2 recording, do machine learning for anormaly detection
"""
#run in root dir of calpy
import numpy
from calpy.entropy import entropy_profile, symbolise_mfcc_multidimension
from testing.machine_learning import *
import os

path_res = './testing/entropy_machine_learning_results/'
path_data = './testing/entropy_machine_learning_data/'
if not os.path.isdir(path_res):
    os.makedirs(path_res)

fname = 'PWD003_life_story2_PWD_mfcc'
def entropys(symbols, entropy_durations, overlap_factors, saveresult=False):
    res = {}
    for dur in entropy_durations:
        for overlap_factor in overlap_factors:
            overlap = int(overlap_factor * dur)
            vfun = lambda x : entropy_profile(x, window_size=dur, window_overlap=overlap)
            ent_prof = numpy.apply_along_axis(vfun, 1, symbols)
            t = numpy.arange(ent_prof.shape[1]) * dur * 0.01 * (1 - overlap_factor)
            t = t.reshape(1, ent_prof.shape[1])
            ent_prof = numpy.append(t, ent_prof, axis=0)
            if saveresult:
                numpy.save(path_res + fname + "_entropy_with_time_info_{}_{:.1f}".format(dur, overlap_factor), ent_prof)
            res[(dur, overlap_factor)] = ent_prof
    return res

data = numpy.loadtxt(path_data + fname + '.csv', delimiter=',')
trouble_times = numpy.loadtxt(path_data + fname + '_trouble_time_stamps.csv', delimiter=',',dtype=int)
trouble_times *= 100
mfcc, t = data[:,1:].T, data[:,0].T.reshape(1, data.shape[0])
y = numpy.zeros(t.shape)
for s, e in trouble_times:
    y[:, s:e+1] = 1
symbols = symbolise_mfcc_multidimension(mfcc)

entropy_durations = [300, 400, 500, 1000]
overlap = [0, 0.5, 0.9]
ent_profiles = entropys(symbols, entropy_durations, overlap)