"""
Provided mfcc and pitch data of Life Story Part2 recording, do machine learning for anormaly detection
"""
#run in root dir of calpy
import numpy
from calpy.entropy import entropy_profile_2D, symbolise_mfcc_multidimension, estimate_Gaussian, multivariate_Gaussion
from calpy.plots import feature_distribution
from scipy.stats.mstats import zscore
import matplotlib.pyplot as plt
import os

################################data preparation################################
path_res = './testing/entropy_machine_learning_results/'
path_data = './testing/entropy_machine_learning_data/'
if not os.path.isdir(path_res):
    os.makedirs(path_res)

fname = 'PWD003_life_story2_PWD_mfcc'

def trouble_filter(troubles, thre=50):
    s = 0
    cnt = 0
    troubles_filtered = troubles
    for idx, trouble in enumerate(troubles):
        if trouble:
            if not cnt:
                s = idx
            cnt += 1
        elif cnt:
            if cnt < thre:
                troubles_filtered[s:idx] = False
            cnt = 0
    return troubles_filtered

data = numpy.loadtxt(path_data + fname + '.csv', delimiter=',')
trouble_times = numpy.loadtxt(path_data + fname + '_trouble_time_stamps.csv', delimiter=',',dtype=int)
trouble_times *= 100
X, t = data[:,1:].T, data[:,0]
Y = numpy.zeros(t.shape)
for s, e in trouble_times:
    Y[:, s:e+1] = 1
#numpy.savez(path_res + fname + '_training_data', X=X, Y=Y)
symbols = symbolise_mfcc_multidimension(mfcc)

entropy_durations = [300, 400, 500, 1000]
overlaps = [0, 0.5, 0.9]
ent_profiles = {}
for win in entropy_durations:
    for overlap in overlaps:
        over_win = int(overlap * win)
        ent_profiles[(win, overlap)] = entropy_profile_2D(X, win, over_win)

#numpy.save(path_res + fname + '_entropys', ent_profiles)


###############################anormaly detection###############################
# call calpy.plots.feature_distribution to observe to mfcc features obey Gaussian distribution
feature_distribution(X, None, savefig=False, showfig=True)

#Use Gaussian distribution to detect anomaly

Mu, Sigma2 = estimate_Gaussian(X)
p = numpy.empty(X.shape[1])
for idx, x in enumerate(X.T):
    p[idx] = multivariate_Gaussion(x.reshape(x.shape[0], 1), Mu, Sigma2)
p /= p.max()
troubles = p < 0.01

#fileter out troubles shorter than 0.5 second
troubles_filtered = trouble_filter(troubles)

#Use PCA to reduce feature dimension

#feature normalisation
X1 = zscore(X, axis=1)

#dimension reduction
m = X1.shape[1]
X1_CV = numpy.dot(X1, X1.T) / m # co-variance matrix
U, S, V = numpy.linalg.svd(X1_CV)
U_reduce = U[:, :3]
X1_reduce = numpy.dot(U_reduce.T, X1)

Mu_reduce, Sigma2_reduce = estimate_Gaussian(X1_reduce)
p_reduce = numpy.empty(X1_reduce.shape[1])
for idx, x_reduce in enumerate(X1_reduce.T):
    p_reduce[idx] = multivariate_Gaussion(x_reduce.reshape(x_reduce.shape[0], 1), Mu_reduce, Sigma2_reduce)

p_reduce /= p_reduce.max()
troubles_reduce = p_reduce < 0.01

#fileter out troubles shorter than 1 second
troubles_reduce_filtered = trouble_filter(troubles_reduce)


gaussian_results = {}
gaussian_results['p'] = p
gaussian_results['troubles'] = troubles
gaussian_results['troubles_filtered'] = troubles_filtered
gaussian_results['p_reduce'] = p_reduce
gaussian_results['troubles_reduce'] = troubles_reduce
gaussian_results['troubles_reduce_filtered'] = troubles_reduce_filtered

numpy.save(path_res + fname + '_gaussian_results', gaussian_results)
#for clustering
X2 = ent_profiles[(300, 0.9)][1:,:]
X2 = zscore(X2, axis=1)

