"""
Provided mfcc and pitch data of Life Story Part2 recording, do machine learning for anormaly detection
"""
#run in root dir of calpy
import numpy
from calpy.entropy import entropy_profile_2D, symbolise_mfcc_multidimension, estimate_Gaussian, multivariate_Gaussion
from calpy.plots import feature_distribution
from scipy.stats.mstats import zscore
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
import os

################################data preparation################################
path_res = './testing/entropy_machine_learning_results/'
path_data = './testing/entropy_machine_learning_data/'
if not os.path.isdir(path_res):
    os.makedirs(path_res)

fname = 'PWD003_life_story2_PWD_mfcc'

'''
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
    if cnt and cnt < thre:
        troubles_filtered[s:idx] = False
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
'''

###############################anormaly detection###############################
# call calpy.plots.feature_distribution to observe to mfcc features obey Gaussian distribution
data = numpy.load(path_res + fname + '_training_data.npz')
X = data['X']
Y = data['Y']
feature_distribution(X, None, savefig=False, showfig=True)

#Estimate probabilty with Gaussian model

Mu, Sigma2 = estimate_Gaussian(X)
vfun = lambda x: multivariate_Gaussion(x.reshape(Mu.shape), Mu, Sigma2)
p = numpy.apply_along_axis(vfun, 0, X).squeeze()
p /=p.max()
# plot out p
fig = plt.figure()
plt.hist(p, bins=100)
fig.show()

#fileter out troubles shorter than 0.5 second
troubles_filtered = trouble_filter(troubles)

#Use PCA to reduce feature dimension

#feature normalisation
X1 = zscore(X, axis=1)

#dimension reduction
m = X1.shape[1]
X1_CV = numpy.dot(X1, X1.T) / m # co-variance matrix
U, S, V = numpy.linalg.svd(X1_CV)

# plot out S
fig = plt.figure()
plt.plot(S)
fig.show()
# S shows that all the dimensions are heavily corelated. So dimension reduction is not suggested to apply here
#U_reduce = U[:, :5]
#X1_reduce = numpy.dot(U_reduce.T, X1)
#
#Mu_reduce, Sigma2_reduce = estimate_Gaussian(X1_reduce)
#vfun_reduce = lambda x: multivariate_Gaussion(x.reshape(Mu_reduce.shape), #Mu_reduce, Sigma2_reduce)
#p_reduce = numpy.apply_along_axis(vfun_reduce, 0, X1_reduce).squeeze()
#p_reduce /= p_reduce.max()
#
## plot out p_reduce
#fig = plt.figure()
#plt.plot(numpy.sort(p_reduce)[::-1])
#fig.show()
#
##fileter out troubles shorter than 1 second
#troubles_reduce_filtered = trouble_filter(troubles_reduce)


#gaussian_results = {}
#gaussian_results['p'] = p
#gaussian_results['troubles'] = troubles
#gaussian_results['troubles_filtered'] = troubles_filtered
#gaussian_results['p_reduce'] = p_reduce
#gaussian_results['troubles_reduce'] = troubles_reduce
#gaussian_results['troubles_reduce_filtered'] = troubles_reduce_filtered

#numpy.save(path_res + fname + '_gaussian_results', gaussian_results)
#for clustering
data = numpy.load(path_res + fname + '_entropys.npy').item()
entropy = data[(300,0.9)].T

n_classes = 5

estimator = GaussianMixture(n_components=n_classes, covariance_type='full', max_iter=30, random_state=0)
estimator.fit(entropy)
labels = estimator.predict(entropy)


t = numpy.arange(labels.shape[0])
data = mfcc[:,:500]
fig = plt.figure()
plt.imshow(data)

fig.show()