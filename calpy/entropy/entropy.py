import numpy

def entropy_profile( symbols, window_size = 100, window_overlap = 0 ):
    """
    :window_size: number of symbols per entropy : int
    :window_overlap: YYY says this is self-explanatory : int
    :rtype: the entropy profile : 1D-numpy.array(float)
    """
    N = len(symbols)
    ent_prof  = []
    time_step = window_size - window_overlap

    for k in range((N - window_size) // time_step + 1):
        window = symbols[k * time_step : k * time_step + window_size]
        key, cnts = numpy.unique(window, return_counts=True)
        cnts = cnts / numpy.sum(cnts)
        probs = dict(zip(key,cnts))
        ent_prof.append(-sum(probs[s] * numpy.log(probs[s]) for s in window))
    return numpy.array(ent_prof)