import numpy

def entropy_profile( symbols, window_size = 100, window_overlap = 0 ):
    """Calculate the entropy profile of a list of symbols.

    Args:
        symbols (list(int)):  A list of symbols.
        window_size (int, optional):  Number of symbols per entropy window.  Defaults to 100.
        window_overlap (int, optional):  How much the entropy windows should overlap.  Defaults to 0.
    
    Returns
        numpy.array(float): The entropy profile.
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