import numpy

def entropy_profile( symbols, window_size = 100, window_overlap = 0 ):
    """Calculate the entropy profile of a list of symbols.

    Args:
        symbols (list(int)):  A list of symbols.
        window_size (int, optional):  Number of symbols per entropy window.  Defaults to 100.
        window_overlap (int, optional):  How much the entropy windows should overlap.  Defaults to 0.
    
    Returns:
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

def symbolise( pitches, eps=8e-2 ):
    """Symbolose a small piece of speech segment according to pitch slopes.

    Args:
        pitches (list(float)):  A list of pitches.
        eps (float, optional):  Treshold of pitch slopes to be considered level.  Defaults to tan(5 degrees).

    Returns:
        int: one symbol of the small piece of speech segment.
    """
    #if input pitches are all 0.0, categorise it as silent (3)
    if not pitches.any():
        return 3

    N  = len(pitches)
    xs = numpy.arange(N)
    ys = pitches

    AA = numpy.vstack([xs, numpy.ones(N)]).T
    m, c = numpy.linalg.lstsq(AA, ys)[0]

    #return m

    if m>eps:
        return 0 #rising
    elif -eps <= m <= eps:
        return 2 # level
    else:
        return 1 # falling

def symbolise_speech(pitches, pauses, thre=250):
    """Symbolise a small speech segment according to pitch and pause.

    Args:
        pitches (list(float)):  A list of pitches.
        pauses (list(int)):  A list of pauses.
        thre (float, optional): Threshold of high pitch.

    Returns:
        int: one symbol of the small speech segment.
    """
    #if all silent
    if pauses.all():
        return 1
    #high pitch
    if numpy.average(pitches) > thre:
        return 2
    #normal pitch
    return 0

def symbolise_mfcc(mfcc):
    symbols = numpy.zeros(mfcc.shape[1])
    mu = numpy.average(mfcc, axis=1)
    low_mu, med_mu, high_mu = numpy.average(mu[:4]), numpy.average(mu[4:8]), numpy.average(mu[8:12])
    for i, mf in enumerate(mfcc.T):
        symbols[i] = symbols[i] * 2 +  int(numpy.average(mf[8:]) > high_mu)
        symbols[i] = symbols[i] * 2 + int(numpy.average(mf[4:8]) > med_mu)
        symbols[i] = symbols[i] * 2 +int(numpy.average(mf[:4]) > low_mu)
    return symbols