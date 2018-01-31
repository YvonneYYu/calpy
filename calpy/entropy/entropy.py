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

def symbolise_speech(pitches, pauses, eps=5e-1):
    """Symbolise a small speech segment according to pitch and pause.

    Args:
        pitches (list(float)):  A list of pitches.
        pauses (list(int)):  A list of pauses.
        eps (float, optional): Threshold of normalised pitch variation to be considered as atypical speech. Default to 0.5

    Returns:
        int: one symbol of the small speech segment.
    """
    #if this is a long pause, then return symbol 2
    if pauses.all():
        return 2

    pitches /= numpy.max(pitches)
    pitch_dif = numpy.sum(numpy.diff(pitches)) / (pitches.shape[0] - 1)

    #if average pitches change is greater than eps, then return symbol 1
    if pitch_dif > eps:
        return 1

    return 0