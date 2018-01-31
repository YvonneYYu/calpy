import numpy

def artificial_signal( frequencys, sampling_frequency=16000, duration=0.025 ):
    """
    Concatonates a sequence of sinusoids of frequency f in frequencies
    """
    sins = map( lambda f : sinusoid(f, sampling_frequency, duration), frequencys)
    return numpy.concatenate( tuple(sins) )

def sinusoid( frequency, sampling_frequency=16000, duration=0.025 ):
    """
    rtype::numpy.array( int(sampling_frequency*duration),  )
    """
    times = numpy.arange(int(sampling_frequency * duration))
    return numpy.sin(2 * numpy.pi * frequency * times / sampling_frequency)

def random_symbols( distribution, length ):
    if sum(distribution) != 1:
        print("Warning: probabilites must sum to 1")
        return

    return numpy.random.choice( len(distribution), length, p=distribution )

def random_run( distributions, length, min_run=100, max_more=100 ):
    
    ans  = list()
    k, N, M = 0, length, len(distributions)
    
    while True:
        more = numpy.random.randint(0,max_more) if max_more else 0
        ext_length = min_run + more
        ext_length = min( ext_length, N )
        
        ans.extend( random_symbols( distributions[k % M], ext_length ) )
        k += 1 % M

        N -= ext_length
        if N <= 0: return ans

def symbolise( pitches, eps=8e-2 ):
    """
        eps :: float = tan(5 degrees) default
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

def symbolise_speech(pitches, pauses, thre=255):
    #if all silent
    if pauses.all():
        return 1
    #high pitch
    if numpy.average(pitches) > thre:
        return 2
    #normal pitch
    return 0