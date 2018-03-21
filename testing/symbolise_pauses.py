import numpy
import matplotlib.pyplot as plt

def _integer_to_edge_condition( k ):
    """Given a pattern id k, return left, middle, and right conditions

    Args:
        k (int): pattern id

    Returns:
        [Lcond, Mcond, Rcond] a list of three functions.
    """
    assert k < 63, "Input k = {} must be < 63.".format(k)

    bdigs = [ int(bit) for bit in '{:06b}'.format(k) ]
    Lcond = lambda x, y : numpy.logical_xor(bdigs[0], x) and numpy.logical_xor(bdigs[1], y)
    Mcond = lambda xs, ys : numpy.logical_and( numpy.logical_xor(bdigs[2],xs),  numpy.logical_xor(bdigs[3],ys) )
    Rcond = lambda x, y : numpy.logical_xor(bdigs[4], x) and numpy.logical_xor(bdigs[5], y)

    return (Lcond, Mcond, Rcond)

def _consecutive(data):
    """Find left and right indices of consecutive elements.

    Args:
        data (numpy.array(int)): 1D numpy.array

    Returns:
        [(L, R)] a list of tuples that contains the left and right indeces in data that breaks continuity.
    """
    if data.shape[0] == 0:
        return []
    runs = numpy.split(data, numpy.where(numpy.diff(data) > 1)[0]+1)
    return [ (run[0],run[-1]) for run in runs ]


def _pause_pattern_boundrys( pause_A, pause_B, Lcond, Mcond, Rcond ):
    """Compute the left and right boundries of a particular pause pattern.

    Args:
        pause_A (numpy.array(int)): 0-1 1D numpy integer array with 1s marking pause of speaker A.
        pause_B (numpy.array(int)): 0-1 1D numpy integer array with 1s marking pause of speaker B.
        Lcond (function): bool, bool -> bool
        Mcond (function): numpy.array(bool), numpy.array(bool) -> numpy.array(bool)
        Rcond (function): bool, bool -> bool
    
    Returns:
        [(L, R)] a list of tuples that contains the left and right indeces in pause profiles that satifies Lcond and Mcond and Rcond
    """
    N = A.shape[0]

    middles = _consecutive(numpy.where( Mcond(pause_A, pause_B) )[0])

    return [ (L,R) for (L,R) in middles if L>0 and R+1<N and Lcond(pause_A[L-1],pause_B[L-1]) and Rcond(pause_A[R+1],pause_B[R+1]) ]

def symbolise_pauses(pause_A, pause_B):
    """symbolise a conversation between two speakers into 63 patterns with their pause profiles.

    Args:
        pause_A (numpy.array(int)): 0-1 1D numpy integer array with 1s marking pause of speaker A.
        pause_B (numpy.array(int)): 0-1 1D numpy integer array with 1s marking pause of speaker B.

    Returns:
        symbols (numpy.array(int)): a 2D numpy.array with shape (64, pause_A.shape[0]). Axis 1 is the temporal dimension and axis 0 marks the pattern
    """

    assert pause_A.shape == pause_B.shape, "input numpy.arrays pause_A and pause_B must have the same shape!"

    N = pause_A.shape[0]

    #DONOT CHANGE THE NUMBER 63 AT ANYTIME!!!!
    symbols = numpy.zeros((63,N))
    for k in range(63):
        for (L,R) in _pause_pattern_boundrys( pause_A, pause_B, *_integer_to_edge_condition(k) ):
            symbols[k][L:R+1] = 1

    return symbols

#PWD
path_data = './symbolise_pauses_data/'
#Interviewer
path_res = './symbolise_pauses_res/'
A = numpy.loadtxt(path_data + 'A_pauses.csv', delimiter=',', usecols=(1))
B = numpy.loadtxt(path_data + 'B_pauses.csv', delimiter=',', usecols=(1))

#A = A[: 1000]
#B = B[: 1000]

Symbols = symbolise_pauses(A, B)
numpy.savetxt(path_res + 'Symbols.csv', Symbols, delimiter=',')

fig = plt.figure(figsize=(16, 9), dpi=150)
plt.imshow(Symbols, cmap="Greys", aspect="auto")
plt.title('Pause patterns of PWD003 life story 2')
plt.savefig(path_res + 'Symbols.png')