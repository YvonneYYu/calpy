import numpy

def integer_to_edge_condition( k ):
    """Given a pattern id k, return left, middle, and right conditions

    Args:
        k (int): pattern id

    Returns:
        (Lcond, Mcond, Rcond) a tuple of three functions.
    """
    assert k < 63, "Input k = {} must be < 63.".format(k)

    bdigs = [ int(bit) for bit in '{:06b}'.format(k) ]
    
    Lcond = lambda x, y : x==bdigs[0] and y==bdigs[1]
    Rcond = lambda x, y : x==bdigs[4] and y==bdigs[5]
    
    #xs and ys here MUST be numpy arrays or the logical check will fail
    #Mcond = lambda xs, ys : numpy.all(xs==bdigs[2]) and numpy.all(ys==bdigs[3])

    Mcond = lambda xs, ys : numpy.logical_and( numpy.logical_xor(not bdigs[2], xs),  numpy.logical_xor(not bdigs[3], ys) )

    return (Lcond, Mcond, Rcond)

def consecutive(data, stepsize=1):
    """Find left and right indices of consecutive elements.

    Args:
        data (numpy.array(int)): 1D numpy.array

    Returns:
        [(L, R)] a list of tuples that contains the left and right indeces in data that breaks continuity.
    """
    if len(data)==0:
        return data

    runs = numpy.split(data, numpy.where(numpy.diff(data) != stepsize)[0]+1)
    return [ (run[0],run[-1]) for run in runs ]


def ranges_satisfying_condition( A, B, Lcond, Mcond, Rcond ):
    """Compute the left and right boundries of a particular pause pattern.

    Args:
        A (numpy.array(int)): 0-1 1D numpy integer array with 1s marking pause of speaker A.
        B (numpy.array(int)): 0-1 1D numpy integer array with 1s marking pause of speaker B.
        Lcond (function): bool, bool -> bool
        Mcond (function): numpy.array(bool), numpy.array(bool) -> numpy.array(bool)
        Rcond (function): bool, bool -> bool
    
    Returns:
        [(L, R)] a list of tuples that contains the left and right indeces in pause profiles that satifies Lcond and Mcond and Rcond
    """
    N = A.shape[0]
    middles = consecutive(numpy.where( Mcond(A, B) == True )[0])
    return [ (L,R) for (L,R) in middles if L>0 and R+1<N and Lcond(A[L-1],B[L-1]) and Rcond(A[R+1],B[R+1]) ]

#up = uptake
#so = successful overtake
#fo = failed overtake
#ip = inner pause
name_to_edge_condition = dict({
    "AupB":integer_to_edge_condition(18),
    "BupA":integer_to_edge_condition(33),
    "AsoB":integer_to_edge_condition(30),
    "AfoB":integer_to_edge_condition(29),
    "BsoA":integer_to_edge_condition(45),
    "BfoA":integer_to_edge_condition(46),
    "AipB":integer_to_edge_condition(34),
    "BipA":integer_to_edge_condition(17)
})

short_to_long_name = dict({
    "AupB":"A uptakes B",
    "BupA":"B uptakes A",
    "AsoB":"A successful overtake B",
    "AfoB":"A failed overtake B",
    "BsoA":"B successful overtake A",
    "BfoA":"B failed overtake A",
    "AipB":"A inner pause",
    "BipA":"B inner pause",
})

################################testing backyard################################
#def test_edge_conditions():
#    #A UP B
#    L,M,R = name_to_edge_condition["AupB"]
#    A = numpy.array([0,0,0,0,1])
#    B = numpy.array([1,0,0,0,0])
#    print( L(A[0],B[0]), M(A[1:-1],B[1:-1]), R(A[-1],B[-1]) )
#
#    #B UP A
#    L,M,R = name_to_edge_condition["BupA"]
#    A = numpy.array([1,0,0,0,0])
#    B = numpy.array([0,0,0,0,1])
#    print( L(A[0],B[0]), M(A[1:-1],B[1:-1]), R(A[-1],B[-1]) )
#
#    #A SO B
#    L,M,R = name_to_edge_condition["AsoB"]
#    A = numpy.array([0,1,1,1,1])
#    B = numpy.array([1,1,1,1,0])
#    print( L(A[0],B[0]), M(A[1:-1],B[1:-1]), R(A[-1],B[-1]) )
#
#    #A FO B
#    L,M,R = name_to_edge_condition["AfoB"]
#    A = numpy.array([0,1,1,1,0])
#    B = numpy.array([1,1,1,1,1])
#    print( L(A[0],B[0]), M(A[1:-1],B[1:-1]), R(A[-1],B[-1]) )
#
#    #B SO A
#    L,M,R = name_to_edge_condition["BsoA"]
#    A = numpy.array([1,1,1,1,0])
#    B = numpy.array([0,1,1,1,1])
#    print( L(A[0],B[0]), M(A[1:-1],B[1:-1]), R(A[-1],B[-1]) )
#
#    #B FO A
#    L,M,R = name_to_edge_condition["BfoA"]
#    A = numpy.array([1,1,1,1,1])
#    B = numpy.array([0,1,1,1,0])
#    print( L(A[0],B[0]), M(A[1:-1],B[1:-1]), R(A[-1],B[-1]) )
#
#    #A IP B
#    L,M,R = name_to_edge_condition["AipB"]
#    A = numpy.array([1,0,0,0,1])
#    B = numpy.array([0,0,0,0,0])
#    print( L(A[0],B[0]), M(A[1:-1],B[1:-1]), R(A[-1],B[-1]) )
#
#    #B IP A
#    L,M,R = name_to_edge_condition["BipA"]
#    A = numpy.array([0,0,0,0,0])
#    B = numpy.array([1,0,0,0,1])
#    print( L(A[0],B[0]), M(A[1:-1],B[1:-1]), R(A[-1],B[-1]) )
