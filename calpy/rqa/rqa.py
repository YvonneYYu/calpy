import numpy
import math
#from .. import utilities

class phase_space(object):
    """Phase space class.
    """
    def __init__(self, xs, tau=1, m=2, eps=.001):
        self.tau, self.m, self.eps = tau, m, eps
        
        N = int(len(xs)-m*tau+tau)

        self.matrix = numpy.empty([N,m],dtype=float)
        for i in range(N):
            self.matrix[i,:] = xs[i:i+1+int(m*tau-tau):tau]
        
        self.recurrence_matrix = None

        return None

    def __repr__(self):
        return "phase_space()"

    def __str__(self):
        return "{} with shape {} and (tau, m, eps) = ({}, {}, {})".format(type(self.matrix), self.matrix.shape, self.tau, self.m, self.eps)

    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        return id(self) == id(other)

def _Theta(x, y, eps):
    """Theta tmp
        
        Args:
            x:
            y:
            eps:

        Returns:
            int: 0 or 1.
    """
    sm  = 0
    for k in range(len(x)):
        sm += (x[k]-y[k])**2
        if sm > eps:
            return 0
    return 1

_recurrence_matrix_cache = dict()
def recurrence_matrix(xps, yps=None, joint=False):
    """Computes cross-reccurence matrix when two inputs are given and self-reccurence otherwise.

        Args:
            xps (numpy.array): Phase_space object(s).
            yps (numpy.array, optional): Phase_space object for cross reccurence.  Defaults to none.
            joint (bool, optional):  Should joint reccurence be calculated?  Defaults to False.
        
        Returns:
            numpy.array : A 2D numpy matrix.
    """

    if not yps:
        yps, cross = xps, False
    else:
        cross = True

    if (xps,yps,joint) in _recurrence_matrix_cache:
        return _recurrence_matrix_cache[xps, yps, joint]

    if (xps.matrix.shape, xps.tau, xps.m, xps.eps) != (yps.matrix.shape, yps.tau, yps.m, yps.eps):
        print("Error: Input phase spaces have different parameters.")
        return

    if joint:
        return numpy.multiply( recurrence_matrix(xps), recurrence_matrix(yps) )

    BB, AA, tau, m, eps = yps.matrix, xps.matrix, xps.tau, xps.m, xps.eps

    N = AA.shape[0]

    ans = numpy.full([N, N],0)
    for i in range(N):
        for j in range(N if cross else i+1):
            #ans[i][j] = _Theta( AA[i], BB[j], eps)
            ans[i][j] = numpy.linalg.norm(AA[i]-BB[j])
    
    _recurrence_matrix_cache[xps,yps,joint] = ans
    return _recurrence_matrix_cache[xps, yps, joint]

def cross_recurrence_matrix( xps, yps ):
    """Cross reccurence matrix.

        Args:
            xps (numpy.array):
            yps (numpy.array):

        Returns:
            numpy.array : A 2D numpy array.

    """
    return recurrence_matrix( xps, yps )

def joint_recurrence_matrix( xps, yps ):
    """Joint reccurence matrix.

        Args:
            xps (numpy.array):
            yps (numpy.array):

        Returns:
            numpy.array : A 2D numpy array.
    """
    return recurrence_matrix( xps, yps, joint=True )

def recurrence_rate( AA ):
    """Computes reccurence-rate from reccurence matrix.

        Args:
            AA (numpy.array): A reccurence matrix.
        
        Returns:
            numpy.array : A numpy array.
    """

    isLower = utilities.is_lower_triangular(AA)
    
    N = AA.shape[0]
    ans = numpy.zeros( N, dtype=float ) 

    for k in range(1,N):
        tmp = numpy.sum(AA[:k,:k])
        ans[k] += tmp
        
        for i in range(1, N-k):
            if isLower:
                tmp += numpy.sum(AA[i+k-1,i:i+k]) - numpy.sum(AA[i-1:i-1+k,i-1])
            else:
                tmp += numpy.sum( AA[i+k-1, i:i+k] ) \
                       + numpy.sum( AA[i:i+k-1, i+k-1] ) \
                       - numpy.sum( AA[i-1:i-1+k, i-1] ) \
                       - numpy.sum( AA[i-1, i:i-1+k] )
            ans[k] += tmp
        ans[k] /= 0.5*(N-k)*k**2 if isLower else (N-k)*k**2

    return ans


_measures_cache = dict()
def determinism( AA ):
    """Calculates percentage of recurrence points which form diagonal lines.
        
        Args:
            AA (numpy.array): A reccurence matrix.
        
        Returns:
            float: The determinism.
    """

    if (id(AA),"determinism") in _measures_cache:
        return _measures_cache[id(AA),"determinism"]

    isLower = utilities.is_lower_triangular(AA)

    N = AA.shape[0]
    H = dict()
    for key in range(N):
        H[key] = 0
    
    def lower_DET(x):
        for i in range(1, N):
            isPrev = False
            count = 0
            for j in range(i, N):
                #search for consective lines in AA[idx1,idx1-idx]
                if x[j, j-i]:
                    if isPrev: 
                        count += 1
                    else:
                        count = 1
                        isPrev = True
                elif isPrev:
                    isPrev = False
                    H[count] += 1 if count > 1 else 0
                    count = 0
            H[count] += 1 if count>1 else 0
        return

    lower_DET(AA)

    if not isLower:
        lower_DET(numpy.transpose(AA))

    num, avg, max_L = 0, 0, 0
    for key, val in H.items():
        max_L = key if val else max_L
        num += key*val
        avg += val
    
    dem = numpy.sum(AA)
    ENTR = 0
    if avg:
        for key, val in H.items():
            p = val/avg
            ENTR -= p*math.log(p) if p else 0
        PRED = num/avg
    else:
        ENTR = None
        PRED = 0

    DIV = 1/max_L if max_L else float('inf')

    _measures_cache[id(AA),"determinism"] = num/dem
    _measures_cache[id(AA),"pred"] = PRED
    _measures_cache[id(AA),"divergence"] = DIV
    _measures_cache[id(AA),"entropy"] = ENTR

    return _measures_cache[id(AA),"determinism"]


def divergence( AA ):
    """Divergence

        Args:
            AA (numpy.array): A numpy array.

        Returns:
            numpy.array: The answer.
    """
    if (id(AA),"divergence") not in _measures_cache:
        determinism(AA)
    
    return _measures_cache[id(AA),"divergence"]


def entropy( AA ):
    """Entropy

        Args:
            AA (numpy.array): A numpy array.

        Returns:
            numpy.array: The answer.
    """
    if (id(AA),"entropy") not in _measures_cache:
        determinism(AA)
    
    return _measures_cache[id(AA),"entropy"]


def pred( AA ):
    """Pred

        Args:
            AA (numpy.array): A numpy array.

        Returns:
            numpy.array: The answer.
    """
    if (id(AA),"pred") not in _measures_cache:
        determinism(AA)
    
    return _measures_cache[id(AA),"pred"]


def trend( AA, longterm=False ):
    """Calculate the TREND of a give 1d numpy array R.
    
        Args:
            AA (numpy.array(float)):  A 2D matrix.
            longterm (bool, optional):  Should long-term trend be calculate?  Defaults to False.

        Returns:
            float: The medium and long range trends a float tuple (Med, Long)
    """
    N = AA.shape[0]
    R_med  = R[:N//2] - np.mean(R[:N//2])
    R_long = R[:-1] - np.mean(R[:-1])
    
    coef = np.array([i - N//4 +1 for i in range(N//2)])
    Med  = np.dot(coef, R_med)/np.dot(coef, coef)
    
    coef = np.array([i - N//2 +1 for i in range(N-1)])
    Long = np.dot(coef, R_long)/np.dot(coef, coef)
    
    return Long if longterm else Med

def laminarity( AA ): #+ Trapping
    """Laminarity.  Calculates percentage of recurrence points which form verticle lines.

    This function calculates Trapping as a side effect.
        
        Args:
            AA (numpy.array(float)):  A 2D matrix.
        
        Returns:
            float: The laminarity
    """

    N = AA.shape[0]
    H = dict()
    for key in range(N):
        H[key] = 0
    
    #Lower Lam
    for j in range(N):
        isPrev, count = False, 0
        for i in range(j+1, N):
            #search for consecutive lines in M[i, j]
            if AA[i, j]:
                if isPrev: 
                    count += 1
                else:
                    isPrev, count = True, 1
            elif isPrev:
                H[count] += 1 if count > 1 else 0
                isPrev, count = False, 0
        H[count] += 1 if count > 1 else 0

    #Upper Lam
    if not utilities.is_lower_triangular(AA):
        for j in range(N):
            isPrev, count = False, 0
            for i in range(j):
                #search for consecutive lines in M[idx1, idx]
                if AA[i,j]:
                    if isPrev: 
                        count += 1
                    else:
                        isPrev, count = True, 1
                elif isPrev:
                    H[count] += 1 if count > 1 else 0
                    isPrev, count = False, 0
            H[count] += 1 if count > 1 else 0

    num, avg= 0, 0
    for key, val in H.items():
        avg += val
        num += key*val
    dem = num + numpy.sum(AA)

    LAMI = num/dem
    TRAP = num/avg if avg else 0

    _measures_cache[id(AA),"laminarity"] = LAMI
    _measures_cache[id(AA),"trapping"] = TRAP

    return _measures_cache[id(AA),"laminarity"]

def trapping( AA ):
    """Trapping.  Calculates ...

    This function calculates Laminiarity as a side effect.
        
        Args:
            AA (numpy.array(float)):  A 2D matrix.
        
        Returns:
            float: The trapping
    """
    if (id(AA),"trapping") not in _measures_cache:
        return laminarity(AA)

    return _measures_cache[id(AA),"trapping"]