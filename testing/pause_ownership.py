import numpy
import matplotlib.pyplot as plt

LAND = lambda xs, ys : numpy.logical_and( xs, ys )
LOR  = lambda xs, ys : numpy.logical_or( xs, ys )
LNOT = lambda xs : numpy.logical_not( xs )

def consecutive(data, stepsize=1):
    """
        Returns ENDPOINTS of consecutive runs
    """
    runs = numpy.split(data, numpy.where(numpy.diff(data) != stepsize)[0]+1)
    return [ (run[0],run[-1]) for run in runs ]


def pause_ownership( A, B, Lcond, Mcond, Rcond ):
    N = len(A)
    middles = consecutive(numpy.where( Mcond(A, B) == True )[0])
    return [ (L,R) for (L,R) in middles if L>0 and R+1<N and Lcond(A[L-1],B[L-1]) and Rcond(A[R+1],B[R+1]) ]

A = numpy.loadtxt('PWD003_life_story2_PWD_pauses.csv', delimiter=',')
B = numpy.loadtxt('PWD003_life_story 2_Interviewer_pauses.csv', delimiter=',')

N = 10**3
A = A[:, 1]
B = B[:, 1]

edge_condition = dict()  # = (Lcond, Mcond, Rcond)

edge_condition["AA"] = ( 
    lambda x, y : not x and not y, 
    lambda xs, ys : LAND( xs, LNOT(ys) ),
    lambda x, y : not x and not y )

edge_condition["BB"] = ( 
    lambda x, y : not x and not y, 
    lambda xs, ys : LAND( LNOT(xs), ys ),
    lambda x, y : not x and not y )

edge_condition["AB"] = ( 
    lambda x, y : not x and y, 
    lambda xs, ys : LAND( xs, ys ),
    lambda x, y : x and not y )

edge_condition["BA"] = ( 
    lambda x, y : x and not y, 
    lambda xs, ys : LAND( xs, ys ),
    lambda x, y : not x and y )

edge_condition["barAB"] = ( 
    lambda x, y : not x and y, 
    lambda xs, ys : LNOT( LOR(xs, ys) ),
    lambda x, y : x and not y )

edge_condition["barBA"] = ( 
    lambda x, y : x and not y, 
    lambda xs, ys : LNOT( LOR(xs, ys) ),
    lambda x, y : not x and y )

H = dict()

AA = numpy.zeros( (6, len(A)) )
ylabel = []
for k, ec in enumerate(edge_condition):
    H[ec] = pause_ownership( A, B, *edge_condition[ec] )
    ylabel.append('  '+ ec + '  ')
    print(ec)
    for (L,R) in H[ec]:
        AA[k, L:R] = 1
print(AA[0,0])
plt.imshow( AA, aspect='auto', cmap='Greys')
plt.ylabel(''.join(ylabel[::-1]),fontsize=30)
plt.show()