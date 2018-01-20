#This is how you call commands from other subpackages from here
from .rqa import rqa

G = utilities.gravity

f = lambda x : G*x
g = lambda x : x**3