import numpy as np

file = 'datasets/ecoli1.dat'

dataset1 = np.genfromtxt(file, delimiter = ",", comments = '@', converters = {(-1): lambda s: True if s == 'positive' else False} )

print(dataset1)