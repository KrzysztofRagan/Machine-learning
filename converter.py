import numpy as np

file = 'datasets/ecoli1.dat'

dataset = np.genfromtxt(file, delimiter=",", comments='@',
                        converters={(-1): lambda s: 0.0 if (s.strip().decode('ascii')) == 'negative' else 1.0})

print(dataset)
