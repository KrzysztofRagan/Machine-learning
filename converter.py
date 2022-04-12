import numpy as np

file = '/Users/krzysztofragan/Documents/studia magisterskie/Machine learning/imb_IRlowerThan9/ecoli1/ecoli1.csv'

# with open(file) as f:
#     with open("/Users/krzysztofragan/Documents/studia magisterskie/Machine learning/imb_IRlowerThan9/ecoli1//ecoli1.csv", "w") as f1:
#         for line in f:
#             f1.write(line)



dataset1 = np.genfromtxt(file, delimiter = ",", comments = '@', converters = {(-1): lambda s: True if s == 'positive' else False} )

print(dataset1)