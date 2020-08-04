import calculate_discrete_distributions as ddist
import calculate_continue_distributions as cdist
import simulate_discrete_variables as sdvar
import simulate_continue_variables as scvar
import arrays as arr
import tests

import numpy as np

### Guia 7
def g7_ex1():
  # We already know the frequencys,
  # we dont have to create an array
  # s of length 564, so instead we
  # create directly the Ni's (frequencys)
  n = 564 # Total amount of samples
  x = [ 'white', 'pink', 'red' ]
  p = [ 1/4, 2/4, 1/4 ]
  N = { 'white': 141, 'pink': 291, 'red': 132 }
  stat, pval = tests.pearson_chi_squared_test(x, p, n, N=N)
  print(f'G7 EX1 - Stat: {stat}, Pval: {pval}')

def g7_ex2():
  n = 1000
  x = [ 1, 2, 3, 4, 5, 6 ]
  p = [ 1/6, 1/6, 1/6, 1/6, 1/6, 1/6 ]
  N = { 1: 158, 2: 172, 3: 164, 4: 181, 5: 160, 6: 165 }
  stat, pval = tests.pearson_chi_squared_test(x, p, n, N=N)
  print(f'G7 EX2 - Stat: {stat}, Pval: {pval}')

def g7_ex3_alt1():
  n = 10
  intervals = 100000 # DISCRETIZATION, More intervals = More precision
  x = np.cumsum(np.full(intervals, 1))
  p = np.full(intervals, 1/intervals)
  s = [ 12, 18, 6, 33, 72, 83, 36, 27, 77, 74 ]
  stat, pval = tests.pearson_chi_squared_test(x, p, n, s=s)
  print(f'G7 EX3 alternative 1 - Stat: {stat}, Pval: {pval}')

def g7_ex3_alt2():
  n = 10
  N = { 0.1: 1, 0.2: 2, 0.3: 1, 0.4: 2, 0.8: 3, 0.9: 1 }
  x = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ] # DISCRETIZATION
  p = [ 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9 ] # Probability of each interval
  stat, pval = tests.pearson_chi_squared_test(x, p, n, N=N)
  print(f'G7 EX3 alternative 2 - Stat: {stat}, Pval: {pval}')

def g7_ex3_alt3():
  s = [ 0.12, 0.18, 0.06, 0.33, 0.72, 0.83, 0.36, 0.27, 0.77, 0.74 ]
  stat, pval = tests.kolmogorov_smirnov(s, 1000, cdist.uniform_CDF, 0, 1)
  print(f'G7 EX3 alternative 3 - Stat: {stat}, Pval: {pval}')

def g7_ex4():
  s = [ 86, 133, 75, 22, 11, 144, 78, 122, 8, 146, 33, 41, 99 ]
  stat, pval = tests.kolmogorov_smirnov(s, 100, cdist.exponential_CDF, 1/50)
  print(f'G7 EX4 - Stat: {stat}, Pval: {pval}')

def g7_ex5():
  n = 18
  s = [ 6, 7, 3, 4, 7, 3, 7, 2, 6, 3, 7, 8, 2, 1, 3, 5, 8, 7 ]
  x = [ 1, 2, 3, 4, 5, 6, 7, 8 ]
  estimated_p = np.sum(s)/(len(s)*8)
  p = [ ddist.binomial_PDF(8, estimated_p, i) for i in range(8) ]
  stat, pval = tests.pearson_chi_squared_test(x, p, n, s=s, unknown_params=1)
  print(f'G7 EX5 - Stat: {stat}, Pval: {pval}')

# Example 8.1, Cap 8, Page 3
def example_8_1():
  s = [
    0, 0,
    1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    7, 7
  ]
  x = [ 0, 1, 2, 3, 4, 5, 6, 7 ]
  p = [ 0.0078125, 0.0546875, 0.1640625, 0.2734375, 0.2734375, 0.1640625, 0.0546875, 0.0078125 ]
  stat, pval = tests.pearson_chi_squared_test(x, p, len(s), s)
  print(f'Example 8.1, Cap. 8, Page 3 - Stat: {stat}, Pval: {pval}')

# Example 8.2, Cap 8, Page 6
def example_8_2():
  # H0 is: Data is from Poisson distribution
  n = 30
  # Where 5 is actually a group with all values greater than 5
  x = [0, 1, 2, 3, 4, 5] # DISCRETIZATION
  tetha = arr.mu(s) # Estimates Lambda, calculated by the experimental data
  p = [
    cdist.poisson_PDF(tetha, 0), # P(X = 0) ningun accidente
    cdist.poisson_PDF(tetha, 1), # P(X = 1) 1 accidente
    cdist.poisson_PDF(tetha, 2), # P(X = 2) 2 accidentes
    cdist.poisson_PDF(tetha, 3), # P(X = 3) 3 accidentes
    cdist.poisson_PDF(tetha, 4), # P(X = 4) 4 accidentes
    1 - cdist.poisson_CDF(tetha, 5) # P(X > 4) accidentes
  ]
  N = { 0: 6, 1: 2, 2: 1, 3: 9, 4: 7, 5: 5 }
  stat, pval = tests.pearson_chi_squared_test(x, p, n, N=N, unknown_params=1)
  print(f'Example 8.2, Cap. 8, Page 6 - Stat: {stat}, Pval: {pval}')

# Example 8.3, Cap 8, Page 11
def example_8_3():
  s = [55, 72, 81, 94, 112, 116, 124, 140, 145, 155]
  kolmogorov_smirnov(s, 10000, cdist.exponential_CDF, 1/100)
  print(f'Example 8.3, Cap. 8, Page 11 - Stat: {stat}, Pval: {pval}')

g7_ex1()
g7_ex2()
g7_ex3_alt1()
g7_ex3_alt2()
g7_ex3_alt3()
g7_ex4()
g7_ex5()
example_8_1()
example_8_2()
example_8_3()













