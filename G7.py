import calculate_discrete_distributions as ddist
import calculate_continue_distributions as cdist
import simulate_discrete_variables as sdvar
import simulate_continue_variables as scvar
import arrays as arr
import tests

import numpy as np

### Guia 7

def g7_ex1(): # Pearson Chi2 Test
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

def g7_ex2(): # Pearson Chi2 Test
  n = 1000
  x = [ 1, 2, 3, 4, 5, 6 ]
  p = [ 1/6, 1/6, 1/6, 1/6, 1/6, 1/6 ]
  N = { 1: 158, 2: 172, 3: 164, 4: 181, 5: 160, 6: 165 }
  stat, pval = tests.pearson_chi_squared_test(x, p, n, N=N)
  print(f'G7 EX2 - Stat: {stat}, Pval: {pval}')

def g7_ex3_alt1(): # Pearson Chi2 Test
  n = 10
  intervals = 100000 # DISCRETIZATION, More intervals = More precision
  x = np.cumsum(np.full(intervals, 1))
  p = np.full(intervals, 1/intervals)
  s = [ 12, 18, 6, 33, 72, 83, 36, 27, 77, 74 ]
  stat, pval = tests.pearson_chi_squared_test(x, p, n, s=s)
  print(f'G7 EX3 alternative 1 - Stat: {stat}, Pval: {pval}')

def g7_ex3_alt2(): # Pearson Chi2 Test
  n = 10
  N = { 0.1: 1, 0.2: 2, 0.3: 1, 0.4: 2, 0.8: 3, 0.9: 1 }
  x = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ] # DISCRETIZATION
  p = [ 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9 ] # Probability of each interval
  stat, pval = tests.pearson_chi_squared_test(x, p, n, N=N)
  print(f'G7 EX3 alternative 2 - Stat: {stat}, Pval: {pval}')

def g7_ex3_alt3(): # Kolmogorov-Smirnov Test
  s = [ 0.12, 0.18, 0.06, 0.33, 0.72, 0.83, 0.36, 0.27, 0.77, 0.74 ]
  stat, pval = tests.kolmogorov_smirnov(s, 1000, cdist.uniform_CDF, 0, 1)
  print(f'G7 EX3 alternative 3 - Stat: {stat}, Pval: {pval}')

def g7_ex4(): # Kolmogorov-Smirnov Test
  s = [ 86, 133, 75, 22, 11, 144, 78, 122, 8, 146, 33, 41, 99 ]
  stat, pval = tests.kolmogorov_smirnov(s, 100, cdist.exponential_CDF, 1/50)
  print(f'G7 EX4 - Stat: {stat}, Pval: {pval}')

def g7_ex5(): # Pearson Chi2 Test
  n = 18
  s = [ 6, 7, 3, 4, 7, 3, 7, 2, 6, 3, 7, 8, 2, 1, 3, 5, 8, 7 ]
  x = [ 1, 2, 3, 4, 5, 6, 7, 8 ]
  estimated_p = arr.mu(s) / 8
  p = [ ddist.binomial_PDF(8, estimated_p, i) for i in range(8) ]
  stat, pval = tests.pearson_chi_squared_test(x, p, n, s=s, unknown_params=1)
  print(f'G7 EX5 - Stat: {stat}, Pval: {pval}')

def g7_ex6(): # Kolmogorov-Smirnov Test
  n = 10
  s = [ scvar.exponential(1) for _ in range(10) ]
  stat, pval = tests.kolmogorov_smirnov(s, 10000, cdist.exponential_CDF, 1)
  print(f'G7 EX6 - Stat: {stat}, Pval: {pval}')

def g7_ex7(): # Kolmogorov-Smirnov Test
  n = 10
  s = [ scvar.t_student(11) for i in range(n) ]
  stat, pval = tests.kolmogorov_smirnov(s, 100000, cdist.normal_CDF, 0, 1)
  print(f'G7 EX7 - Stat: {stat}, Pval: {pval}')

def g7_ex8(): # Kolmogorov-Smirnov Test Unknown Parameters
  sims, n = 100000, 15
  s = [ 1.6, 10.3, 3.5, 13.5, 18.4, 7.7, 24.3, 10.7, 8.4, 4.9, 7.9, 12, 16.2, 6.8, 14.7 ]
  estimated_lamda = arr.mu(s)
  stat, pval = tests.kolmogorov_smirnov(s, sims, cdist.exponential_CDF, 1 / estimated_lamda)
  print(f'G7 EX8 - Stat: {stat}, Pval: {pval}')
  print(f'Suppose we dont accept H0.')

  new_pval = 0
  for i in range(sims):
    new_sample = [ scvar.exponential(1 / estimated_lamda) for _ in range(n) ]
    new_lamda = arr.mu(new_sample)
    new_Fe = tests.empirical_dist(new_sample)
    new_stat = tests.ks_statistic(new_sample, new_Fe, cdist.exponential_CDF, 1 / new_lamda)
    if new_stat > stat:
      new_pval += 1

  print(f'New pvalue calculated with {sims} new samples - Pval: {new_pval/sims}')

def g7_ex9(): # Kolmogorov-Smirnov Test Unknown Parameters
  sims, n = 3, 12
  s = [ 91.9, 97.8, 111.4, 122.3, 105.4, 95.0, 103.8, 99.6, 96.6, 119.3, 104.8, 101.7 ]
  est_sigma, est_mu = arr.sigma_and_mu(s)
  stat, pval = tests.kolmogorov_smirnov(s, sims, cdist.normal_CDF, est_mu, est_sigma**(1/2))
  print(f'G7 EX9 - Stat: {stat}, Pval: {pval}')
  print(f'Suppose we dont accept H0.')

  new_pval = 0
  for i in range(sims):
    new_sample = [ scvar.normal_reject_method(est_mu, est_sigma) for _ in range(n) ]
    new_sigma, new_mu = arr.sigma_and_mu(new_sample)
    new_Fe = tests.empirical_dist(new_sample)
    new_stat = tests.ks_statistic(new_sample, new_Fe, cdist.normal_CDF, new_mu, new_sigma**(1/2))
    if new_stat > stat:
      new_pval += 1

  print(f'New pvalue calculated with {sims} new samples - Pval: {new_pval/sims}')

def g7_ex10(): # P-Value Range Tests (Recursive, Aprox by Normal and Simulated)
  print('G7 EX10')
  s1 = [ 65.2, 67.1, 69.4, 78.4, 74.0, 80.3 ]
  s2 = [ 59.4, 72.1, 68.0, 66.2, 58.5 ]

  pval = tests.two_samples_recursive(s1, s2)
  print(f'Recursive algorithm - Pval: {pval}')
  pval = tests.two_samples_normal(s1, s2)
  print(f'Aproximation with normal - Pval: {pval}')
  pval = tests.two_samples_simulated(s1, s2, 10000)
  print(f'Aproximation with 10000 simulations - Pval: {pval}')

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
  x = [ 0, 1, 2, 3, 4, 5 ] # DISCRETIZATION
  s = [ 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 8 ]
  tetha = arr.mu(s) # Estimates Lambda, calculated by the experimental data
  p = [
    ddist.poisson_PDF(tetha, 0), # P(X = 0) ningun accidente
    ddist.poisson_PDF(tetha, 1), # P(X = 1) 1 accidente
    ddist.poisson_PDF(tetha, 2), # P(X = 2) 2 accidentes
    ddist.poisson_PDF(tetha, 3), # P(X = 3) 3 accidentes
    ddist.poisson_PDF(tetha, 4), # P(X = 4) 4 accidentes
    1 - ddist.poisson_CDF(tetha, 5) # P(X > 4) accidentes
  ]
  N = { 0: 6, 1: 2, 2: 1, 3: 9, 4: 7, 5: 5 }
  stat, pval = tests.pearson_chi_squared_test(x, p, n, N=N, unknown_params=1)
  print(f'Example 8.2, Cap. 8, Page 6 - Stat: {stat}, Pval: {pval}')

# Example 8.3, Cap 8, Page 11
def example_8_3():
  s = [55, 72, 81, 94, 112, 116, 124, 140, 145, 155]
  stat, pval = tests.kolmogorov_smirnov(s, 10000, cdist.exponential_CDF, 1/100)
  print(f'Example 8.3, Cap. 8, Page 11 - Stat: {stat}, Pval: {pval}')

g7_ex1()
g7_ex2()
g7_ex3_alt1()
g7_ex3_alt2()
g7_ex3_alt3()
g7_ex4()
g7_ex5()
g7_ex6()
g7_ex7()
g7_ex8()
g7_ex9()
g7_ex10()
example_8_1()
example_8_2()
example_8_3()













