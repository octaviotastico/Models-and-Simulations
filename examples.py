import calculate_discrete_distributions as ddist
import calculate_continue_distributions as cdist
import simulate_discrete_variables as sdvar
import simulate_continue_variables as scvar
import arrays as arr
import tests

### Guia 7
def g7_ex1():
  # We already know the frequencys,
  # we dont have to create an array
  # s of length 564, so instead we
  # create directly the Ni's (frequencys)

  n = 564 # Total amount of samples
  x = [ 'white', 'pink', 'red' ]
  N = { 'white': 141, 'pink': 291, 'red': 132 }
  p = [ 1/4, 2/4, 1/4 ]
  stat, pval = tests.pearson_chi_squared_test(x, p, n, N=N)
  print(f'Stat: {stat}, Pval: {pval}')

def g7_ex2():
  n = 1000
  x = [ 1, 2, 3, 4, 5, 6 ]
  N = { 1: 158, 2: 172, 3: 164, 4: 181, 5: 160, 6: 165 }
  p = [ 1/6, 1/6, 1/6, 1/6, 1/6, 1/6 ]
  stat, pval = tests.pearson_chi_squared_test(x, p, n, N=N)
  print(f'Stat: {stat}, Pval: {pval}')

g7_ex1()
g7_ex2()

def example_chi_1():
  s = [1, 1, 1, 2, 3, 4, 5] # Muestra
  x = [1, 2, 3, 4, 5] # Posibles X
  p = [0.3, 0.1, 0.2, 0.1, 0.3] # Probabilidades de x
  print(tests.chi_squared_statistic(x, p, s))

def example_chi_2():
  s = [
    0, 0,
    1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    7, 7
  ] # No importa el orden de los datos, solo que nos haya salido esto en el experimento.
  x = [0, 1, 2, 3, 4, 5, 6, 7]
  p = [
    0.0078125, # Probabilidad de que salga un 0
    0.0546875, # Probabilidad de que salga un 1
    0.1640625, # Probabilidad de que salga un 2
    0.2734375, # Probabilidad de que salga un 3
    0.2734375, # Probabilidad de que salga un 4
    0.1640625, # Probabilidad de que salga un 5
    0.0546875, # Probabilidad de que salga un 6
    0.0078125  # Probabilidad de que salga un 7
  ]
  print(tests.pearson_chi_squared_test(x, p, len(s), s))

# We don't know the mean
# Example 8.2 page 138
def example_chi_3():
  x = [0, 1, 2, 3, 4, 5] # Where 5 actually a group with all values greater than 5.
  s = [
    0, 0, 0, 0, 0, 0,
    1, 1,
    2,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 8
  ] # Experimental data.
  tetha = arr.mu(s) # Estimates Lambda, calculated by the experimental data.
  p = [
    cdist.poisson_PDF(tetha, 0), # P(X = 0) ningun accidente
    cdist.poisson_PDF(tetha, 1), # P(X = 1) 1 accidente
    cdist.poisson_PDF(tetha, 2), # P(X = 2) 2 accidentes
    cdist.poisson_PDF(tetha, 3), # P(X = 3) 3 accidentes
    cdist.poisson_PDF(tetha, 4), # P(X = 4) 4 accidentes
    1 - cdist.poisson_CDF(tetha, 5)
  ]
  N = { 0: 6, 1: 2, 2: 1, 3: 9, 4: 7, 5: 5 }
  # H0 is: Data is from Poisson distribution.
  p = tests.pearson_chi_squared_test_unknown_params(s, x, p, 0.05, 1, N, 30, True)
  print(p)

def example_kolsmir_1():
  kolmogorov_smirnov(
    [3, 1, 2, 5],
    1000,
    cdist.exponential_CDF,
    1/3
  )

def example_kolsmir_2():
  kolmogorov_smirnov(
    [55, 72, 81, 94, 112, 116, 124, 140, 145, 155],
    10000,
    cdist.exponential_CDF,
    1/100
  )