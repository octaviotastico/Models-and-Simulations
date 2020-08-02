from tests import chi_squared_statistic, pearson_chi_squared_test, pearson_chi_squared_test_unknown_params
from calculate_continue_distributions import poisson_PDF, poisson_CDF
from arrays import mu

def example_chi_1():
  m = [1, 1, 1, 2, 3, 4, 5] # Muestra
  x = [1, 2, 3, 4, 5] # Posibles X
  p = [0.3, 0.1, 0.2, 0.1, 0.3] # Probabilidades de x
  print(chi_squared_statistic(m, x, p))

def example_chi_2():
  m = [
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
  print(pearson_chi_squared_test(m, x, p, 0.05))

# We don't know the mean
# Example 8.2 page 138
def example_chi_3():
  x = [0, 1, 2, 3, 4, 5] # Where 5 actually a group with all values greater than 5.
  m = [
    0, 0, 0, 0, 0, 0,
    1, 1,
    2,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 8
  ] # Experimental data.
  tetha = mu(m) # Estimates Lambda, calculated by the experimental data.
  p = [
    poisson_PDF(tetha, 0), # P(X = 0) ningun accidente
    poisson_PDF(tetha, 1), # P(X = 1) 1 accidente
    poisson_PDF(tetha, 2), # P(X = 2) 2 accidentes
    poisson_PDF(tetha, 3), # P(X = 3) 3 accidentes
    poisson_PDF(tetha, 4), # P(X = 4) 4 accidentes
    1 - poisson_CDF(tetha, 5)
  ]
  N = { 0: 6, 1: 2, 2: 1, 3: 9, 4: 7, 5: 5 }
  # H0 is: Data is from Poisson distribution.
  p = pearson_chi_squared_test_unknown_params(m, x, p, 0.05, 1, N, 30, True)
  print(p)
