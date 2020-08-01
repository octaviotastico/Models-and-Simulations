from arrays import frequency
from calculate_continue_distributions import chi_square_CDF

##### Chi-squared test

# Returns chi squared T statistic
def chi_squared_statistic(m, x, p):
  N, T, n = frequency(m), 0, len(m)
  for i in range(len(x)):
    T += (((N[x[i]] if x[i] in N.keys() else 0) - n*p[i])**2) / (n*p[i])
  return T

# Simulates p-value for a pearson chi square
# test for a discrete random variable
def pearson_chi_squared_test(m, x, p, alpha):
  T = chi_squared_statistic(m, x, p)
  p_value = chi_square_CDF(len(x) - 1, T)

  return p_value, 'Reject H0, pval<alfa' if (p_value <= alpha) else 'Dont reject H0, pval>alfa'

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
