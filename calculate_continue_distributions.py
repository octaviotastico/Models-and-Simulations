import scipy.special as sp
import math

# returns P(X = x) for a uniform distribution
def uniform_PDF(a, b):
  return 1 / (b - a)

# returns P(X <= x) for a uniform distribution
def uniform_CDF(a, b, x):
  return max((x - a) / (b - a), 0) if x < b else 1

# returns P(X = x) for an exponential distribution
def exponential_PDF(l, x):
  return l * math.e**(-l*x)

# returns P(X <= x) for an exponential distribution
def exponential_CDF(l, x):
  return 1 - math.e**(-l*x)

# returns P(X = x) for a gamma distribution with parameters a=alfa and b=beta
def gamma_PDF(a, b, x):
  return (b**a * x**(a - 1) * math.e**(-b * x)) / math.gamma(a)

# returns P(X <= x) for a gamma distribution
def gamma_CDF(a, b, x):
  return sp.gammainc(a, b*x) / math.gamma(a)

# Returns P(X2_k = x)
def chi_square_PDF(k, x):
  return (x**((k/2)-1) * math.e**(-x/2)) / (2**(k/2) * math.gamma(k/2))

# Returns P(X2_k >= x)
# where X2_k is chi square with k degrees of liberty
def chi_square_CDF(k, x):
  return 1 - sp.gammainc(k/2, x/2)

# Returns P(X = x) for a t-student distribution
def t_student_PDF(v, m, s, x):
  left = math.gamma((v + 1) / 2) / (math.gamma(v / 2) * (math.pi * v) * s)
  right = (1 + (1 / v) * ((x - m) / s)**2)**(-(v + 1) / 2)
  return left * right
