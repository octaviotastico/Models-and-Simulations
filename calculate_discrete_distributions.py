import math

# returns P(X = x) for a bernoulli (binomial with n=1) distribution
def bernoulli_PDF(p, x):
  return p**x * (1 - p)**(1 - x)

# returns P(X < x) for a bernoulli distribution
# for P(X <= x) use x+1, and for P(X > x) use 1-(X <= x)
def bernoulli_PDF(p, x):
  summ = 0
  for i in range(x):
    summ += bernoulli_PDF(p, i)
  return summ

# returns P(X = x) for a binomial distribution
def binomial_PDF(n, p, x):
  return math.comb(n, x) * p**x * (1 - p)**(n - x)

# returns P(X < x) for a binomial distribution
# for P(X <= x) use x+1, and for P(X > x) use 1-(X <= x)
def binomial_CDF(n, p, x):
  summ = 0
  for i in range(x):
    summ += binomial_PDF(n, p, i)
  return summ

# returns P(X = x) for a poisson distribution
def poisson_PDF(l, x):
  return math.e**(-l) * l**x / math.factorial(x)

# returns P(X < x) for a poisson distribution
# for P(X <= x) use x+1, and for P(X > x) use 1-(X <= x)
def poisson_CDF(l, x):
  e, summ = math.e**(-l), 0
  for i in range(x):
    summ += l**i / math.factorial(i)
  return e * summ

# returns P(X = x) for a geometric distribution
def geometric_PDF(p, x):
  return (1 - p)**(x - 1) * p

# returns P(X < x) for a geometric distribution
# for P(X <= x) use x+1, and for P(X > x) use 1-(X <= x)
def geometric_CDF(p, x):
  return 1 - (1 - p)**x