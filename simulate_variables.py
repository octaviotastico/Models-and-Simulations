import numpy as np

# Returns an uniform variable
# between the interval [a, b]
def uniform(a, b):
  u = np.random.uniform(0, 1)
  return int(u * (b - a + 1)) + a

# Returns a geometric variable
# with parameters n, p
def geometric(n, p):
  u = np.random.uniform(0, 1)
  return int(np.log(u)/np.log(1 - p)) + 1
