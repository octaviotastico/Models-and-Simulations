import calculate_discrete_distributions as ddist
import calculate_continue_distributions as cdist
import simulate_discrete_variables as sdvar
import simulate_continue_variables as scvar
import arrays as arr
import tests

import numpy as np
import time
import math

def g6_ex1():
  normals = [ scvar.normal_reject_method(0, 1) ]
  mean, std = normals[0], 0
  while len(normals) < 30 or std / len(normals)**(1 / 2) < 1 / 10:
    X, Y = scvar.normal_box_muller(0, 1)
    normals += [ X, Y ]
    std = arr.sigma(normals, True)
  print(f'G6 EX1 - {len(normals)} normals generated, mean {arr.mu(normals)}, var {arr.sigma(normals)}')


# g6_ex1()