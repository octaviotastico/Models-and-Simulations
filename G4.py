import calculate_discrete_distributions as ddist
import calculate_continue_distributions as cdist
import simulate_discrete_variables as sdvar
import simulate_continue_variables as scvar
import arrays as arr
import tests

import numpy as np
import time
import math

### Guia 4

def g4_ex1(): # Coincidence
  coincidences = 0
  sims, success = 10000, 0
  cards = list(range(1, 101))
  for sim in range(sims):
    cards = arr.permutation(cards)
    for i in range(1, 101):
      if cards[i - 1] == i:
        coincidences += 1
      if i == coincidences == 10:
        success += 1
  print(f'G4 EX1 - {success} Successes')
  print(f'Total iterations: {sims}')
  print(f'Total amount of coincidences: {coincidences}')
  print(f'E(x) = {coincidences/sims}')

def g4_ex2(): # Montecarlo mean
  def f(x):
    return math.e ** (x / 10000)

  s1 = time.perf_counter_ns()
  exact_amount = 0
  for i in range(10001):
    exact_amount += f(i)
  e1 = time.perf_counter_ns()

  s2 = time.perf_counter_ns()
  aproximation = arr.montecarlo_mean(100, 10000, f)
  e2 = time.perf_counter_ns()

  print(f'G4 EX2')
  print(f'Exact amount {exact_amount} vs Montecarlo Mean {aproximation}')
  print(f'Time spent in exact amount {(e1 - s1)/1000000}ms vs time spent in Montecarlo Mean {(e2 - s2)/1000000}ms')

def g4_ex3(): # Uniform sum
  sims = 10000
  success = []
  for _ in range(sims):
    dices, throws = set(), 0
    while len(list(dices)) < 11:
      dices.add(sdvar.uniform(1, 6) + sdvar.uniform(1, 6))
      throws += 1
    success.append(throws)
  print('G4 EX3')
  print(f'Total iterations: {sims}')
  print(f'Success average: {arr.mu(success)}')
  print(f'N > 15: {arr.proportion(success, 15, ">")}, N < 10: {arr.proportion(success, 10, "<")}')

def g4_ex4_ar(): # Accept Reject Method
  px = [ 0.15, 0.20, 0.10, 0.35, 0.20 ]
  py = [ ddist.binomial_PDF(4, 0.45, i) for i in range(5) ]
  sim_x = sdvar.accept_reject(px, py, sdvar.binomial, 4, 0.45)
  print(f'G4 EX4 Accept/Reject Method - X Simulated: {sim_x}')

def g4_ex4_it(): # Inverse Transform Method
  px = [ 0.15, 0.20, 0.10, 0.35, 0.20 ]
  sim_x = sdvar.inverse_transform(px)
  print(f'G4 EX4 Inverse Transform Method - X Simulated: {sim_x + 1}')

def g4_ex5_ar(): # Accept Reject Method
  px = [ 0.11, 0.14, 0.09, 0.08, 0.12, 0.10, 0.09, 0.07, 0.11, 0.09 ]
  py = [ cdist.uniform_PDF(0, 100) for i in range(100) ]
  sim_x = sdvar.accept_reject(px, py, sdvar.uniform, 0, 9)
  print(f'G4 EX5 Accept/Reject Method - X Simulated: {sim_x}')

def g4_ex5_it(): # Inverse Transform Method
  px = [ 0.11, 0.14, 0.09, 0.08, 0.12, 0.10, 0.09, 0.07, 0.11, 0.09 ]
  sim_x = sdvar.inverse_transform(px)
  print(f'G4 EX5 Inverse Transform Method - X Simulated: {sim_x + 1}')

def g4_ex5_urn(): # Urn method
  px = [ 0.11, 0.14, 0.09, 0.08, 0.12, 0.10, 0.09, 0.07, 0.11, 0.09 ]
  sim_x = sdvar.urn(px)
  print(f'G4 EX5 Urn Method - X Simulated: {sim_x + 1}')

def g4_ex6_it(): # Inverse Transform Method
  px = [ ddist.binomial_PDF(10, 0.3, i) for i in range(10) ]
  sim_x = sdvar.inverse_transform(px)
  print(f'G4 EX5 Inverse Transform Method - X Simulated: {sim_x + 1}')

def g4_ex6_bernoulli(): # N Bernoulli Simulation
  s = sdvar.n_bernoulli(10, 0.3)
  print(f'G4 EX6 N-Bernoulli Simulation - X Simulated: {np.sum(s)}')

def g4_ex7_it(): # Inverse Transform Method
  success, sims = 0, 1000
  px = [ ddist.poisson_PDF(0.7, i) for i in range(3) ] # Get P(x=0), P(x=1), P(x=2)
  px += [ 1 - np.sum(px) ] # P(x>2) el resto
  for _ in range(sims):
    x = sdvar.inverse_transform(px)
    if x > 2:
      success += 1
  print(f'G4 EX7 Transform Method - P(X > 2) = {success / sims}')

def g4_ex7_proportion(): # Proportion - Equivalent?
  success, sims = 0, 1000
  for _ in range(sims):
    Y = sdvar.poisson(0.7)
    if Y > 2:
      success += 1
  print(f'G4 EX7 Using Proportion - P(X > 2) = {success / sims}')

def g4_ex8(): # Inverse Transform Method
  success, sims = 0, 1000

  def f(l, k, x):
    top = math.e**(-l) * (l**x / math.factorial(x))
    bottom = 0
    for i in range(k):
      bottom += math.e**(-l) * (l**i / math.factorial(i))
    return top / bottom

  px = [ f(0.7, 10, i) for i in range(3) ] # Get P(x=0), P(x=1), P(x=2)
  px += [ 1 - np.sum(px) ] # P(x>2) el resto
  for _ in range(sims):
    x = sdvar.inverse_transform(px)
    if x > 2:
      success += 1
  print(f'G4 EX8 - P(X > 2) = {success / sims}')

def g4_ex9(): # Composition of two Geometrics
  geom_A = sdvar.geometric(1/2)
  geom_B = sdvar.geometric(1/3)
  print(f'G4 EX9 - X Simulated: {1/2 * geom_A + 1/2 * geom_B}')

def g4_ex10_rit(): # Recursive Inverse Transform
  p = 0.2
  F, rec, i = p, p, 0
  u = np.random.uniform(0, 1)
  while u > F:
    i += 1
    rec *= (1 - p)
    F += rec
  print(f'G4 EX10 Inverse Transform Method - X Simulated: {i}')

def g4_ex10_bernoulli(): # Recursive Inverse Transform
  p, sim_x = 0.2, 0
  while sdvar.bernoulli(p) == 0:
    sim_x += 1
  print(f'G4 EX10 Generating Bernoullis - X Simulated: {sim_x}')

# g4_ex1()
# g4_ex2()
# g4_ex3()
# g4_ex4_ar()
# g4_ex4_it()
# g4_ex5_ar()
# g4_ex5_it()
# g4_ex5_urn()
# g4_ex6_it()
# g4_ex6_bernoulli()
# g4_ex7_it()
# g4_ex7_proportion()
# g4_ex8()
# g4_ex9()
# g4_ex10_rit()
# g4_ex10_bernoulli()