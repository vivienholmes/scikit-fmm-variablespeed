import skfmm
import numpy as np
import itertools as it
from Domain import Domain

class Problem:
    __init__(self, domain, drivers, phi, speeds):
        self.domain = domain
        self.drivers = drivers
        self.phi = phi
        self.speeds = speeds
        self.solution = self.Solution(self.domain, self.drivers)
        self.tau = np.full(domain.size, math.inf)
        self.beta = np.zeros(domain.size)

    def solve(self):
        self.tau, self.beta = skfmm.travel_time_genes(self.phi, self.drivers, self.speeds, dx=self.domain.dx

    def tau_at_point(self, real_point):
        return self.tau[tuple(self.domain.real_to_array(real_point))]

    def beta_at_point(self, real_point):
        return self.beta[tuple(self.domain.real_to_array(real_point))]


if __name__ == '__main__':

    domain = Domain(1000, (2.0,2.0))
    X, Y = domain.array_space
    phi = X**2 + Y**2
    drivers = {1: [-0.5,-0.5], 2: [-0.2,0.3], 4: [-0.6,-0.1]} # a dictionary with n entries
    speeds = [1+X**4, 2+X**4, 3+X**4, 4*X**4,5+X**4,6+X**4,7+X**4,8+X**4] # a list of 2^n speed functions
    num_drivers = len(drivers)
    num_branches = 2 ** num_drivers

    print(drivers)
    # add white noise to speeds:
    sigma = 0.01 # noise loudness
    speeds = [np.random.normal(speed, sigma) for speed in speeds]
    print('solving')
    problem = Problem(domain, drivers, phi, speeds)
    problem.solve()
