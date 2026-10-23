import numpy as np
import itertools as it

class Domain:
    def __init__(self, resolution, size, offset=None):
        self.resolution = resolution
        self.size = size
        self.offset = tuple([x / 2.0 for x in size])
        if offset:
            self.offset = offset
        self.array_space = np.meshgrid(*[np.linspace(0, dimension, resolution + 1) for dimension in size])
        self.dx = size[0] / resolution
        self.dim = len(size)
        self.bilinear_interpolation_matrix = self.set_interpolation_matrix()

    def periodic_add(self, a, b):
        return np.mod(a + np.array(b), np.array(self.size) + 1, dtype='i', casting='unsafe')

    def real_to_array(self, real_point):
        return np.array([round((x[1] + x[0]) / self.dx) for x in zip(real_point, self.offset)])

    def array_to_real(self, array_point):
        return array_point*self.dx - self.offset

    def distance(self, x_real, y_real):
        return np.sqrt(sum([difference**2 for difference in x_real - y_real]))

    def neighbours(self, x):
        base = self.real_to_array(x)
        #print(base)
        #print(x)
        neighbours = [base + i for i in list(it.product(range(0,2),repeat=self.dim))]
        if self.dim == 2:
            neighbours = [self.periodic_add(base, [0,0]), self.periodic_add(base, [1,0]), self.periodic_add(base, [0,1]), self.periodic_add(base, [1,1])]
        if self.dim == 3:
            neighbours = [base, base + np.array([1,0,0]), base + np.array([0,1,0]), base + np.array([0,0,1]), base + np.array([1,1,0]), base+np.array([1,0,1]), base + np.array([0,1,1]), base+np.array([1,1,1])]
        print('neighbours: ' + str(neighbours))
        return neighbours

    def set_interpolation_matrix(self):
        if self.dim == 2:
            return np.matrix([[1, 0, 0, 0], [-1, 0, 1, 0], [-1, 1, 0, 0],[1, -1, -1, 1]])
        if self.dim == 3:
            return np.matrix([[1,0,0,0,0,0,0,0],[-1,1,0,0,0,0,0,0],[-1,0,1,0,0,0,0,0],[-1,0,0,1,0,0,0,0],[1,-1,-1,0,1,0,0,0],[1,-1,0,-1,0,1,0,0],[1,0,-1,-1,0,0,1,0],[-1,1,1,1,-1,-1,-1,1]])
        return None

    def get_distance_from_grid(self, x_0):
        base = self.array_to_real(self.real_to_array(x_0))
        return x_0 - base

    def is_point_in_grid(self,x_0):
        for i in x_0:
            if i > self.resolution or i < 0: return False
        return True

if __name__ == '__main__':

    domain = Domain(1000, (2.0,2.0))

    X, Y = domain.array_space
    phi = X**2 + Y**2
    drivers = {1: [-0.5,-0.5], 2: [-0.2,0.3], 4: [-0.6,-0.1]} # a dictionary with n entries
    speeds = [1+X**4, 2+X**4, 3+X**4, 4*X**4,5+X**4,6+X**4,7+X**4,8+X**4] # a list of 2^n speed functions
    num_drivers = len(drivers)
    num_branches = 2 ** num_drivers

    print(drivers)
    print(domain.offset)
    print('offset')

    print('200 800 to real, expecting -0.6, 0.6, actual: ')
    print(domain.array_to_real(np.array([200, 800])))

    print('-0.6, 0.6 to array, expecting [200, 800], actual: ')
    print(domain.real_to_array(np.array([-0.6, 0.6])))

    print('expecting 150, 300, actual: ')
    print(domain.real_to_array(domain.array_to_real(np.array([150, 300]))))

    print('expecting [0.6, -0.2], actual: ')
    print(domain.array_to_real(domain.real_to_array(np.array([0.6, -0.2]))))
