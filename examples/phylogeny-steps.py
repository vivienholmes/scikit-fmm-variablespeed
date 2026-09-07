import skfmm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib import cm
import itertools as it

clamp = lambda x, l, u: l if x < l else u if x > u else x

class Domain:
    def __init__(self, resolution, size, offset=None):
        self.resolution = resolution
        self.size = size
        self.offset = np.array([0]*len(size))
        if offset:
            self.offset = offset
        self.array_space = np.meshgrid(*[np.linspace(0, dimension, resolution + 1) for dimension in size])
        self.dx = size[0] / resolution
        self.dim = len(size)
        self.bilinear_interpolation_matrix = self.set_interpolation_matrix()

    def real_to_array(self, real_point):
        return np.array([round((x[0] - x[1]) / self.dx) for x in zip(real_point, self.offset)])

    def array_to_real(self, array_point):
        return np.multiply(array_point,self.dx) + self.offset

    def distance(self, x_real, y_real):
        return np.sqrt(sum([difference**2 for difference in x_real - y_real]))

    def grid_neighbours(self, x_0, grid=False):
        if self.dim == 2:
            neighbours = [tuple(x_0 + np.array(x)) for x in [[-1,0],[1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,-1],[-1,1]]]
        if self.dim == 3:
            neighbours = [tuple(x_0 + np.array(x)) for x in [[-1,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]]
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
            if i > self.resolution:
                return False
            if i < 0:
                return False
        return True


def step_back_through_grid(x_0, tau, domain):
    ancestors = []
    current_point = tuple(domain.real_to_array(x_0))
    current_tau = tau[current_point]
    while current_tau > 0:
        ancestors.append(domain.array_to_real(current_point))
        neighbours = domain.grid_neighbours(current_point)
        prev_neighbours = {neighbour: tau[tuple(neighbour)] for neighbour in neighbours if domain.is_point_in_grid(neighbour) and tau[tuple(neighbour)] < current_tau}
        current_point = max(prev_neighbours, key=prev_neighbours.get)
        current_tau = tau[current_point]
    return ancestors


# resolution of grid for plots:
domain = Domain(1000, (2.0,2.0))

plt.figure()
X, Y = domain.array_space
phi = X**2 + Y**2
drivers = {1: [-0.5, -0.5], 2: [-0.2,-0.3], 4: [-0.6,-0.1]} # a dictionary with n entries
speeds = [1+X**2, 2+X**2, 3+X**2, 4*X**2,5+X**2,6+X**2,7+X**2,8+X**2] # a list of 2^n speed functions
num_drivers = len(drivers)
num_branches = 2 ** num_drivers

print(drivers)
# add white noise to speeds:
sigma = 0.001 # noise loudness
speeds = [np.random.normal(speed, sigma) for speed in speeds]
print('solving')
tau, bfield = skfmm.travel_time_genes(phi, drivers, speeds, dx=domain.dx)
bfield_max = bfield.max()

ancestors = step_back_through_grid([1.8,1.8], tau, domain)

print('plotting gif')
plt.figure()
num_frames = 50
time_min = 0
time_max =  tau.max()
time_steps = np.linspace(time_min,time_max,num_frames)
frames = []

for time_threshold in time_steps:
	fig,ax=plt.subplots(dpi=80,figsize=(12,12))
	ax.contour(tau,levels=[time_threshold],colors=['red'],linewidths=2.5)
	current_bfield = np.ma.array(bfield.copy())
	current_bfield[tau > time_threshold] = np.ma.masked
	ax.contourf(current_bfield,levels=num_branches,vmin=0,vmax=bfield_max)
	ax.set_title(f'Elapsed time: {time_threshold:.2f}') 
	buf = io.BytesIO()
	fig.savefig(buf,format='png',bbox_inches='tight',dpi=80)
	buf.seek(0)
	frames.append(Image.open(buf).convert('RGB'))
	plt.close(fig)

frames[0].save('phylo_travel_time.gif',save_all=True,append_images=frames[1:],duration=250,loop=0)

print('plotting graphs')

plt.subplot(121)
plt.title("branch function")
plt.contourf(bfield, levels=num_branches)
plt.plot(*zip(*ancestors),color="red")
plt.gca().set_aspect(1)
plt.xticks([]); plt.yticks([])

plt.subplot(122)
plt.title("Travel time")
plt.contour(X, Y, phi, [0], colors='black', linewidths=(3))
plt.contour(X, Y, tau, 15)
plt.plot(*zip(*ancestors),color="red")
plt.gca().set_aspect(1)
plt.xticks([]); plt.yticks([])

plt.savefig("phylo-time-and-branches.png")
print('plots done')
