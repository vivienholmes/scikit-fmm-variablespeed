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
        self.offset = [0]*len(size)
        if offset:
            self.offset = offset
        self.array_space = np.meshgrid(*[np.linspace(0, dimension, resolution + 1) for dimension in size])
        self.dx = size[0] / resolution
        self.dim = len(size)
        self.bilinear_interpolation_matrix = self.set_interpolation_matrix()

    def real_to_array(self, real_point):
        return np.array([round((x[0] - x[1]) / self.dx) for x in zip(real_point, self.offset)])

    def array_to_real(self, array_point):
        return array_point*self.dx + self.offset

    def distance(self, x_real, y_real):
        return np.sqrt(sum([difference**2 for difference in x_real - y_real]))

    def neighbours(self, x):
        base = self.real_to_array(x)
        print(base)
        print(x)
        neighbours = [base + i for i in list(it.product(range(0,2),repeat=self.dim))]
        if self.dim == 2:
            neighbours = [base, base + np.array([1,0]), base + np.array([0,1]), base + np.array([1,1])]
        if self.dim == 3:
            neighbours = [base, base + np.array([1,0,0]), base + np.array([0,1,0]), base + np.array([0,0,1]), base + np.array([1,1,0]), base+np.array([1,0,1]), base + np.array([0,1,1]), base+np.array([1,1,1])]
        print(neighbours)
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


def calculate_gradients(tau, bfield, domain, drivers, phi):
    drivers_points = [domain.real_to_array(index) for index in drivers.values()]
    ndim = tau.ndim
    gradient = [tau * np.nan] * ndim
    for indices,tau_value in np.ndenumerate(tau):
       for dim in range(ndim):
          if phi[indices] == 0:
              gradient[dim][indices] = 0
              continue
          either_side = []
          if indices[dim] + 1 < tau.shape[dim]:
             either_side.append(1)
          if indices[dim] > 0:
             either_side.append(-1)
          for delta_index in either_side:
               #for each neighbour on side delta_index if it's within bounds in dimension dim of tau_value in indices
               neighbour_array_index = tuple(x + delta_index if i == dim else x for i,x in enumerate(list(indices)))
               if tau[neighbour_array_index] < tau[indices]:
                   # either we are looking at a normal point or a new driver mutation:
                   # if the first, we only care about (older) neighbours with the same b value,
                   # if the second, we care about all its relevant neighbours
                   if bfield[neighbour_array_index] == bfield[indices] or indices in drivers_points:
                        #print(tau[neighbour_array_index])
                        gradient[dim][indices] = delta_index * (tau_value  - tau[neighbour_array_index])/domain.dx
                        #print(tau_value)
                        #print(gradient[dim][indices])
                        #print('-----------')
    return gradient

def estimate_gradient(x_0, domain, tau):
    print('estimating gradient at ' + str(x_0))
    dx = domain.dx
    neighbours = domain.neighbours(x_0)
    neighbouring_taus = {tuple((coordinate - neighbours[0]).tolist()): tau[*coordinate.tolist()] for coordinate in neighbours}
    print(neighbouring_taus)
    tau_list = np.array([[x] for x in neighbouring_taus.values()])
    print(tau_list)
    polynomial_coefs = (domain.bilinear_interpolation_matrix * np.matrix(tau_list)).tolist()
    print(polynomial_coefs)
    coefs = {x:y for x,y in zip(neighbouring_taus.keys(),polynomial_coefs)}
    print(coefs)
    estimated_gradient = None
    residuals = domain.get_distance_from_grid(x_0)
    print(residuals)
    x = residuals[0]
    y = residuals[1]
    print(x)
    print(coefs)
    print(coefs[(1,0)])
    if domain.dim == 2:
        estimated_gradient = [coelfs[(1,0)] + coefs[(1,1)]*y,coefs[(0,1)]+coefs[(1,1)]*x]
    if domain.dim == 3:
        z = residuals[2]
        estimated_gradient = [coefs[(1,0,0)] + coefs[(1,1,0)]*y + coefs[(1,0,1)]*z+coefs[(1,1,1)]*y*z,
                              coefs[(0,1,0)] + coefs[(1,1,0)]*x + coefs[(0,1,1)]*z + coefs[(1,1,1)]*x*z,
                              coefs[(0,0,1)]+coefs[(1,0,1)]*x + coefs[(0,1,1)]*y + coefs[(1,1,1)]*x*y]
    return estimated_gradient

def retrace_phylogeny(descendent, domain, tau, step=0.03, tau_min=0.1):
    descendent_indices = domain.real_to_array(descendent)
    current_point = descendent
    indices = descendent_indices
    ancestors = [descendent]
    print(tau[indices])
    while tau[*indices] > tau_min:
        estimated_gradient = estimate_gradient(current_point, domain, tau)
        delta_point = - estimated_gradient * step
        current_point += delta_point
        indices = domain.real_to_array(current_point)
        ancestors.append(current_point)
    return ancestors


def retrace_phylogenies(domain, gradients, descendent, speeds, bfield, tau, tau_min=0.1, step=0.05):
    descendent_indices = domain.real_to_array(descendent)
    print(descendent)
    print(descendent_indices)
    current_point = np.array(descendent,dtype=np.float64)
    ancestors = [descendent]
    current_indices = descendent_indices
    while tau[*current_indices] > tau_min:
       current_gradients = np.array([gradient[*current_indices] for gradient in gradients])
# / speeds[bfield[*current_indices]][*current_indices] ** 2
       current_point = [clamp(x, 0.0, domain.size[0]) for x in current_point - step * current_gradients]
       current_indices = domain.real_to_array(current_point)
       ancestors.append(current_point)
       print('current point: ' + str(current_point))
       print('current index' + str(current_indices))
       print('current gradient: ' + str(current_gradients))
       print('tau: ' + str(tau[*current_indices]))
       print('-------------------------')
    return ancestors


# resolution of grid for plots:
domain = Domain(1000, (2.0,2.0))

plt.figure()
X, Y = domain.array_space
phi = X**2 + Y**2
drivers = {}
#{1: [1.0, 1.0], 2: [0.2,0.3], 4: [0.6,0.1]} # a dictionary with n entries
speeds = [1+X*0]
#, 4+X**4, 3+X**4, 7*X**4,2+X**4,3+X**4,4+X**4,5+X**4] # a list of 2^n speed functions
num_drivers = len(drivers)
num_branches = 2 ** num_drivers

print(drivers)
# add white noise to speeds:
sigma = 0.001 # noise loudness
speeds = [np.random.normal(speed, sigma) for speed in speeds]
print('solving')
tau, bfield = skfmm.travel_time_genes(phi, drivers, speeds, dx=domain.dx)
bfield_max = bfield.max()


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
#plt.plot(*zip(*ancestors),color="red")
plt.gca().set_aspect(1)
plt.xticks([]); plt.yticks([])

plt.subplot(122)
plt.title("Travel time")
plt.contour(X, Y, phi, [0], colors='black', linewidths=(3))
plt.contour(X, Y, tau, 15)
#plt.plot(*zip(*ancestors),color="red")
plt.gca().set_aspect(1)
plt.xticks([]); plt.yticks([])

plt.savefig("phylo-time-and-branches.png")
print('plots done')


#print('calculating gradients')
#gradients = calculate_gradients(tau,bfield,domain,drivers,phi)
#print(tau)
#print(gradients)

#plt.figure()
#plt.title("X Gradients")
#plt.contour(X, Y, phi, [0], colors='black', linewidths=(3))
#plt.contour(X, Y, gradients[0], 15)
#plt.gca().set_aspect(1)
#plt.xticks([]); plt.yticks([])
#plt.savefig("xgradients.png")


#plt.figure()
#plt.title("Y Gradients")
#plt.contour(X, Y, phi, [0], colors='black', linewidths=(3))
#plt.contour(X, Y, gradients[1], 15)
#plt.gca().set_aspect(1)
#plt.xticks([]); plt.yticks([])
#plt.savefig("ygradients.png")

print('tracinging phlogeny')
ancestors = retrace_phylogeny((0.5,0.5),domain,tau)
#domain, gradients, [0.3,0.3],speeds,bfield,tau,step=0.5)
print(ancestors[:5])
