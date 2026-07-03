import skfmm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib import cm



def cartesian_to_rowcol(cartesian):
     rowcol = [taps - int(taps*(index + (x_width / 2))/x_width) for index in cartesian]
     rowcol[0] = taps - rowcol[0]
     return rowcol

def calculate_gradients(tau, bfield,dx,drivers):
    drivers_points = [cartesian_to_rowcol(index) for index in drivers.values()]
    ndim = tau.ndim
    gradient = [tau * np.nan] * ndim
    for indices,tau_value in np.ndenumerate(tau):
       for dim in range(ndim):
          either_side = []
          if indices[dim] + 1 < tau.shape[dim]:
             either_side.append(1)
          if indices[dim] > 0:
             either_side.append(-1)
          for delta_index in either_side:
               #for each neighbour on side delta_index if it's within bounds in dimension dim of tau_value in indices
               neighbouring_indices = tuple(x + delta_index if i == dim else x for i,x in enumerate(list(indices)))
               if tau[neighbouring_indices] < tau[indices]:
                   # either we are looking at a normal point or a new driver mutation:
                   # if the first, we only care about (older) neighbours with the same b value,
                   # if the second, we care about all its relevant neighbours
                   if bfield[neighbouring_indices] == bfield[indices] or indices in drivers_points:
                      	gradient[dim][indices] = (tau_value - tau[neighbouring_indices])/dx
    return gradient

def retrace_phylogenies(gradients, descendent, speeds, bfield, tau, tau_min=0.0, step=0.01):
    descendent_indices = cartesian_to_rowcol(descendent)
    print(descendent)
    print(descendent_indices)
    current_point = np.array(descendent,dtype=np.float64)
    ancestors = [descendent_indices]
    current_indices = descendent_indices
    while tau[*current_indices] > tau_min:
       current_gradients = np.array([gradient[*current_indices] for gradient in gradients])
       gradient_descent = - speeds[bfield[*current_indices]][*current_indices] ** 2 * step * current_gradients
       current_point += gradient_descent
       current_indices = cartesian_to_rowcol(current_point)
       ancestors.append(current_indices)
    return ancestors

# resolution of grid for plots:
taps = 1000
x_width = 2.0
y_width = 2.0
dx=x_width/taps

plt.figure()
X, Y = np.meshgrid(np.linspace(-0.5 * x_width, +0.5 * x_width, taps + 1), 
                   np.linspace(-0.5 * y_width, +0.5 * y_width, taps + 1))
phi = (Y)**2
drivers = {1: [0.5, 0], 2: [-0.2,0.3], 4: [-0.4,0.6]} # a dictionary with n entries
speeds = [1+X**4, 4+X**4, 3+X**4, 7*X**4,2+X**4,3+X**4,4+X**4,5+X**4] # a list of 2^n speed functions
num_drivers = len(drivers)
num_branches = 2 ** num_drivers

print(drivers)
# add white noise to speeds:
sigma = 0.03 # noise loudness
speeds = [np.random.normal(speed, sigma) for speed in speeds]
print('solving')
tau, bfield = skfmm.travel_time_genes(phi, drivers, speeds, dx=dx)
bfield_max = bfield.max()

print('calculating gradients')
gradients = calculate_gradients(tau,bfield,dx,drivers)
print(tau)
print(gradients)
print('tracinging phlogeny')
ancestors = retrace_phylogenies(gradients, [-1,0.3],speeds,bfield,tau)
print(ancestors[:5])

print('plotting gif')
plt.figure()
num_frames = 50
time_min = 0
time_max =  0.8
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

print(bfield)
print(tau)

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
print('done')
