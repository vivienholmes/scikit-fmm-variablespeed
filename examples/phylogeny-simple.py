import skfmm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib import cm
import itertools as it
from Domain import Domain

clamp = lambda x, l, u: l if x < l else u if x > u else x

def basic_phylogeny(origin, domain, drivers, beta, tau):
    drivers[0] = origin
    print(beta)
    print(beta[tuple(domain.real_to_array([0.3,-0.2]))])
    print(beta[tuple(domain.real_to_array([-0.1,-0.6]))])
    print(tau)
    driver_results = [(position, tau[tuple(domain.real_to_array(position))], beta[tuple(domain.real_to_array(position))]) for position in drivers.values()]
    print(driver_results)
    parents = []
    for d in driver_results:
        current_position, current_tau, current_beta = d
        best_tau = 0
#        print(current_position)
        for d2 in driver_results:
            trial_position, trial_tau, trial_beta = d2
#            print('trial_beta')
#            print(trial_beta)
#            print(trial_position)
            if trial_beta & current_beta == trial_beta:
#                print('equal')
                if trial_tau > best_tau:
#                    print('best')
                    best_tau = trial_tau
                    mrca = trial_position
        parents.append((domain.real_to_array(current_position), domain.real_to_array(mrca)))
#        print(parents)
    return parents

# resolution of grid for plots:
domain = Domain(1000, (2.0,2.0))

plt.figure()
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
tau, bfield = skfmm.travel_time_genes(phi, drivers, speeds, dx=domain.dx)
bfield_max = bfield.max()

print('tracing phylogeny')
driver_phylogeny = basic_phylogeny([-1,-1],domain, drivers, bfield, tau)
print(driver_phylogeny)

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
	ax.contourf(current_bfield,levels=num_branches,vmin=0,vmax=bfield_max,zorder=1)
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
for ancestry in driver_phylogeny:
    print(ancestry)
    print(*zip(*ancestry))
    plt.plot(*zip(*ancestry), 'ro-')
plt.gca().set_aspect(1)
plt.xticks([]); plt.yticks([])

plt.subplot(122)
plt.title("Travel time")
plt.contour(X, Y, phi, [0], colors='black', linewidths=(3))
plt.contour(X, Y, tau, 15)
plt.gca().set_aspect(1)
plt.xticks([]); plt.yticks([])

plt.savefig("phylo-time-and-branches.png")
print('plots done')
