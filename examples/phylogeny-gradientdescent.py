import skfmm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib import cm

from scipy.interpolate import RegularGridInterpolator

# resolution of grid for plots:
taps = 1000
x_width = 2.0
y_width = 2.0

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
tau, bfield = skfmm.travel_time_genes(phi, drivers, speeds, dx=x_width/taps)
bfield_max = bfield.max()

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

frames[0].save('travel_time.gif',save_all=True,append_images=frames[1:],duration=250,loop=0)

# before plotting phi, tau and branch, choose a point and retrace its line of
# descent

def retrace_line_of_descent(child, tau):
    # child is the chosen point. what is its tau value (time of arrival)?
    # assuming that initial time=0, parametrise the line with 0 < time < tau
    grad_y, grad_x = np.gradient(tau)

    # Interpolate gradient at (x, y)    
    grad_x_interp = RegularGridInterpolator((Y, X), grad_x, method='linear')    
    grad_y_interp = RegularGridInterpolator((Y, X), grad_y, method='linear')

    ancestor=child
    learning_rate = 0.01
    line_of_descent = []
    time = tau[child] #?
    epsilon = 0.1
    while (time > 0 + epsilon): # walk backwards to each of your ancestors
        x,y = ancestor
        gx = grad_x_interp([y, x])[0]    
        gy = grad_y_interp([y, x])[0]        
        # Update step (move opposite to gradient)    
        x -= learning_rate * gx    
        y -= learning_rate * gy
        ancestor = (x, y)
        time = tau[ancestor]
        line_of_descent.append((x,y))

    return line_of_descent

descent1 = retrace_line_of_descent((0.75,0.75), tau)

plt.subplot(121)
#plt.title("Zero-contour of phi")
#plt.contour(X, Y, phi, [1e-6], colors='black', linewidths=(3))
plt.title("branch function")
plt.contourf(bfield, levels=num_branches)
plt.gca().set_aspect(1)
plt.xticks([]); plt.yticks([])

plt.subplot(122)
plt.title("Travel time")
plt.contour(X, Y, phi, [0], colors='black', linewidths=(3))

plt.contour(X, Y, tau, 15)
plt.gca().set_aspect(1)
plt.xticks([]); plt.yticks([])

plt.savefig("phylo-time-and-branches.png")
