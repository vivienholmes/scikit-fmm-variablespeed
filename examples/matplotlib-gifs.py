import skfmm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# resolution of grid for plots:
taps = 1000
x_width = 2.0
y_width = 2.0

plt.figure()
X, Y = np.meshgrid(np.linspace(-0.5 * x_width, +0.5 * x_width, taps + 1), 
                   np.linspace(-0.5 * y_width, +0.5 * y_width, taps + 1))
phi = (X)**2+(Y)**2
drivers = {1: [0.5, 0], 2: [-0.2,0.3], 4: [-0.4,0.6]} # a dictionary with n entries
speeds = [1+X**4, 4+X**4, 3+X**4, 7*X**4,2+X**4,3+X**4,4+X**4,5+X**4] # a list of 2^n speed functions
num_drivers = len(drivers)
num_branches = 2 ** num_drivers

print(drivers)

tau, bfield = skfmm.travel_time_genes(phi, drivers, speeds, dx=x_width/taps)

plt.figure()
num_frames = 50
time_min = 0
time_max =  0.8
time_steps = np.linspace(time_min,time_max,num_frames)
frames = []

for time_threshold in time_steps:
	fig,ax=plt.subplots(dpi=80,figsize=(12,12))
	ax.contour(tau,levels=[time_threshold],colors=['red'],linewidths=2.5)
	current_bfield = bfield.copy()
	current_bfield[tau > time_threshold] = num_branches + 1
	ax.contourf(current_bfield,levels=num_branches)
	ax.set_title(f'Elapsed time: {time_threshold:.2f}')
	buf = io.BytesIO()
	fig.savefig(buf,format='png',bbox_inches='tight',dpi=80)
	buf.seek(0)
	frames.append(Image.open(buf).convert('RGB'))
	plt.close(fig)

frames[0].save('travel_time.gif',save_all=True,append_images=frames[1:],duration=250,loop=0)

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

plt.savefig("time-and-branches.png")
