import skfmm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from matplotlib import cm

<<<<<<< HEAD
=======
def rowcol_to_space(x,y,dx,width):
	return (x*dx + 0.5 * width,0.5 * width-y*dx)

def space_to_rowcol(x,y,dx,width):
	return (x+0.5*width,y-0.5*width)

def add_maze_wall(a,b,speeds,epsilon):
	taps=speeds[0].shape[0]
	distance_ab = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
	wall_points = []
	for x in range(taps):
		for y in range(taps):
			x,y = rowcol_to_space(x,y,)
			distance_ax = np.sqrt((x-a[0])**2+(y-a[1])**2)
			distance_xb = np.sqrt((x-b[0])**2+(y-b[1])**2)
			if distance_ax + distance_xb < distance_ab + epsilon:
				wall_points.append((x,y))

	for (x,y) in wall_points:
		for i in range(len(speeds)):
			speeds[i][x][y] = 0

	return speeds

>>>>>>> 47ff3bb (add front coarsening test)
# resolution of grid for plots:
taps = 1000
x_width = 2.0
y_width = 2.0

plt.figure()
# NB: origin (0,0) should be in the centre of the space:
X, Y = np.meshgrid(np.linspace(-0.5 * x_width, +0.5 * x_width, taps + 1), 
                   np.linspace(-0.5 * y_width, +0.5 * y_width, taps + 1))
phi = (Y)**2+(X)**2
drivers = {1: [0.5, 0], 2: [-0.2,0.3], 4: [-0.4,0.6]} # a dictionary with n entries
speeds = [1+X**4, 4+X**4, 3+X**4, 7*X**4,2+X**4,3+X**4,4+X**4,5+X**4] # a list of 2^n speed functions
num_drivers = len(drivers)
num_branches = 2 ** num_drivers

print(drivers)
# add white noise to speeds:
sigma = 0.1 # noise loudness
speeds = [np.random.normal(speed, sigma) for speed in speeds]

# build a maze:
def dist(a, b):
   # return distance between two (real) points on a plane (2D):
   return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def add_maze_wall(a, b, speeds, X_mesh, Y_mesh, epsilon):
	dsq_ab = dist(a, b)
	dx = (X_mesh[0][1] - X_mesh[0][0])
	x_taps = X_mesh.shape[0]
	y_taps = X_mesh.shape[1]
	wall_points = []
	for x, y in np.nditer([X_mesh, Y_mesh]):
		if ((dist((x,y), a) + dist((x,y), b)) < (np.sqrt(dsq_ab) + epsilon)):
			wall_points.append((x,y))

	for (x,y) in wall_points:
		for i in range(len(speeds)):
			row = int(x / dx + 0.5 * x_taps) % x_taps
			col = int(y / dx + 0.5 * y_taps) % y_taps
			speeds[i][row][col] = 0

	return speeds

speeds = add_maze_wall((0.3,0.3),(0.3,0.6),speeds,X, Y, -0.15)

# solve the system:
tau, bfield = skfmm.travel_time_genes(phi, drivers, speeds, dx=x_width/taps)
bfield_max = bfield.max()

#plot the solution:
speeds = add_maze_wall((0.3,0.3),(0.3,0.6),speeds,10)

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

frames[0].save('maze.gif',save_all=True,append_images=frames[1:],duration=250,loop=0)

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

plt.savefig("maze-time-and-branches.png")
