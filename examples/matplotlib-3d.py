import skfmm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# resolution of grid for plots:
taps = 100
x_width = 5.0
y_width = 5.0
z_width = 5.0
plt.figure()
X, Y, Z = np.meshgrid(np.linspace(-0.5 * x_width, +0.5 * x_width, taps + 1), 
                   np.linspace(-0.5 * y_width, +0.5 * y_width, taps + 1),
		   np.linspace(-0.5 * z_width, +0.5 * z_width, taps + 1))
phi = 0.02-(X)**2 - Y**2 - Z**2
drivers = {1: [0.5, 0.05, 0.01], 2: [-0.5, 0.1, -0.1],4: [-0.1,-0.1,-0.1]} # a dictionary with n entries
base_speeds = [1+X**2, 2+X**2, 3+X**2, 4+X**2, 5+Z**2,6+Z**2,7+Z**2,8+Z**2] # a list of 2^n speed functions
num_drivers = len(drivers)
num_branches = 2 ** num_drivers
print(drivers)

coordinates = []
for x in range(taps):
	for y in range(taps):
		for z in range(taps):
			coordinates.append((x,y,z))
inverse_cylinder = [(x,y,z) for (x,y,z) in coordinates if abs((x-0.5*taps)**2+(y-0.5*taps)**2-40) > 30]

for (x,y,z) in inverse_cylinder:
	for i in range(num_branches):
		base_speeds[i][x][y][z] /= 3


# add white noise to speeds:
sigma = 0.12 # noise loudness
speeds = [np.random.normal(speed, sigma) for speed in base_speeds]
#for (x,y,z) in inverse_cylinder:
#	for i in range(num_branches):
#		if x * y * z == 0 or max([x,y,z]) >= taps:
#			speeds[i][x][y][z] = 0

tau, bfield = skfmm.travel_time_genes(phi, drivers, speeds, dx=x_width/taps)

plt.figure()
num_frames = 30
rotation_speed = 360/num_frames
time_min = 0
time_max =  0.8
time_steps = np.linspace(time_min,time_max,num_frames)
frames = []

middle = int(2 * taps / 3)

cmap = cm.get_cmap('tab20')
region_colours = {region_id: cmap(region_id % cmap.N) for region_id in range(num_branches)}

for frame_id, time_threshold in enumerate(time_steps):
    fig = plt.figure(dpi=80, figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a masked array for current state
    current_bfield = bfield.copy()
    current_bfield[tau > time_threshold] = num_branches + 1
    
    for region_id in np.unique(current_bfield):
        if region_id >= num_branches + 1:  # skip "outside" region
            continue
        coords = np.argwhere(current_bfield == region_id)
        if len(coords) > 0:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                      s=1, alpha=0.6, color=region_colours[int(region_id)], label=f'Region {int(region_id)}')
    
    # Set equal aspect ratio and limits
    ax.set_xlim([0, taps])
    ax.set_ylim([0, taps])
    ax.set_zlim([0, taps])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Elapsed time: {time_threshold:.2f}')
    ax.legend()

    azimuth = frame_id * rotation_speed
    elevation = 20
    ax.view_init(elev=elevation, azim=azimuth)
#Save frame to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
    buf.seek(0)
    frames.append(Image.open(buf).convert('RGB'))
    plt.close(fig)

    print(f"Frame {len(frames)}/{num_frames}")

# Save as GIF
frames[0].save('travel_time_3d.gif', save_all=True, append_images=frames[1:], 
               duration=250, loop=0)



plt.subplot(121)
#plt.title("Zero-contour of phi")
#plt.contour(X, Y, phi, [1e-6], colors='black', linewidths=(3))
plt.title("branch function")
plt.contourf(bfield[:,:,middle].squeeze(), levels=num_branches)
plt.gca().set_aspect(1)
plt.xticks([]); plt.yticks([])

plt.subplot(122)
plt.title("Travel time")
plt.contour(X[:,:,middle].squeeze(), Y[:,:,middle].squeeze(), phi[:,:,middle].squeeze(), [0], colors='black', linewidths=(3))

plt.contour(X[:,:,middle].squeeze(), Y[:,:,middle].squeeze(), tau[:,:,middle].squeeze(), 15)
plt.gca().set_aspect(1)
plt.xticks([]); plt.yticks([])

plt.savefig("time-and-branches-slice.png")
