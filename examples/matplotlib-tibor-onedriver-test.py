import skfmm
import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams['text.usetex'] = True

# resolution of grid for plots:
taps = 140
x_width = 6.0
y_width = 6.0

fig, ax = plt.subplots(1,2)
X, Y = np.meshgrid(np.linspace(-0.5 * x_width, +0.5 * x_width, taps + 1), 
                   np.linspace(-0.5 * y_width, +0.5 * y_width, taps + 1))
phi = (X)**2+(Y)**2
drivers = {1: [1.0, 0]} # a dictionary with n entries
v_spd = 1.555 # TODO speeds smaller than 3ish appear to create weird artifacts
# for low speeds, fewer steps (and bigger dx values) appear to be better?
# TODO: output b-fields, i suspect these are not updated properly when speeds
# are "too low" or too close to one another.
speeds = [np.ones((taps+1, taps+1)), v_spd * np.ones((taps+1, taps+1))] # a list of 2^n speed functions

tau, bfield = skfmm.travel_time_genes(phi, drivers, speeds, dx=x_width/taps)

ax[0].set_title('tau')
ax[0].contour(X, Y, phi, [1e-4], colors='black', linewidths=(3))
ax[0].contour(X, Y, tau, levels=30)

plt.gca().set_aspect(1)

ax[1].set_title('tau')
ax[1].contour(X, Y, phi, [1e-4], colors='black', linewidths=(3))
ax[1].contour(X, Y, bfield, levels=2)

# Draw Tibor's analytical curve for comparison
beta = np.sqrt(v_spd**2 - 1) # theoretical curve
theta = np.linspace(0, np.pi, 100)
x_spiral = np.exp(theta / beta) * np.cos(theta)
y_spiral = np.exp(theta / beta) * np.sin(theta)

plt.gca().set_aspect(1)
plt.xticks([]); plt.yticks([])

ax[0].plot(x_spiral, y_spiral, 'r--')
ax[0].plot(x_spiral, -y_spiral, 'r--')
ax[0].set_xlim(-0.5 * x_width, +0.5 * x_width)
ax[0].set_ylim(-0.5 * y_width, +0.5 * y_width)
#fig.show()
plt.savefig("testgraph2.png")

