# This is the base space class from my old (2025) python reimplementation of FMM
# that never worked properly. Some of the ideas might be reusable but I think
# this class should be rewritted from scratch with clearer language (maybe
# "Domain" insead of "mesh").
# Fri 22 May 13:43:29 BST 2026 -- Chay

###

# The base space (or "mesh") B is an array of points in Euclidean space: each point is a tuple of two
# coordinates (we are in 2d for this prototype)

import numpy

class Mesh:
# TODO the mesh should be a Numpy Array with given extents. say that points = a
# numpy array.
    # we initialise a mesh with two arrays, being the x and y extents, and a
    # mesh step size. The mesh is always square: dx and dy are both equal to
    # delta.
    # xrange and yrange are extents (tuples) and delta is a size:
    def __init__(self, xrange, yrange, delta):
        self.extent = (xrange, yrange) # the real x and y extents
        self.delta = delta # mesh step size
        xticks = int((self.extent[0][1]-self.extent[0][0])/delta)
        yticks = int((self.extent[1][1]-self.extent[1][0])/delta)
        self.points = numpy.zeros((xticks, yticks))
        # points is a numpy array and is accessed with integers (0,0 being the
        # origin of the array)

    def get_closest(self, point):
        # find the closest node in the mesh to a chosen real point and return its
        # array indices as a tuple of integers
        xindex = int((point[0] - self.extent[0][0])/self.delta)
        yindex = int((point[1] - self.extent[1][0])/self.delta)
        xindex = min(xindex, self.points.shape[0] - 1)
        yindex = min(yindex, self.points.shape[1] - 1)
        xindex = max(xindex, 0)
        yindex = max(yindex, 0)
        return (xindex, yindex)

    def real_coords(self, indices):
        x = self.extent[0][0] + self.delta * indices[0]
        y = self.extent[1][0] + self.delta * indices[1]
        return (x, y)

    def contains(self, indices):
        inside = True
        inside = inside and (indices[0] < self.points.shape[0])
        inside = inside and (indices[0] >= 0)
        inside = inside and (indices[1] < self.points.shape[1])
        inside = inside and (indices[1] >= 0)
        return inside

    def get_neighbours(self, indices):
        neighbours = []
        for dx in [-1, +1]:
            neighbour = (indices[0] + dx, indices[1])
            if (self.contains(neighbour)):
                neighbours.append(neighbour)

        for dy in [-1, +1]:
            neighbour = (indices[0], indices[1] + dy)
            if (self.contains(neighbour)):
                neighbours.append(neighbour)

        return neighbours

