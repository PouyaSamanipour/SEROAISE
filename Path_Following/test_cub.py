


import numpy as np

def generate_hypercube_vertices(bounds):
    dimensions = len(bounds)
    if dimensions == 0:
        return [[]]

    def helper(dim):
        if dim == dimensions:
            return [[]]

        vertices = []
        for vertex in helper(dim + 1):
            for value in bounds[dim]:
                vertices.append([value] + vertex)

        return vertices

    return np.array([helper(0)])


bounds = [(-1, 1), (-5, 5), (-10, 10)]
vertices = generate_hypercube_vertices(bounds)
print(vertices)