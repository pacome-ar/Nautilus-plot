import plotter, spirals

import numpy as np
from matplotlib import pyplot as plt

def run_example():
    np.random.seed(42)
    tmin, tmax = 0, 12.*np.pi
    spiral = spirals.ArchimedesSpiral() #LituusSpiral()
    pie = np.array([1, 3, 5, 1, 1])#np.random.rand(5)
    pielabels = list(zip(['first', 'second', 'third', 'fourth', 'fifth'], pie))

    t_quads = plotter.prepare_pie(spiral, pie,
                                        tmin=tmin, tmax=tmax, sort=False)
    sections, minmax = plotter.premake_patches(
                                        spiral, t_quads, n_points=1000)

    plotter.plot_pie(spiral, sections, tmax, pielabels, minmax=minmax,
                     cmap='viridis', figsize=(10, 10),
                     edgewidth=3, spiralkwargs={'ls':'', 'c':'w'},
                     axisoff=True)
    plt.legend()
    plt.show()

# run_example()
