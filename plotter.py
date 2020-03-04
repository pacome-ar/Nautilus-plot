import spirals
import plot_complex_polygons
import numpy as np
import matplotlib.pyplot as plt

### Generic transformation function
def to_cartesian(rho, theta):
    return rho*np.cos(theta), rho*np.sin(theta)

def to_polar(x, y):
    return np.sqrt(x**2 + y**2), np.arctan(y/x)

### Preprocessing
def prepare_pie(spiral, pie, tmin=0, tmax=2.*np.pi, sort=False):
    '''WARNING to be verified, using interpolation inside'''

    temp_pie = np.asarray(pie)
    if sort:
        temp_pie.sort()

    temp_pie = temp_pie / temp_pie.sum()
    total_area = spiral.get_area(tmin, tmax)
    target_area = total_area * temp_pie

    # get primitives
    primitive, invprimitive = spiral.make_inverse_interpolant(
                                        tmin, tmax, interpoints=100)

    t0 = tmin
    target_ts = [t0]
    for area in target_area[:-1]:
        t1 = spirals.get_theta(area, t0, primitive, invprimitive)
        target_ts.append(t1)
        t0 = t1

    t_pairs = np.array(list(zip(target_ts, target_ts[1:] + [tmax])))

    temp = t_pairs - (2.*np.pi)
    temp = (1 - (temp <= 0)) * temp

    t_quads = np.concatenate((temp, t_pairs), axis=1)

    return t_quads


### Prepare Plotting
def build_arc(spiral, tA, tB, n_points=100):
    '''returns the carthesian coordinates
    of n_points between A and B following spiral'''
    ts = np.linspace(tA, tB, n_points)
    arc = np.array(to_cartesian(spiral.radius(ts), ts))
    return arc

def build_line(spiral, tA, tB):
    ts = np.array([tA, tB])
    line = np.array(to_cartesian(spiral.radius(ts), ts))
    return line

def get_minmax(spiral, tmax):
    x, y = to_cartesian(spiral.radius(tmax), tmax)
    x = max(abs(x), abs(y))
    return sorted([x, -x])

def premake_patches(spiral, t_quads, n_points=100):
    '''converts the list of angles an radii to list of points
    two cases WARNING: breaks for circles
    # Make vectorial !'''
    sections = []

    for ts in t_quads:
        # each contains coordinates with increasing t: A, B, C, D
        tA, tB, tC, tD = ts
        if ts[1] <= ts[2]:
            # case no hole
            # build arcBA, lineAC, arcCD, lineDA
            poly = np.concatenate(
                (build_arc(spiral, tB, tA, n_points),
                 build_line(spiral, tA, tC),
                 build_arc(spiral, tC, tD, n_points),
                 build_line(spiral, tD, tB)), axis=1)
            sections.append([poly])
        else:
            # case hole
            # build ext arcBD, lineDB
            # build hole arcAC, lineCA
            exterior = np.concatenate(
                (build_arc(spiral, tB, tD, n_points),
                 build_line(spiral, tD, tB)), axis=1)
            interior = np.concatenate(
                (build_arc(spiral, tC, tA, n_points),
                 build_line(spiral, tA, tC)), axis=1)
            sections.append([exterior, interior])
    tmax = t_quads[-1][-1]

    return sections, [get_minmax(spiral, tmax), get_minmax(spiral, tmax)]

### Plotting pie
def plot_spiral(spiral, tmax, ax=None, nbpoints=500, **spiralkwargs):
    thetas = np.linspace(0, tmax, nbpoints)
    if ax:
        ax.plot(*to_cartesian(spiral.radius(thetas), thetas), **spiralkwargs)
    else:
        plt.plot(*to_cartesian(spiral.radius(thetas), thetas), **spiralkwargs)

def plot_pie(spiral, sections, tmax, labels,
                     minmax=[[-1, 1], [-1, 1]],
                     ax=None, colors=None, cmap=None, figsize=None,
                     edgecolor='w', edgewidth=3, axisoff=True,
                     patchkwargs={}, spiralkwargs={'ls':''}, nbpoints=500):

    if colors is None:
        if cmap is None:
            cmap = plt.get_cmap('viridis')
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        colors = cmap(np.linspace(0, 1, len(sections)))

    elif isinstance(colors, (list, tuple, np.ndarray)):
        colors = colors

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        if axisoff:
            ax.axis('off')

    for i, s in enumerate(sections):
        patch = plot_complex_polygons.patchify(s)
        patch.set_edgecolor(edgecolor)
        patch.set_linewidth(edgewidth)
        patch.set_facecolor(colors[i])
        patch.__dict__.update(**patchkwargs)
        patch.set_label(labels[i])
        ax.add_patch(patch)

    ax.set_aspect('equal')
    plt.xlim(*minmax[0])
    plt.ylim(*minmax[1])

    plot_spiral(spiral, tmax, ax=ax, nbpoints=nbpoints, **spiralkwargs)

### Wrappers for ui
def parse_pie(pie, labels=None, name=None):

    try:
        import pandas
        if isinstance(pie, pandas.Series):
            pieval = pie.values
            pielab = pie.index.values
            piename = pie.name
            if piename is None:
                piename = name
    except (NameError, ImportError):
        raise TypeError('wrong input type')

    if isinstance(pie, (list, set, np.ndarray, tuple)):
        pieval = pie
        pielab = labels
        piename = name

    if pielab is None:
        pielab = range(len(pie))
    if piename is None:
        piename = "Nautilus plot"

    return pieval, pielab, piename


def plot(pie, labels=None, name=None, spiral=spirals.ArchimedesSpiral(),
         tmin=0, tmax=4*np.pi, sort=False, n_points=1000,
         cmap='viridis', figsize=(10, 10),
         edgewidth=3, spiralkwargs={'ls':'', 'c':'w'},
         axisoff=True):

    pieval, pielab, piename = parse_pie(pie, labels, name)


    t_quads = prepare_pie(spiral, pieval, tmin, tmax, sort)
    sections, minmax = premake_patches(spiral, t_quads, n_points)
    plot_pie(spiral, sections, tmax, pielab, minmax=minmax,
                     cmap=cmap, figsize=figsize,
                     edgewidth=edgewidth,
                     spiralkwargs=spiralkwargs,
                     axisoff=axisoff)
    plt.title(piename)
    plt.legend()
