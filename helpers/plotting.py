import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.decomposition import PCA

def plot_RT_histogram(rts, DI=None, bins=None, ax=None, cmap=None, lw=1, legend=None):
    """
    rts:    reaction times dictionary. keys are epochs, values are list of RTs
    DI:     dict with each target's DI. Vals get added to legend
    bins:   either int or range to specify bins for the histogram. If int, will use 
                that many bins between 0 and 2 sec
    ax:     default is None. If not None, make the plot on the specified axis
    cmap:   mpl iterable colormap generator (see tin_helpers.make_tbp_colormaps)
    """
    if bins is None:
        bins = np.arange(0, 2, 0.1)
    
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    skeys = rts.keys()

    # Now, for each soundID (reference, target1, target2 etc.),
    # create histogram
    for i, k in enumerate(skeys):
        counts, xvals = np.histogram(rts[k], bins=bins)
        if DI is not None:
            try:
                _di = round(DI[k], 2)
            except:
                _di = 'N/A'
        else:
            _di = 'N/A'
        n = len(rts[k])
        if cmap is not None:
            color = cmap(i)
        else:
            color = None
        if legend is not None:
            leg = legend[i]
        else:
            leg = f'{k}, DI: {_di}, n: {n}'
        ax.step(xvals[:-1], np.cumsum(counts) / len(rts[k]), 
                    label=leg, lw=lw, color=color)
    
    ax.legend(frameon=False)
    ax.set_xlabel('Reaction time (s)')
    ax.set_ylabel('Cummulative Probability')

    return ax



# DB free version of @Mateo code for plotting penetration map
def penetration_map(sites, areas, best_frequencies, coordinates, equal_aspect=False, flip_X=False, flatten=False, flip_YZ=False):
    """
    Plots a 3d map of the list of specified sites, displaying the best frequency as color, and the brain region as
    maker type (NA: circle, A1: triangle, PEG: square).
    The site location, brain area and best frequency are extracted from celldb, specifically from the penetration (for
    coordinates and rotations) and the penetration (for area and best frequency) sites. If no coordinates are found the
    site is ignored. no BF is displayed as an empty marker.
    The values in the plot increase in direction Posterior -> Anterior, Medial -> Lateral and Ventral -> Dorsal. The
    antero posterior (X) axis can be flipped with the according parameter.
    :param sites: list of str specifying sites, with the format "ABC001a"
    :param equal_aspect: boolean. whether to constrain the data to a cubic/square space i.e. equal dimensions in XYZ/XY
    :flip_X: Boolean. Flips the direction labels for the antero-posterior (X) axis. The default is A > P .
    Y. Lateral > Medial, Z. Dorsal > Ventral.
    :flatten: Boolean. PCA 2d projection. Work in progress.
    :flip_YZ: Boolean. Flips the direction and labels of the YZ principal component when flattening.
    :landmarks: dict of vectors, where the key specifies the landmark name, and the vector has the values
    [x0, y0, z0, x, y, z, tilt, rot]. If the landmark name is 'OccCrest' or 'MidLine' uses the AP and ML values as zeros
    respectively.
    :return: matplotlib figure
    """
    area_marker = {'NA': 'o', 'A1': '^', 'PEG': 's'}
    good_sites = sites

    coordinates = np.stack(coordinates, axis=1)
    best_frequencies = np.asarray(best_frequencies)
    areas = np.asarray(areas)
    good_sites = np.asarray(good_sites)

    # centers data and transforms cm to mm
    center = np.mean(coordinates, axis=1)


    coordinates = coordinates - center[:, None]
    coordinates = coordinates * 10

    # defines BF colormap range if valid best frequencies are available.
    vmax = best_frequencies.max() if best_frequencies.max() > 0 else 32000
    vmin = best_frequencies[best_frequencies != 0].min() if best_frequencies.min() > 0 else 100

    if flatten is False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if equal_aspect:
            X, Y, Z = coordinates
            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')


        for area in set(areas):
            coord_subset = coordinates[:, areas == area]
            BF_subset = best_frequencies[areas == area]
            site_subset = good_sites[areas == area]

            X, Y, Z = coord_subset
            p = ax.scatter(X, Y, Z, s=100, marker=area_marker[area], edgecolor='black',
                           c=BF_subset, cmap='inferno',
                           norm=colors.LogNorm(vmin=vmin, vmax=vmax))

            for coord, site in zip(coord_subset.T, site_subset):
                x, y, z = coord
                ax.text(x, y, z, site[3:6])

        # formats axis
        ax.set_xlabel('anterior posterior (mm)')
        ax.set_ylabel('Medial Lateral (mm)')
        ax.set_zlabel('Dorsal ventral (mm)')

        fig.canvas.draw()
        x_tick_loc = ax.get_xticks().tolist()
        xlab = [f'{x:.1f}' for x in x_tick_loc]

        if flip_X:
            xlab[0] = 'A'
            xlab[-1] = 'P'
        else:
            xlab[0] = 'P'
            xlab[-1] = 'A'

        _ = ax.xaxis.set_major_locator(mticker.FixedLocator(x_tick_loc))
        _ = ax.set_xticklabels(xlab)

        y_tick_loc = ax.get_yticks().tolist()
        ylab = [f'{x:.1f}' for x in y_tick_loc]
        ylab[0] = 'M'
        ylab[-1] = 'L'
        _ = ax.yaxis.set_major_locator(mticker.FixedLocator(y_tick_loc))
        _ = ax.set_yticklabels(ylab)

        z_tick_loc = ax.get_zticks().tolist()
        zlab = [f'{x:.1f}' for x in z_tick_loc]
        zlab[0] = 'V'
        zlab[-1] = 'D'
        _ = ax.zaxis.set_major_locator(mticker.FixedLocator(z_tick_loc))
        _ = ax.set_zticklabels(zlab)

    elif flatten is True:
        # flattens doing a PCA over the Y and Z dimensions, i.e. medio-lateral and dorso-ventral.
        # this keeps the anteroposterior orientations to help locate the flattened projection
        pc1 = PCA().fit_transform(coordinates[1:,:].T)[:,0]
        
        flat_coords = np.stack((coordinates[0,:], pc1), axis=0)

        # FLips data on axes since 2d plots cannot be rotated interactively
        fx = -1 if flip_X is True else 1
        fyz = -1 if flip_YZ is True else 1
        flat_coords = flat_coords * np.array([[fx],[fyz]])


        fig, ax = plt.subplots()

        for area in set(areas):
            flat_coords_subset = flat_coords[:, areas == area]
            BF_subset = best_frequencies[areas == area]
            site_subset = good_sites[areas == area]

            X, Y = flat_coords_subset
            p = ax.scatter(X, Y, s=100, marker=area_marker[area], edgecolor='black',
                           c=BF_subset, cmap='inferno',
                           norm=colors.LogNorm(vmin=vmin, vmax=vmax))

            for coord, site in zip(flat_coords_subset.T, site_subset):
                x, y = coord
                ax.text(x, y, site[3:6])

        # formats axis
        if equal_aspect:
            ax.axis('equal')
        ax.set_xlabel('anterior posterior (mm)')
        ax.set_ylabel('1PC_YZ (mm)')



    cbar = fig.colorbar(p)
    cbar.ax.set_ylabel('BF (Hz)', rotation=-90, va="top")

    return fig, coordinates


def compute_ellipse(x, y):
    inds = np.isfinite(x) & np.isfinite(y)
    x= x[inds]
    y = y[inds]
    data = np.vstack((x, y))
    mu = np.mean(data, 1)
    data = data.T - mu

    D, V = np.linalg.eig(np.divide(np.matmul(data.T, data), data.shape[0] - 1))
    # order = np.argsort(D)[::-1]
    # D = D[order]
    # V = abs(V[:, order])

    t = np.linspace(0, 2 * np.pi, 100)
    e = np.vstack((np.sin(t), np.cos(t)))  # unit circle
    VV = np.multiply(V, np.sqrt(D))  # scale eigenvectors
    e = np.matmul(VV, e).T + mu  # project circle back to orig space
    e = e.T

    return e