import os
from pathlib import Path
from typing import Sequence, Dict, Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde, pearsonr

# Import the functions we need from suite2p
from suite2p.extraction import masks

# Declare matplotlib display settings
import matplotlib as mpl
jet = matplotlib.cm.get_cmap("jet").copy()


def load_s2p_output(output_dir):
    """
    Loads suite2p output for directory of choice
    :param output_dir: path to directory with stat.npy et al.
    :return f: numpy array with fluorescence values for each ROI
    :return f_neu: numpy array with fluorescence values for neuropil mask for each ROI
    :return spks: numpy array with deconvolved fluorescence
    :return stats: numpy array containing dict with stats output
    :return iscell: boolean array of ROIs identified as cells
    :return ops: dict of suite2p run settings
    """
    if not os.path.exists(os.path.join(output_dir, 'stat.npy')):
        raise Exception("stat.npy appears to be missing. Please set output_dir to the output of a suite2p run.")

    ops = np.load(Path(output_dir).joinpath("ops.npy"), allow_pickle=True).item()
    stats = np.load(Path(output_dir).joinpath("stat.npy"), allow_pickle=True)
    iscell = np.load(Path(output_dir).joinpath("iscell.npy"), allow_pickle=True)[:, 0].astype('bool')
    f = np.load(Path(output_dir).joinpath('F.npy'))
    f_neu = np.load(Path(output_dir).joinpath('Fneu.npy'))
    spks = np.load(Path(output_dir).joinpath('spks.npy'))

    return f, f_neu, spks, stats, iscell, ops


def stats_to_array(stats: Sequence[Dict[str, Any]], Ly: int, Lx: int, label_id: bool = False):
    """
    converts stats sequence of dictionaries to an array
    :param stats: sequence of dictionaries from stat.npy
    :param Ly: number of pixels along dim Y from ops dictionary
    :param Lx: number of pixels along dim X
    :param label_id: keeps ROI indexing
    :return: numpy stack of arrays, each containing x and y pixels for each ROI
    """
    arrays = []
    for i, stat in enumerate(stats):
        arr = np.zeros((Ly, Lx), dtype=float)
        arr[stat['ypix'], stat['xpix']] = 1
        if label_id:
            arr *= i + 1
        arrays.append(arr)
    return np.stack(arrays)


def plot_detection_outcome(stats, ops, iscell, fname, output_dir):
    """
    generates a four panel plot with maximum intensity projection, both cell and non-cell ROIs
    detected in recording, all non-cell ROIs and all cell ROIs
    :param stats: stats array from stat.npy
    :param ops: dictionary of suite2p settings
    :param iscell: boolean array of which ROIs are identified as cells
    :param fname: name of recording for writing plots to file
    :param output_dir: path to directory for writing plots to file
    :return: none
    """
    im = stats_to_array(stats, Ly=ops['Ly'], Lx=ops['Lx'], label_id=True)
    im[im == 0] = np.nan

    plt.ioff()
    fig, ax = plt.subplots(nrows=1, ncols=4)
    plt.subplot(1, 4, 1)
    plt.imshow(ops['max_proj'], cmap='gray')
    plt.title("registered image, max projection")

    plt.subplot(1, 4, 2)
    plt.imshow(np.nanmax(im, axis=0), cmap='jet')
    plt.title("all ROIs detected")

    plt.subplot(1, 4, 3)
    plt.imshow(np.nanmax(im[~iscell], axis=0), cmap='jet')
    plt.title("all non-cell ROIs")

    plt.subplot(1, 4, 4)
    plt.imshow(np.nanmax(im[iscell], axis=0), cmap='jet')
    plt.title("all cell ROIs")

    fig_name = Path(output_dir).joinpath("%s_cell-detect-outcomes.svg" % fname)

    fig.savefig(fig_name, format="svg", dpi=1200)
    plt.close(fig)


def make_bounding_box(stat):
    """
    utility function for creating a bounding box around cells and neuropil masks
    :param stat: numpy array from stat.npy
    :returns y_lim1, y_lim2, x_lim1, x_lim2: x and y pixels for adding ~ 40 px border around cell ROI or mask
    """
    y_min = stat['ypix'].min()
    y_max = stat['ypix'].max()

    x_min = stat['xpix'].min()
    x_max = stat['xpix'].max()

    y_pad = round((80 - np.ptp(stat['ypix'])) / 2)
    x_pad = round((80 - np.ptp(stat['xpix'])) / 2)

    y_lim1 = y_min - y_pad
    y_lim2 = y_max + y_pad

    x_lim1 = x_min - x_pad
    x_lim2 = x_max + x_pad

    return y_lim1, y_lim2, x_lim1, x_lim2


def plot_roi_and_neuropil(f, f_neu, spks, ops, stat, which_roi, fname, out_dir):
    """
    utility for generating a multipanel plot by user-selected ROI. plots
    fluoresence trace, neuropil fluo trace, deconvolved spikes, cell ROI and
    neuropil mask. f, f_neu and spks are only plotted for first 2000 frames
    :param f: numpy array of fluorescence values
    :param f_neu: numpy array of neuropil fluorescence values
    :param spks: numpy array of deconvolved spikes
    :param ops: dictionary of suite2p settings
    :param stat: numpy array of stat.npy
    :param which_roi: index of ROI to be plotted
    :param fname: string containing name of recording for writing figure to file
    :param out_dir: path to write .svg files
    :return: none
    """
    im = stats_to_array(stat, Ly=ops['Ly'], Lx=ops['Lx'], label_id=True)
    im[im == 0] = np.nan

    fig = plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(3, 4, wspace=0.1, hspace=0.1, figure=fig)

    f_ax = fig.add_subplot(grid[0:2, 0:2])
    f_ax.plot(range(0, 2000, 1), f[which_roi, range(0, 2000, 1)], 'g', alpha=0.8)
    f_ax.plot(range(0, 2000, 1), f_neu[which_roi, range(0, 2000, 1)], 'm', alpha=0.5)
    f_ax.set_ylabel('fluorescence')

    # Adjust spks range to match range of fluorescence traces
    fmax = np.maximum(f.max(), f_neu.max())
    fmin = np.minimum(f.min(), f_neu.min())
    frange = fmax - fmin
    sp = spks[which_roi, ]
    sp /= sp.max()
    sp *= frange
    sp = sp[range(0, 2000, 1)]

    spks_ax = fig.add_subplot(grid[2, 0:2])
    spks_ax.plot(range(0, 2000, 1), sp, 'k')
    spks_ax.set_xlabel('frame')
    spks_ax.set_ylabel('deconvolved')

    # Calculate bounding box for visualising ROI overlaid on meanImgE (consider writing as its own function)
    s = stat[which_roi]
    y_lim1, y_lim2, x_lim1, x_lim2 = make_bounding_box(s)

    img_ax = fig.add_subplot(grid[:, 2])
    img_ax.imshow(ops['meanImgE'][y_lim1:y_lim2, x_lim1:x_lim2], cmap='gray')
    img_ax.imshow(im[which_roi, y_lim1:y_lim2, x_lim1:x_lim2], alpha=0.5, cmap='spring')
    img_ax.title.set_text('ROI index %s' % which_roi)

    # Get pixels for each neuropil
    _, neu_ipix = masks.create_masks(ops=ops, stats=stat)

    # Convert neuropil pixel indices into mask
    neu_mask = np.zeros(ops['Ly']*ops['Lx'])
    neu_mask[neu_ipix[which_roi]] = 1

    # Check that pixels are being mapped back to 2D array correctly
    neu_mask = np.reshape(neu_mask, (ops['Ly'], ops['Lx']))
    neu_mask[neu_mask == 0] = np.nan

    mask_ax = fig.add_subplot(grid[:, 3])
    mask_ax.imshow(ops['meanImgE'][y_lim1:y_lim2, x_lim1:x_lim2], cmap='gray')
    mask_ax.imshow(neu_mask[y_lim1:y_lim2, x_lim1:x_lim2], cmap='spring_r', alpha=0.5)
    mask_ax.title.set_text('neuropil mask, no. of pixels = %s' % len(neu_ipix[which_roi]))

    # Set up output directory and filenames to write, create output directory if
    # it does not yet exist
    if not os.path.isdir(out_dir):
        Path(out_dir).mkdir(parents=False, exist_ok=False)

    basename = fname + "_f-cell-neuropil_roi%s.svg" % which_roi
    fig_name = Path(out_dir).joinpath(basename)

    fig.savefig(fig_name, format="svg", dpi=1200)
    plt.close(fig)


def plot_f_f_neu(f, f_neu, which_roi, fname, out_dir):
    """
    a function that plots f vs f_neu for user-selected ROI, displays as a colour map of the KDE
    :param f: numpy array of cell fluorescence values
    :param f_neu: numpy array of neuropil fluorescence values
    :param which_roi: index of ROI
    :param fname: str, name of recording for writing figure to file
    :param out_dir: str, path to directory for writing .svg files
    :return: none
    """
    x = f[which_roi, :]
    y = f_neu[which_roi, :]
    # calculate Pearson's R from cell and neuropil fluorescence
    corr, _ = pearsonr(x, y)

    # create a gaussian KDE on a regular grid of 256 x 256 bins
    nbins = 256
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # make a figure of cell and neuropil fluorescence values, density
    # of points
    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca()

    ax.tick_params(axis='y', left=True, which='major', labelleft=True)
    ax.contourf(xi, yi, zi.reshape(xi.shape), cmap='Greens')
    cset = ax.contour(xi, yi, zi.reshape(xi.shape), colors='k')

    ax.clabel(cset, inline=1, fontsize=8)
    ax.set_xlabel('cell fluorescence')
    ax.set_ylabel('neuropil fluorescence')
    ax.set_title('ROI index %s' % which_roi)
    ax.text(0.1, 0.9, 'Pearson\'s R = %3f' % corr,
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes)

    basename = fname + "_f-cell-neuropil-corr_roi%s.svg" % which_roi
    fig_name = Path(out_dir).joinpath(basename)

    fig.savefig(fig_name, format="svg", dpi=1200)
    plt.close(fig)


def plot_reg_metrics(ops, fname, out_dir):
    """
    plots registration metrics for s2p run, writes to .svg file
    :param ops: dictionary of options values from ops.npy
    :param fname: str, name of recording for writing figure to file
    :param out_dir: str, path to directory for writing .svg files
    :return: none
    """
    fig = plt.figure(figsize=(8, 6))
    grid = plt.GridSpec(3, 4, wspace=0.5, hspace=0.8, figure=fig)

    ax1 = fig.add_subplot(grid[0, 0:2])
    ax1.plot(ops['tPC'][:, 0], 'k')
    ax1.set_xlabel('frames')
    ax1.set_ylabel('PC 1')

    ax2 = fig.add_subplot(grid[0, 2:4])
    ax2.plot(ops['regDX'][:, 1], 'o-')
    ax2.plot(ops['regDX'][:, 2], 'go-')
    ax2.set_xlabel('PC')
    ax2.set_ylabel('NR offset')

    ax3 = fig.add_subplot(grid[1:3, 0:2])
    ax3.imshow(ops['regPC'][0, 0, :, :])
    ax3.set_title('mean top 500 frames of PC1')
    ax3.tick_params(axis='y', left=False, which='major', labelleft=False)
    ax3.tick_params(axis='x', bottom=False, which='major', labelbottom=False)

    ax4 = fig.add_subplot(grid[1:3, 2:4])
    ax4.imshow(ops['regPC'][1, 0, :, :])
    ax4.set_title('mean bottom 500 frames of PC1')
    ax4.tick_params(axis='y', left=False, which='major', labelleft=False)
    ax4.tick_params(axis='x', bottom=False, which='major', labelbottom=False)

    basename = fname + "_reg-metrics.svg"
    fig_name = Path(out_dir).joinpath(basename)

    fig.savefig(fig_name, format="svg", dpi=1200)
    plt.close(fig)
