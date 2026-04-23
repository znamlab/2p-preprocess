import os
from pathlib import Path
from typing import Sequence, Dict, Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde, pearsonr


# Declare matplotlib display settings
import matplotlib as mpl

jet = matplotlib.cm.get_cmap("jet").copy()


def load_s2p_output(output_dir):
    """
    Load Suite2p output for a specific plane directory.

    Args:
        output_dir (str or Path): Path to the directory containing 'stat.npy', 'ops.npy', etc.

    Returns:
        tuple: (f, f_neu, spks, stats, iscell, ops)
            - f (np.ndarray): Fluorescence values for each ROI.
            - f_neu (np.ndarray): Neuropil fluorescence values for each ROI.
            - spks (np.ndarray): Deconvolved fluorescence (spikes).
            - stats (np.ndarray): List of dictionaries containing ROI statistics.
            - iscell (np.ndarray): Boolean array identifying ROIs as cells.
            - ops (dict): Dictionary of Suite2p run settings.
    """
    if not os.path.exists(os.path.join(output_dir, "stat.npy")):
        raise Exception(
            "stat.npy appears to be missing. Please set output_dir to the output of a suite2p run."
        )

    ops = np.load(Path(output_dir).joinpath("ops.npy"), allow_pickle=True).item()
    stats = np.load(Path(output_dir).joinpath("stat.npy"), allow_pickle=True)
    iscell = np.load(Path(output_dir).joinpath("iscell.npy"), allow_pickle=True)[
        :, 0
    ].astype("bool")
    f = np.load(Path(output_dir).joinpath("F.npy"))
    f_neu = np.load(Path(output_dir).joinpath("Fneu.npy"))
    spks = np.load(Path(output_dir).joinpath("spks.npy"))

    return f, f_neu, spks, stats, iscell, ops


def stats_to_array(
    stats: Sequence[Dict[str, Any]], Ly: int, Lx: int, label_id: bool = False
):
    """
    Convert Suite2p ROI stats to a 3D numpy array of masks.

    Args:
        stats (Sequence[Dict[str, Any]]): Sequence of dictionaries from stat.npy.
        Ly (int): Number of pixels along the Y dimension.
        Lx (int): Number of pixels along the X dimension.
        label_id (bool, optional): If True, pixels are labeled with the ROI index (1-indexed).
            If False, pixels are set to 1. Default False.

    Returns:
        np.ndarray: Stack of ROI masks (n_rois x Ly x Lx).
    """
    arrays = []
    for i, stat in enumerate(stats):
        arr = np.zeros((Ly, Lx), dtype=float)
        arr[stat["ypix"], stat["xpix"]] = 1
        if label_id:
            arr *= i + 1
        arrays.append(arr)
    return np.stack(arrays)


def plot_detection_outcome(stats, ops, iscell, fname=None, output_dir=None):
    """
    Generate a four-panel plot showing ROI detection outcomes.

    The panels include: Max Intensity Projection, All ROIs, Non-cell ROIs, and Cell ROIs.

    Args:
        stats (np.ndarray): Stats array from stat.npy.
        ops (dict): Dictionary of Suite2p settings.
        iscell (np.ndarray): Boolean array identifying which ROIs are cells.
        fname (str, optional): Name of the recording for the output filename.
        output_dir (str or Path, optional): Directory to save the plot.

    Returns:
        matplotlib.figure.Figure: The generated figure handle.
    """
    im = stats_to_array(stats, Ly=ops["Ly"], Lx=ops["Lx"], label_id=True)
    im[im == 0] = np.nan

    plt.ioff()
    fig, ax = plt.subplots(nrows=1, ncols=4)
    plt.subplot(1, 4, 1)
    plt.imshow(ops["max_proj"], cmap="gray")
    plt.title("registered image, max projection")

    plt.subplot(1, 4, 2)
    plt.imshow(np.nanmax(im, axis=0), cmap="jet")
    plt.title("all ROIs detected")

    plt.subplot(1, 4, 3)
    plt.imshow(np.nanmax(im[~iscell], axis=0), cmap="jet")
    plt.title("all non-cell ROIs")

    plt.subplot(1, 4, 4)
    plt.imshow(np.nanmax(im[iscell], axis=0), cmap="jet")
    plt.title("all cell ROIs")
    if output_dir is not None:
        assert fname is not None, "fname must be provided if output_dir is provided"
        fig_name = Path(output_dir).joinpath("%s_cell-detect-outcomes.svg" % fname)
        fig.savefig(fig_name, format="svg", dpi=1200)
    return fig


def make_bounding_box(stat):
    """
    Utility for creating a bounding box around a single ROI.

    Args:
        stat (dict): Single ROI dictionary from stat.npy.

    Returns:
        tuple: (y_lim1, y_lim2, x_lim1, x_lim2) coordinates for an ~80px box around the ROI.
    """
    y_min = stat["ypix"].min()
    y_max = stat["ypix"].max()

    x_min = stat["xpix"].min()
    x_max = stat["xpix"].max()

    y_pad = round((80 - np.ptp(stat["ypix"])) / 2)
    x_pad = round((80 - np.ptp(stat["xpix"])) / 2)

    y_lim1 = y_min - y_pad
    y_lim2 = y_max + y_pad

    x_lim1 = x_min - x_pad
    x_lim2 = x_max + x_pad

    return y_lim1, y_lim2, x_lim1, x_lim2


def plot_roi_and_neuropil(f, f_neu, spks, ops, stat, which_roi, fname, out_dir):
    """
    Generate a multipanel plot for a user-selected ROI.

    The plot includes: raw fluorescence, neuropil fluorescence, deconvolved spikes,
    cell ROI, and neuropil mask.

    Args:
        f (np.ndarray): Fluorescence traces.
        f_neu (np.ndarray): Neuropil fluorescence traces.
        spks (np.ndarray): Deconvolved spikes.
        ops (dict): Suite2p run settings.
        stat (np.ndarray): Stats array from stat.npy.
        which_roi (int): Index of the ROI to plot.
        fname (str): Name of the recording for the output filename.
        out_dir (str or Path): Directory to save the plot (.svg).
    """
    # Import the functions we need from suite2p
    try:
        from suite2p.extraction import masks
    except ImportError:
        raise ImportError(
            "suite2p is not installed. Please see 2p-preprocess ReadMe to install it"
        )

    im = stats_to_array(stat, Ly=ops["Ly"], Lx=ops["Lx"], label_id=True)
    im[im == 0] = np.nan

    fig = plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(3, 4, wspace=0.1, hspace=0.1, figure=fig)

    f_ax = fig.add_subplot(grid[0:2, 0:2])
    f_ax.plot(range(0, 2000, 1), f[which_roi, range(0, 2000, 1)], "g", alpha=0.8)
    f_ax.plot(range(0, 2000, 1), f_neu[which_roi, range(0, 2000, 1)], "m", alpha=0.5)
    f_ax.set_ylabel("fluorescence")

    # Adjust spks range to match range of fluorescence traces
    fmax = np.maximum(f.max(), f_neu.max())
    fmin = np.minimum(f.min(), f_neu.min())
    frange = fmax - fmin
    sp = spks[which_roi,]
    sp /= sp.max()
    sp *= frange
    sp = sp[range(0, 2000, 1)]

    spks_ax = fig.add_subplot(grid[2, 0:2])
    spks_ax.plot(range(0, 2000, 1), sp, "k")
    spks_ax.set_xlabel("frame")
    spks_ax.set_ylabel("deconvolved")

    # Calculate bounding box for visualising ROI overlaid on meanImgE (consider writing as its own function)
    s = stat[which_roi]
    y_lim1, y_lim2, x_lim1, x_lim2 = make_bounding_box(s)

    img_ax = fig.add_subplot(grid[:, 2])
    img_ax.imshow(ops["meanImgE"][y_lim1:y_lim2, x_lim1:x_lim2], cmap="gray")
    img_ax.imshow(im[which_roi, y_lim1:y_lim2, x_lim1:x_lim2], alpha=0.5, cmap="spring")
    img_ax.title.set_text("ROI index %s" % which_roi)

    # Get pixels for each neuropil
    _, neu_ipix = masks.create_masks(stats=stat, Lx=ops["Lx"], Ly=ops["Ly"], ops=ops)

    # Convert neuropil pixel indices into mask
    neu_mask = np.zeros(ops["Ly"] * ops["Lx"])
    neu_mask[neu_ipix[which_roi]] = 1

    # Check that pixels are being mapped back to 2D array correctly
    neu_mask = np.reshape(neu_mask, (ops["Ly"], ops["Lx"]))
    neu_mask[neu_mask == 0] = np.nan

    mask_ax = fig.add_subplot(grid[:, 3])
    mask_ax.imshow(ops["meanImgE"][y_lim1:y_lim2, x_lim1:x_lim2], cmap="gray")
    mask_ax.imshow(neu_mask[y_lim1:y_lim2, x_lim1:x_lim2], cmap="spring_r", alpha=0.5)
    mask_ax.title.set_text(
        "neuropil mask, no. of pixels = %s" % len(neu_ipix[which_roi])
    )

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
    Plot F vs F_neu correlation for a selected ROI using a KDE colormap.

    Args:
        f (np.ndarray): Cell fluorescence traces.
        f_neu (np.ndarray): Neuropil fluorescence traces.
        which_roi (int): Index of the ROI.
        fname (str): Name of the recording for the output filename.
        out_dir (str or Path): Directory to save the plot (.svg).
    """
    x = f[which_roi, :]
    y = f_neu[which_roi, :]
    # calculate Pearson's R from cell and neuropil fluorescence
    corr, _ = pearsonr(x, y)

    # create a gaussian KDE on a regular grid of 256 x 256 bins
    nbins = 256
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # make a figure of cell and neuropil fluorescence values, density
    # of points
    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca()

    ax.tick_params(axis="y", left=True, which="major", labelleft=True)
    ax.contourf(xi, yi, zi.reshape(xi.shape), cmap="Greens")
    cset = ax.contour(xi, yi, zi.reshape(xi.shape), colors="k")

    ax.clabel(cset, inline=1, fontsize=8)
    ax.set_xlabel("cell fluorescence")
    ax.set_ylabel("neuropil fluorescence")
    ax.set_title("ROI index %s" % which_roi)
    ax.text(
        0.1,
        0.9,
        "Pearson's R = %3f" % corr,
        verticalalignment="bottom",
        horizontalalignment="left",
        transform=ax.transAxes,
    )

    basename = fname + "_f-cell-neuropil-corr_roi%s.svg" % which_roi
    fig_name = Path(out_dir).joinpath(basename)

    fig.savefig(fig_name, format="svg", dpi=1200)
    plt.close(fig)


def plot_reg_metrics(ops, fname, out_dir):
    """
    Plot registration metrics (PCs and offsets) for a Suite2p run.

    Args:
        ops (dict): Dictionary of Suite2p run settings and metrics.
        fname (str): Name of the recording for the output filename.
        out_dir (str or Path): Directory to save the plot (.svg).
    """
    fig = plt.figure(figsize=(8, 6))
    grid = plt.GridSpec(3, 4, wspace=0.5, hspace=0.8, figure=fig)

    ax1 = fig.add_subplot(grid[0, 0:2])
    ax1.plot(ops["tPC"][:, 0], "k")
    ax1.set_xlabel("frames")
    ax1.set_ylabel("PC 1")

    ax2 = fig.add_subplot(grid[0, 2:4])
    ax2.plot(ops["regDX"][:, 1], "o-")
    ax2.plot(ops["regDX"][:, 2], "go-")
    ax2.set_xlabel("PC")
    ax2.set_ylabel("NR offset")

    ax3 = fig.add_subplot(grid[1:3, 0:2])
    ax3.imshow(ops["regPC"][0, 0, :, :])
    ax3.set_title("mean top 500 frames of PC1")
    ax3.tick_params(axis="y", left=False, which="major", labelleft=False)
    ax3.tick_params(axis="x", bottom=False, which="major", labelbottom=False)

    ax4 = fig.add_subplot(grid[1:3, 2:4])
    ax4.imshow(ops["regPC"][1, 0, :, :])
    ax4.set_title("mean bottom 500 frames of PC1")
    ax4.tick_params(axis="y", left=False, which="major", labelleft=False)
    ax4.tick_params(axis="x", bottom=False, which="major", labelbottom=False)

    basename = fname + "_reg-metrics.svg"
    fig_name = Path(out_dir).joinpath(basename)

    fig.savefig(fig_name, format="svg", dpi=1200)
    plt.close(fig)
