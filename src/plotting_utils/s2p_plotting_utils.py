from pathlib import Path
from typing import Sequence, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

import suite2p

# Declare matplotlib display settings
import matplotlib as mpl
mpl.rcParams.update({
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.subplot.wspace': .01,
    'figure.subplot.hspace': .01,
    'figure.figsize': (18, 13),
    'ytick.major.left': False,
})
jet = mpl.cm.get_cmap("jet").copy()
jet.set_bad(color='k')

def load_s2p_output(output_dir):
    ops = np.load(Path(output_dir).joinpath("ops.npy"), allow_pickle=True).item()
    stats = np.load(Path(output_dir).joinpath("stat.npy"), allow_pickle=True)
    iscell = np.load(Path(output_dir).joinpath("iscell.npy"), allow_pickle=True)[:, 0].astype('bool')
    f = np.load(Path(output_dir).joinpath('F.npy'))
    f_neu = np.load(Path(output_dir).joinpath('Fneu.npy'))
    spks = np.load(Path(output_dir).joinpath('spks.npy'))
    return f, f_neu, spks, stats, iscell, ops

def stats_to_array(stats: Sequence[Dict[str, Any]], Ly: int, Lx: int, label_id: bool = False):
    arrays = []
    for i, stat in enumerate(stats):
        arr = np.zeros((Ly, Lx), dtype=float)
        arr[stat['ypix'], stat['xpix']] = 1
        if label_id:
            arr *= i + 1
        arrays.append(arr)
    return(np.stack(arrays))

def plot_detection_outcome(stats, ops, iscell, fname, output_dir):
    im = stats_to_array(stats, Ly = ops['Ly'], Lx = ops['Lx'], label_id=True)
    im[im == 0] = np.nan

    plt.ioff()
    fig, ax = plt.subplots(nrows=1, ncols=4)
    plt.subplot(1, 4, 1)
    plt.imshow(ops['max_proj'], cmap='gray')
    plt.title("registered image, max projection");

    plt.subplot(1, 4, 2)
    plt.imshow(np.nanmax(im, axis=0), cmap='jet')
    plt.title("all ROIs detected");

    plt.subplot(1, 4, 3)
    plt.imshow(np.nanmax(im[~iscell], axis=0), cmap='jet')
    plt.title("all non-cell ROIs");

    plt.subplot(1, 4, 4)
    plt.imshow(np.nanmax(im[iscell], axis=0), cmap='jet')
    plt.title("all cell ROIs");

    fig_name = Path(output_dir).joinpath("%s_cell-detect-outcomes.svg"%fname)

    fig.savefig(fig_name, format="svg", dpi=1200)
    plt.close(fig)

def make_bounding_box(stat):
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
    im = stats_to_array(stats, Ly=ops['Ly'], Lx=ops['Lx'], label_id=True)
    im[im == 0] = np.nan

    fig = plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(3, 4, wspace=0.1, hspace=0.1, figure=fig)

    f_ax = fig.add_subplot(grid[0:2, 0:2])
    f_ax.plot(range(0, 2000, 1), f[which_roi, range(0, 2000, 1)], 'g', alpha=0.8)
    f_ax.plot(range(0,2000,1), f_neu[which_roi, range(0,2000,1)], 'm', alpha=0.5)
    f_ax.set_ylabel('fluorescence')

    # Adjust spks range to match range of fluorescence traces
    fmax = np.maximum(f.max(), f_neu.max())
    fmin = np.minimum(f.min(), f_neu.min())
    frange = fmax - fmin
    sp = spks[which_roi,]
    sp /= sp.max()
    sp *= frange
    sp = sp[range(0,2000,1)]

    spks_ax = fig.add_subplot(grid[2, 0:2])
    spks_ax.plot(range(0,2000,1), sp, 'k')
    spks_ax.set_xlabel('frame')
    spks_ax.set_ylabel('deconvolved')

    # Calculate bounding box for visualising ROI overlaid on meanImgE (consider writing as its own function)
    s = stat[which_roi]
    y_lim1, y_lim2, x_lim1, x_lim2 = make_bounding_box(s)

    img_ax = fig.add_subplot(grid[:, 2])
    img_ax.imshow(ops['meanImgE'][y_lim1:y_lim2, x_lim1:x_lim2], cmap='gray')
    img_ax.imshow(im[which_roi, y_lim1:y_lim2, x_lim1:x_lim2], alpha=0.5, cmap='spring')
    img_ax.title.set_text('ROI index %s'%(which_roi))

    cell_pix=np.zeros((ops['Ly'], ops['Lx']))
    lammap=np.zeros((ops['Ly'], ops['Lx']))
    ypix=s['ypix']
    xpix=s['xpix']
    lam=s['lam']
    lammap[ypix, xpix] = np.maximum(lammap[ypix, xpix], lam)
    cell_pix = lammap > 0.0

    mask=suite2p.extraction.create_neuropil_masks(
        ypixs=s['ypix'],
        xpixs=s['xpix'],
        cell_pix=cell_pix,
        inner_neuropil_radius=ops['inner_neuropil_radius'],
        min_neuropil_pixels=ops['min_neuropil_pixels'],
        circular=ops.get('circular_neuropil', False)
    )

    neu_mask = np.zeros(512*512)
    neu_mask[mask[0]] = 1
    neu_mask = np.reshape(neu_mask, (512,512))
    neu_mask[neu_mask == 0] = np.nan

    mask_ax = fig.add_subplot(grid[:, 3])
    mask_ax.imshow(ops['meanImgE'][y_lim1:y_lim2, x_lim1:x_lim2], cmap='gray')
    mask_ax.imshow(neu_mask[y_lim1:y_lim2, x_lim1:x_lim2], cmap=plt.cm.get_cmap('spring').reversed(), alpha=0.5)
    mask_ax.title.set_text('neuropil mask')

    basename = fname + "_f-cell-neuropil_roi%s.svg"%which_roi
    fig_name = Path(out_dir).joinpath(basename)

    fig.savefig(fig_name, format="svg", dpi=1200)
    plt.close(fig)