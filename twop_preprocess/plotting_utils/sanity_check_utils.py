import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot_trace(
    F, rois, plot_baseline=False, baseline=0, ncols=2, icol=0, linecolor="b", title="F"
):
    for i, roi in enumerate(rois):
        plt.subplot2grid((len(rois), ncols), (i, icol))
        plt.plot(F[roi, :], c=linecolor)
        plt.title(f"ROI{roi} {title}")
        if plot_baseline:
            if len(baseline[roi, :]) == 1:
                plt.axhline(baseline[roi, :], color="r")
            else:
                plt.plot(baseline[roi, :], color="r")


def plot_raw_trace(F, random_rois, Fneu=[], titles=["F", "Fneu"]):
    plt.figure(figsize=(10, 3 * len(random_rois)))
    plot_trace(F, random_rois, ncols=2, icol=0, title=titles[0])
    if len(Fneu) > 0:
        plot_trace(Fneu, random_rois, ncols=2, icol=1, title=titles[1])
    plt.tight_layout()


def plot_detrended_trace(
    F_original,
    F_trend,
    F_detrended,
    Fneu_original,
    Fneu_trend,
    Fneu_detrended,
    random_rois,
):
    plt.figure(figsize=(20, 3 * len(random_rois)))
    plot_trace(
        F_original,
        random_rois,
        plot_baseline=True,
        baseline=F_trend,
        ncols=4,
        icol=0,
        title="F",
    )
    plot_trace(F_detrended, random_rois, ncols=4, icol=1, title="F_detrended")
    plot_trace(
        Fneu_original,
        random_rois,
        plot_baseline=True,
        baseline=Fneu_trend,
        ncols=4,
        icol=2,
        title="Fneu",
    )
    plot_trace(Fneu_detrended, random_rois, ncols=4, icol=3, title="Fneu_detrended")
    plt.tight_layout()


def plot_dff(Fast, dff, F0, random_rois):
    plt.figure(figsize=(20, 3 * len(random_rois)))
    plot_trace(
        Fast,
        random_rois,
        plot_baseline=True,
        baseline=F0,
        ncols=4,
        icol=0,
        title="Fast",
    )
    plot_trace(dff, random_rois, ncols=4, icol=1, title="dff")
    plot_trace(dff[:, 5000:6000], random_rois, ncols=4, icol=2, title="dff")
    rounded_dff = np.round(dff, 2)
    for i, roi in enumerate(random_rois):
        plt.subplot2grid((len(random_rois), 4), (i, 3))
        plt.hist(dff[i, :], bins=50)
        plt.title(f"median {np.round(np.median(rounded_dff[i,:]),2)}")
    plt.tight_layout()


def plot_fluorescence_matrices(F, Fneu, Fast, dff, neucoeff=0.7, max_frames=4000):
    idx = np.min([F.shape[1], max_frames])
    to_plot = {
        "F": F[:, :idx],
        "Fneu": Fneu[:, :idx],
        "Fast": Fast[:, :idx],
        "dF/F": dff[:, :idx],
        f"F - {neucoeff} * Fneu": F[:, :idx] - Fneu[:, :idx] * neucoeff,
    }
    fig, axs  = plt.subplots(len(to_plot), 1, figsize=(9, 22), layout="constrained")
    for ax, key in zip(axs.flat,to_plot.keys()):
        x = to_plot[key]
        ax.imshow(
            (x - np.mean(x, axis=1)[:, None]) / np.std(x, axis=1)[:, None],
            vmin=-2,
            vmax=2,
            cmap="RdBu_r",
            aspect="auto",
        )
        ax.set_title(key)