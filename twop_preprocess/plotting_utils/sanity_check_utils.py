import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from scipy.stats import norm


def plot_trace(
    F,
    rois,
    plot_baseline=False,
    baseline=0,
    ncols=2,
    icol=0,
    linecolor="b",
    title="F",
    save_path=None,
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
    if save_path is not None:
        plt.savefig(save_path)


def plot_raw_trace(F, random_rois, Fneu=[], titles=["F", "Fneu"], save_path=None):
    plt.figure(figsize=(10, 3 * len(random_rois)))
    plot_trace(F, random_rois, ncols=2, icol=0, title=titles[0])
    if len(Fneu) > 0:
        plot_trace(Fneu, random_rois, ncols=2, icol=1, title=titles[1])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)


def plot_detrended_trace(
    F_original,
    F_trend,
    F_detrended,
    Fneu_original,
    Fneu_trend,
    Fneu_detrended,
    random_rois,
    save_path=None,
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
    if save_path is not None:
        plt.savefig(save_path)


def plot_dff(Fast, dff, F0, random_rois, save_path=None):
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
    if save_path is not None:
        plt.savefig(save_path)


def plot_fluorescence_matrices(
    F, Fneu, Fast, dff, neucoeff=0.7, max_frames=4000, save_path=None
):
    idx = np.min([F.shape[1], max_frames])
    to_plot = {
        "F": F[:, :idx],
        "Fneu": Fneu[:, :idx],
        "Fast": Fast[:, :idx],
        "dF/F": dff[:, :idx],
        f"F - {neucoeff} * Fneu": F[:, :idx] - Fneu[:, :idx] * neucoeff,
    }
    fig = plt.figure(figsize=(10, 20))
    for i, key in enumerate(to_plot.keys()):
        x = to_plot[key]
        plt.subplot(5, 1, i + 1)
        plt.imshow(
            (x - np.mean(x, axis=1)[:, None]) / np.std(x, axis=1)[:, None],
            vmin=-2,
            vmax=2,
            cmap="RdBu_r",
        )
        plt.title(key)
    return fig


def plot_offset_gmm(F, Fneu, cell_id, n_components, nframes=3000, save_path=None):
    fig = plt.figure(figsize=(10, 5))
    f = F[cell_id] - 0.7 * (Fneu[cell_id] - np.median(Fneu[cell_id]))
    ax = plt.subplot2grid((2, 5), (0, 0), colspan=4)
    s = len(f) // 2
    e = s + nframes
    plt.plot(F[cell_id, s:e], label="F")
    plt.plot(Fneu[cell_id, s:e], label="Fneu")
    plt.plot(f[s:e], label="F - 0.7 * Fneu")
    plt.legend(loc="upper right")
    plt.ylabel("Fluorescence (a.u.)")
    plt.title("Neuropil subtraction")
    plt.xlabel("Frame #")
    gmm = mixture.GaussianMixture(n_components=n_components, random_state=42).fit(
        f.reshape(-1, 1)
    )

    # find useful parameters
    gmm_order = np.argsort(gmm.means_[:, 0])
    gmm_means = gmm.means_[gmm_order]
    covs = gmm.covariances_[gmm_order]
    weights = gmm.weights_[gmm_order]

    bins = np.arange(np.percentile(f, 0.1), np.percentile(f, 99.9), 5)

    ax_hist = plt.subplot2grid((2, 5), (0, 4), colspan=1, sharey=ax)
    plt.hist(f, bins=bins, density=True, orientation="horizontal")
    comps = []
    for i in range(n_components):
        comps.append(
            norm.pdf(bins, float(gmm_means[i][0]), np.sqrt(float(covs[i][0][0])))
            * weights[i]
        )
        l = plt.plot(comps[i], bins)[0]
        plt.axhline(
            gmm_means[i], color=l.get_color(), label="f0" if i == 0 else "__no_label__"
        )
    comps = np.vstack(comps)
    plt.plot(comps.sum(axis=0), bins, linestyle="--", color="k")
    plt.legend(loc="upper right")
    plt.xlabel("Density")
    plt.title("GMM f0")

    ax_dff = plt.subplot2grid((2, 5), (1, 0), colspan=4, sharex=ax)
    f0 = gmm_means[0]
    this_dff = (f - f0) / f0
    ax_dff.plot(this_dff[s:e])
    plt.ylabel("dff")
    plt.xlabel("Frame #")

    ax_dff_hist = plt.subplot2grid((2, 5), (1, 4), colspan=1, sharey=ax_dff)
    dff_range = [this_dff.min(), this_dff.max()]
    plt.hist(
        this_dff,
        bins=np.linspace(*dff_range, 100),
        density=True,
        orientation="horizontal",
    )
    plt.xlabel("Density")
    ax.set_xlim(0, e - s)
    ax_dff.set_ylim(*dff_range)
    for x in fig.axes:
        x.axhline(0, color="k")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    return fig
