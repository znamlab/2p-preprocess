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
        if np.any(np.isnan(dff[i, :])):
            plt.title("NaN in dff, skipping plot")
        else:
            plt.hist(dff[i, :], bins=50)
            plt.title(f"median {np.round(np.median(rounded_dff[i,:]),2)}")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)


def plot_fluorescence_matrices(F, Fneu, Fast, dff, neucoeff=0.7, max_frames=4000):
    idx = np.min([F.shape[1], max_frames])
    to_plot = {
        "F": F[:, :idx],
        "Fneu": Fneu[:, :idx],
        "Fast": Fast[:, :idx],
        "dF/F": dff[:, :idx],
        f"F - {neucoeff} * Fneu": F[:, :idx] - Fneu[:, :idx] * neucoeff,
    }
    fig, axs = plt.subplots(len(to_plot), 1, figsize=(9, 22), layout="constrained")
    for ax, key in zip(axs.flat, to_plot.keys()):
        x = to_plot[key]
        ax.imshow(
            (x - np.mean(x, axis=1)[:, None]) / np.std(x, axis=1)[:, None],
            vmin=-2,
            vmax=2,
            cmap="RdBu_r",
            aspect="auto",
        )
        ax.set_title(key)
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


def plot_optical_offset_gmm(pixels, gmm, offset, save_path=None):
    """
    Illustrate the optical offset calculation by plotting the pixel intensity
    histogram and the fitted GMM components in both linear and log-y scales.

    Args:
        pixels (np.ndarray): Flattened pixel intensities from a raw frame.
        gmm (mixture.GaussianMixture): The fitted GMM object.
        offset (float): The estimated optical offset (lowest component mean).
        save_path (str or Path, optional): Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Determine histogram range
    hist_range = [np.percentile(pixels, 0.01), np.percentile(pixels, 99.9)]
    x = np.linspace(hist_range[0], hist_range[1], 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    responsibilities = gmm.predict_proba(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    gmm_order = np.argsort(gmm.means_[:, 0])

    for i, ax in enumerate(axes):
        is_log = i == 1
        # Plot histogram
        ax.hist(
            pixels,
            bins=100,
            range=hist_range,
            density=True,
            alpha=0.5,
            color="gray",
            label="Pixel intensities",
        )

        # Plot GMM components
        for j, idx in enumerate(gmm_order):
            ax.plot(
                x,
                pdf_individual[:, idx],
                label=f"Comp {j} (μ={gmm.means_[idx, 0]:.2f})",
            )

        ax.plot(x, pdf, "k--", label="Full GMM")

        # Highlight the selected offset
        ax.axvline(
            offset,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Offset: {offset:.2f}",
        )

        if is_log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-6, top=pdf.max() * 2)
            ax.set_ylabel("Density (Log)")
            ax.set_title("Log-y scale")
        else:
            ax.set_ylabel("Density")
            ax.set_title("Linear scale")

        ax.set_xlabel("Intensity (a.u.)")
        ax.legend(loc="upper right", fontsize="small")
        ax.grid(alpha=0.3, which="both" if is_log else "major")

    fig.suptitle("Optical Offset Estimation (GMM Fit to Raw Frame)", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    return fig


def plot_population_metrics(f0, dff, save_path=None):
    """
    Plot population-level quality metrics: F0 distribution, median dF/F distribution,
    and frequency of large spikes.

    Args:
        f0 (np.ndarray): F0 values (n_rois x n_frames or n_rois x 1).
        dff (np.ndarray): dF/F values (n_rois x n_frames).
        save_path (str, optional): Path to save the plot.
    """
    n_rois = dff.shape[0]

    # Calculate metrics
    # F0 can be a trace or a single value per ROI
    if f0.ndim == 2 and f0.shape[1] > 1:
        f0_means = np.nanmean(f0, axis=1)
    else:
        f0_means = f0.flatten()

    median_dff = np.nanmedian(dff, axis=1)
    max_dff = np.nanmax(dff, axis=1)

    f0_below_zero = np.sum(f0_means < 0)
    median_dff_below_zero = np.sum(median_dff < 0)
    large_spikes = np.sum(max_dff > 10.0)  # > 1000% is 10.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. F0 Distribution
    axes[0].hist(f0_means, bins=50, color="skyblue", edgecolor="black")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title(f"F0 Distribution\n({f0_below_zero}/{n_rois} cells < 0)")
    axes[0].set_xlabel("F0 value")
    axes[0].set_ylabel("Count")

    # 2. Median dF/F Distribution
    axes[1].hist(median_dff, bins=50, color="salmon", edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title(
        f"Median dF/F Distribution\n({median_dff_below_zero}/{n_rois} cells < 0)"
    )
    axes[1].set_xlabel("Median dF/F")
    axes[1].set_ylabel("Count")

    # 3. Max dF/F (Spikes) Distribution
    axes[2].hist(max_dff, bins=50, color="lightgreen", edgecolor="black")
    axes[2].axvline(10, color="red", linestyle="--")
    axes[2].set_title(f"Max dF/F Distribution\n({large_spikes}/{n_rois} cells > 1000%)")
    axes[2].set_xlabel("Max dF/F")
    axes[2].set_ylabel("Count")
    axes[2].set_yscale("log")  # Log scale often better for max values

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
