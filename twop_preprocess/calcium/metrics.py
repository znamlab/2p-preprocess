import numpy as np


def calculate_quality_metrics(f0, dff):
    """
    Calculate session-level quality metrics for each ROI.

    Args:
        f0 (np.ndarray): Baseline fluorescence (n_rois x n_frames or n_rois x 1).
        dff (np.ndarray): dF/F traces (n_rois x n_frames).

    Returns:
        dict: A dictionary containing:
            - f0_means (np.ndarray): Mean F0 per ROI.
            - median_dff (np.ndarray): Median dF/F per ROI.
            - max_dff (np.ndarray): Maximum absolute dF/F per ROI.
            - f0_bad_idx (np.ndarray): Indices of ROIs with F0 <= 0.
            - dff_median_bad_idx (np.ndarray): Indices of ROIs with median dF/F < 0.
            - dff_max_bad_idx (np.ndarray): Indices of ROIs with max abs dF/F > 100.0.
    """
    if f0.ndim == 2 and f0.shape[1] > 1:
        f0_means = np.nanmean(f0, axis=1)
    else:
        f0_means = f0.flatten()

    median_dff = np.nanmedian(dff, axis=1)
    max_dff = np.nanmax(np.abs(dff), axis=1)

    f0_bad_idx = np.where(f0_means <= 0)[0]
    dff_median_bad_idx = np.where(median_dff < 0)[0]
    dff_max_bad_idx = np.where(max_dff > 100.0)[0]

    return {
        "f0_means": f0_means,
        "median_dff": median_dff,
        "max_dff": max_dff,
        "f0_bad_idx": f0_bad_idx,
        "dff_median_bad_idx": dff_median_bad_idx,
        "dff_max_bad_idx": dff_max_bad_idx,
    }


def get_problematic_rois(metrics):
    """
    Get a unique list of problematic ROI indices from calculated metrics.
    """
    return np.unique(
        np.concatenate(
            [
                metrics["f0_bad_idx"],
                metrics["dff_median_bad_idx"],
                metrics["dff_max_bad_idx"],
            ]
        )
    ).astype(int)


def select_diagnostic_rois(valid_rois, problem_rois, n_random=5, n_problem_max=20):
    """
    Select a subset of ROIs for detailed diagnostic plotting.

    Prioritizes problematic ROIs (up to n_problem_max) and ensures a minimum
    number of total ROIs are selected by adding random ones if necessary.
    """
    # 1. Take a subset of problematic ROIs
    rois_to_plot = list(problem_rois[:n_problem_max])

    # 2. Ensure at least n_random total ROIs are plotted (even if not problematic)
    if len(rois_to_plot) < n_random:
        other_rois = [r for r in valid_rois if r not in rois_to_plot]
        if len(other_rois) > 0:
            n_needed = min(n_random - len(rois_to_plot), len(other_rois))
            extra_rois = np.random.choice(other_rois, n_needed, replace=False)
            rois_to_plot.extend(list(extra_rois))

    return sorted(rois_to_plot)
