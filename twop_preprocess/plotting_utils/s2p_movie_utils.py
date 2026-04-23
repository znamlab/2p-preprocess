from skimage import io
from scipy.ndimage import uniform_filter1d


def simple_open_binaries(ops):
    """
    Open the primary binary file (raw or registered) for a Suite2p session.

    Args:
        ops (dict): Suite2p settings dictionary.

    Returns:
        list: A list containing the opened file handle.
    """
    reg_file = []
    if ops["keep_movie_raw"]:
        reg_file.append(open(ops["raw_file"], "rb"))
    else:
        reg_file.append(open(ops["reg_file"], "rb"))
    return reg_file


def load_s2p_reg_stack(reg_path):
    """
    Load registered TIFF files from a Suite2p run.

    Args:
        reg_path (str or Path): Path to the registered TIFF files.

    Returns:
        np.ndarray: The registered movie (n_frames x Ly x Lx).
    """
    im = io.imread(reg_path)
    return im


def moving_average_im(im, w=100):
    """
    Calculate a moving average of pixel values along the time axis.

    Args:
        im (np.ndarray): Movie array (n_frames x Ly x Lx).
        w (int, optional): Size of the window for the moving average. Default 100.

    Returns:
        np.ndarray: The smoothed movie array.
    """
    m_im = uniform_filter1d(im, size=w, axis=0, mode="nearest")
    return m_im


def write_moving_average_tif(im, out_dir, fname, w=100):
    """
    Calculate a moving average and save it as a TIFF file.

    Args:
        im (np.ndarray): Movie array (n_frames x Ly x Lx).
        out_dir (str or Path): Directory to save the output file.
        fname (str): Base name for the recording.
        w (int, optional): Size of the window for the moving average. Default 100.
    """

    # calculate moving average along time axis with window size w
    m_im = moving_average_im(im, w=w)
    # write m_im to file
    fname = fname + "_w%s_time-avg.tif" % w
    fname = out_dir + "/" + fname
    io.imsave(fname=fname, arr=m_im, check_contrast=False)
