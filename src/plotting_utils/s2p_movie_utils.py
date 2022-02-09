from skimage import io
from scipy.ndimage import uniform_filter1d

# add function for loading image from collection of tif files from s2p output
# with file[0-9]_chan[0-9] naming pattern

# add skimage to environment.yml for 2p-preprocess?

def load_s2p_reg_stack(reg_path):
    """
    loads registered tif files from run_s2p with reg_tif=True

    :param reg_path: str, path to registered tif files (including reg_tif directory)
    :return im: ndarray, array of registered movie of nframes x xpix x ypix
    """
    im = io.imread(reg_path)
    return im

def moving_average_im(im, w=100):
    """
    Calculates moving average of pixel values of single plane recording in a numpy array of nframes x xpix x ypix

    :param im: ndarray, array with single plane recording of nframes x xpix x ypix
    :param w: int, size of window for calculating the moving average
    :return m_im: ndarray, moving average of pixel values along time axis in array of nframes x xpix x ypix
    """
    m_im = uniform_filter1d(im, size=w, axis=0, mode='nearest')
    return m_im

def write_moving_average_tif(im, w=100, out_dir, fname):
    """
    Writes output of moving_average_im to a tif file

    :param im: ndarray, array with single plane recording of nframes x xpix x ypix
    :param w: int, size of window for calculating the moving average
    :param out_dir: str, path to output directory for writing tif file
    :param fname: str, basename for recording and run, if applicable
    :return: none
    """

    # calculate moving average along time axis with window size w
    m_im = moving_average_im(im, w=w)
    # write m_im to file
    fname = fname + "_w%s_time-avg.tif"%w
    fname = out_dir + "/" + fname
    io.imsave(fname=fname, arr=m_im, check_contrast=False)

