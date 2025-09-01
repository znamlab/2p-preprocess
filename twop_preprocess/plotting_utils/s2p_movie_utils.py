from skimage import io
from scipy.ndimage import uniform_filter1d

def simple_open_binaries(ops):
    
    reg_file = []

    if ops["keep_movie_raw"]:
        reg_file.append(open(ops["raw_file"], "rb"))
    else:
        reg_file.append(open(ops["reg_file"], "rb"))

    return reg_file


def moving_average_im(im, w=100):
    """
    Calculates moving average of pixel values of single plane recording in a numpy array of nframes x xpix x ypix

    :param im: ndarray, array with single plane recording of nframes x xpix x ypix
    :param w: int, size of window for calculating the moving average
    :return m_im: ndarray, moving average of pixel values along time axis in array of nframes x xpix x ypix
    """
    m_im = uniform_filter1d(im, size=w, axis=0, mode='nearest')
    return m_im

def write_moving_average_tif(im, out_dir, fname, w=100):
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

