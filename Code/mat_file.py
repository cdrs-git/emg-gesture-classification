# .mat file handling
import scipy.io as spio

def extract_mat_file(path):
    """
    Extracts x from .mat file specified in path

    This function returns x

    Args:
        x

    Returns:
        x
    """
    mat = spio.loadmat(path, squeeze_me=True)
    # Try to extract signal labels. If not d1 input, it will return empty list
    try:
        subject         = mat['subject']
        exercise        = mat['exercise']
        emg             = mat['emg']
        stimulus        = mat['stimulus']
        restimulus      = mat['restimulus']
        repetition      = mat['repetition']
        rerepetition    = mat['rerepetition']
    except Exception:
        raise Exception('Invalid .mat file')

    return subject, exercise, emg, stimulus, restimulus, repetition, rerepetition

def create_mat_file(output_file_path, subject, exercise, emg, stimulus, restimulus, repetition, rerepetition):
    """
    Creates a .mat file in path, containing x

    Args:
        output_file_path (path): Location of .mat file create/ store in.
        x

    """
    # Create dictionary
    dictionary = {
    'subject'       : subject,
    'exercise'      : exercise,
    'emg'           : emg,
    'restimulus'    : restimulus,
    'rerepetition'  : rerepetition,
    }
    spio.savemat(output_file_path, dictionary)