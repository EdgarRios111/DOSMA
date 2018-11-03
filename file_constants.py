"""Basic file constants to be shared with program"""

import os

DEBUG = 0

# Elastix files

__DIR__ = os.path.dirname(__file__)
__PATH_TO_ELASTIX_FOLDER__ = os.path.join(__DIR__, 'elastix_params')

ELASTIX_AFFINE_PARAMS_FILE = os.path.join(__PATH_TO_ELASTIX_FOLDER__, 'parameters-affine.txt')
ELASTIX_BSPLINE_PARAMS_FILE = os.path.join(__PATH_TO_ELASTIX_FOLDER__, 'parameters-bspline.txt')
ELASTIX_RIGID_PARAMS_FILE = os.path.join(__PATH_TO_ELASTIX_FOLDER__, 'parameters-rigid.txt')

ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE = os.path.join(__PATH_TO_ELASTIX_FOLDER__,
                                                        'parameters-affine-interregister.txt')
ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE = os.path.join(__PATH_TO_ELASTIX_FOLDER__,
                                                       'parameters-rigid-interregister.txt')

# Temporary file path
TEMP_FOLDER_PATH = os.path.join(__DIR__, 'temp')

# TODO: nipype logging
NIPYPE_LOGGING = 'stream'


def set_debug():
    global DEBUG, NIPYPE_LOGGING
    DEBUG = 1
    NIPYPE_LOGGING = 'stream'
