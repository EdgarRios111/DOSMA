import os
import re
import unittest

import natsort

from data_io.format_io import ImageDataFormat

UNITTEST_DATA_PATH = os.path.join(os.path.dirname(__file__), '../unittest-data/')
UNITTEST_SCANDATA_PATH = os.path.join(UNITTEST_DATA_PATH, 'scans')
TEMP_PATH = os.path.join(UNITTEST_SCANDATA_PATH, 'temp')  # should be used when for writing with assert_raises clauses

SCANS = ['qdess', 'mapss', 'cubequant']
SCANS_INFO = {'mapss': {'expected_num_echos': 7},
              'qdess': {'expected_num_echos': 2},
              'cubequant': {'expected_num_echos': 4}}

SCAN_DIRPATHS = [os.path.join(UNITTEST_SCANDATA_PATH, x) for x in SCANS]

# Decimal precision for analysis (quantitative values, etc)
DECIMAL_PRECISION = 1  # (+/- 0.1ms)


def get_scan_dirpath(scan: str):
    for ind, x in enumerate(SCANS):
        if scan == x:
            return SCAN_DIRPATHS[ind]


def get_dicoms_path(fp):
    return os.path.join(fp, 'dicoms')


def get_write_path(fp, data_format: ImageDataFormat):
    return os.path.join(fp, 'multi-echo-write-%s' % data_format.name)


def get_read_paths(fp, data_format: ImageDataFormat):
    """Get ground truth data (produced by imageviewer like itksnap, horos, etc)"""
    base_name = os.path.join(fp, 'multi-echo-gt-%s' % data_format.name)
    files_or_dirs = os.listdir(base_name)
    fd = [x for x in files_or_dirs if re.match('e[0-9]+', x)]
    files_or_dirs = natsort.natsorted(fd)

    return [os.path.join(base_name, x) for x in files_or_dirs]


def get_data_path(fp):
    return os.path.join(fp, 'data')


def get_expected_data_path(fp):
    return os.path.join(fp, 'expected')


class ScanTest(unittest.TestCase):
    from scan_sequences.scans import ScanSequence
    SCAN_TYPE = ScanSequence  # override in subclasses

    def setUp(self):
        print("Testing: ", self._testMethodName)

    @classmethod
    def setUpClass(cls):
        cls.dicom_dirpath = get_dicoms_path(os.path.join(UNITTEST_SCANDATA_PATH, cls.SCAN_TYPE.NAME))
        cls.data_dirpath = get_data_path(os.path.join(UNITTEST_SCANDATA_PATH, cls.SCAN_TYPE.NAME))
