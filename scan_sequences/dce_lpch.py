import os
from copy import deepcopy

from keras import backend as K
from numpy import np

from data_io import format_io_utils as fio_utils
from data_io.format_io import ImageDataFormat
from data_io.med_volume import MedicalVolume
from data_io.nifti_io import NiftiReader
from data_io.orientation import CORONAL
from defaults import DEFAULT_OUTPUT_IMAGE_DATA_FORMAT
from models.model import SegModel
from scan_sequences.scans import TargetSequence


class DceLpch(TargetSequence):
    NAME = 'dce_lpch'  # TODO (edgar): change name of class and the `NAME` field here
    __SCAN_SAVE_FILENAME_FORMAT__ = 'scan%d.nii.gz'

    def __init__(self, dicom_path, load_path=None):
        super().__init__(dicom_path=dicom_path, load_path=load_path)

        if not self.validate_dce():
            raise ValueError('dicoms in \'%s\' are not acquired from %s sequence' % (self.dicom_path, self.NAME))

    def validate_dce(self):
        """Validate that the dicoms are of expected DCE sequence by checking for dicom header tags
        # TODO (edgar)
        :return: a boolean
        """
        return False

    def segment(self, model, tissue):
        volume = deepcopy(self.volumes[0])
        mask = model.generate_mask(volume)

        tissue.set_mask(mask)

        return mask

    def save_data(self, base_save_dirpath: str, data_format: ImageDataFormat = DEFAULT_OUTPUT_IMAGE_DATA_FORMAT):
        super().save_data(base_save_dirpath, data_format=data_format)

        base_save_dirpath = self.__save_dir__(base_save_dirpath)

        # write echos
        for i in range(len(self.volumes)):
            nii_registration_filepath = os.path.join(base_save_dirpath, self.__SCAN_SAVE_FILENAME_FORMAT__ % (i + 1))
            filepath = fio_utils.convert_format_filename(nii_registration_filepath, data_format)
            self.volumes[i].save_volume(filepath, data_format=data_format)

    def load_data(self, base_load_dirpath):
        super().load_data(base_load_dirpath)

        base_load_dirpath = self.__save_dir__(base_load_dirpath, create_dir=False)

        self.volumes = []

        for i in range(1):
            nii_registration_filepath = os.path.join(base_load_dirpath, self.__SCAN_SAVE_FILENAME_FORMAT__ % (i + 1))
            subvolume = NiftiReader().load(nii_registration_filepath)
            self.volumes.append(subvolume)


class DCEKidneySegModel(SegModel):
    sigmoid_threshold = 0.5

    def __load_keras_model__(self, input_shape):
        # TODO (edgar): fill in architecture for the model
        pass

    def generate_mask(self, volume: MedicalVolume):
        """Segment the MRI volumes

        :param volume: A Medical Volume (height, width, slices)

        :rtype: A Medical volume with volume as binarized (0,1) uint8 3D numpy array of shape volumes.shape

        :raise ValueError if volumes is not 3D numpy array
        :raise ValueError if tissue is not a string or not in list permitted tissues

        """
        vol_copy = deepcopy(volume)

        # reorient to the expected coronal plane
        vol_copy.reformat(CORONAL)
        vol = vol_copy.volume

        # preprocess the volume
        vol = self.__preprocess_volume__(vol)

        # reshape volumes to be (slice, y, x, 1)
        v = np.transpose(vol, (2, 0, 1))
        v = np.expand_dims(v, axis=-1)

        model = self.keras_model
        mask = model.predict(v, batch_size=self.batch_size)

        # binarize mask
        mask = (mask > self.sigmoid_threshold).astype(np.uint8)

        # reshape mask to be (y, x, slice)
        mask = np.transpose(np.squeeze(mask, axis=-1), (1, 2, 0))

        K.clear_session()

        vol_copy.volume = mask

        # reorient to match with original volume
        vol_copy.reformat(volume.orientation)

        return vol_copy

    def __preprocess_volume__(self, volume: np.ndarray):
        """Preprocess the numpy array to prepare it for segmentation
        :param volume:
        :return:
        """
        # TODO (edgar): fill in necessary preprocessing
        return volume
