import os
from copy import deepcopy

import numpy as np
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model

from data_io import format_io_utils as fio_utils
from data_io.format_io import ImageDataFormat
from data_io.med_volume import MedicalVolume
from data_io.nifti_io import NiftiReader
from data_io.orientation import CORONAL
from defaults import DEFAULT_OUTPUT_IMAGE_DATA_FORMAT
from models.model import SegModel
from scan_sequences.scans import TargetSequence
from tissues.tissue import Tissue
from utils.cmd_line_utils import ActionWrapper

TEMPORAL_POSITION_IDENTIFIER_TAG = 0x00200100
NUM_TEMPORAL_POSITIONS_TAG = 0x00200105
IMAGES_IN_ACQUISITION_TAG = 0x00201002


class DceLpch(TargetSequence):
    NAME = 'dce_lpch'
    __SCAN_SAVE_FILENAME_FORMAT__ = 'scan%d.nii.gz'

    # DCE DICOM header keys from metadata
    __NumberofTemporalPositions__ = 50
    __ImagesinAcquisition__ = 2500  # 5000

    def __init__(self, dicom_path, load_path=None):
        super().__init__(dicom_path=dicom_path, load_path=load_path)

        if not self.__validate_scan__():
            raise ValueError('dicoms in \'%s\' are not acquired from %s sequence' % (self.dicom_path, self.NAME))

    def __validate_scan__(self):
        ref_dicom = self.ref_dicom
        return self.__NumberofTemporalPositions__ == ref_dicom[
            NUM_TEMPORAL_POSITIONS_TAG].value and self.__ImagesinAcquisition__ == ref_dicom[
                   IMAGES_IN_ACQUISITION_TAG].value

    def segment(self, model: SegModel, tissue: Tissue):
        volume = deepcopy(self.volumes[0])
        if model != 'dce_kidney':
            raise ValueError('Segmentation model not found')

        k_model = DCEKidneySegModel(input_shape=(volume.volume.shape[0], volume.volume.shape[1], 50),
                                    weights_path=tissue.weights_filepath)

        mask = k_model.generate_mask(volume)

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

    @classmethod
    def cmd_line_actions(cls):
        """Provide command line information (such as name, help strings, etc) as list of dictionary"""

        segment_action = ActionWrapper(name=cls.segment.__name__,
                                       help='generate automatic segmentation',
                                       param_help={
                                           'model': 'the model to use. Currently only supports `dce_kidney`'},
                                       alternative_param_names={'use_rms': ['rms']})

        return [(cls.segment, segment_action)]


class DCEKidneySegModel(SegModel):
    sigmoid_threshold = 0.4
    num_channels = 50

    def __load_keras_model__(self, input_shape):
        inputs = Input(input_shape)
        s = Lambda(lambda x: x / 255)(inputs)

        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])

        return model

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

        # reshape volumes to be (z, y, x, c)
        v = self.shift_dimensions_yxz2zyxc(vol, 50)

        model = self.keras_model
        mask = model.predict(v, batch_size=self.batch_size)

        # binarize mask
        mask = (mask > self.sigmoid_threshold).astype(np.uint8)

        # reshape mask to be (y, x, slice)
        mask = np.transpose(np.squeeze(mask, axis=-1), (1, 2, 0))

        K.clear_session()

        vol_copy = MedicalVolume(mask, affine=deepcopy(vol_copy.affine), headers=deepcopy(vol_copy.headers[:mask.shape[-1]]))

        # reorient to match with original volume
        vol_copy.reformat(volume.orientation)

        return vol_copy

    def shift_dimensions_yxz2zyxc(self, volume: np.ndarray, num_channels):
        """
        Reformat 3D volume of shape (Y,X,Z*C) to a 4D volume of shape (Z, Y, X, C)
        :param volume: a numpy array
        :param num_channels: the number of channels (C) that are in the volume
        :return:
        """

        # I have a X Y Z volume and UNET asks for a Z X Y C volume (C is for image_channels, like in RGB)
        [X, Y, Z] = volume.shape
        C = num_channels
        Z = int(
            Z / C)  # int(Z/C) dividing the number of new slices by the num of channels (phases) -->  5kimgs n 50 phases, 100 slices per vol, if C=1 no pasa naa'
        OutVol = np.zeros(shape=(Z, X, Y, C))
        for c in range(C):
            c = c * Z  # c*100
            for z in range(Z):
                temp = volume[:, :, c + z]
                OutVol[z, :, :, int(c / 100)] = temp
        return OutVol
