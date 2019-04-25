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
    
    NAME = 'dce_lpch'                                       # TODO (edgar): change name of class and the `NAME` field here --> Edgar: dce_lpch is fine for now
    __SCAN_SAVE_FILENAME_FORMAT__ = 'scan%d.nii.gz'

    # DCE DICOM header keys from metadata
    __NumberofTemporalPositions__ = 50
    __ImagesinAcquisition__ = 2500 #5000

    def __init__(self, dicom_path, load_path=None):
        super().__init__(dicom_path=dicom_path, load_path=load_path)

        if not self.validate_dce():
            raise ValueError('dicoms in \'%s\' are not acquired from %s sequence' % (self.dicom_path, self.NAME))

    def validate_dce(self):
        #Edgar - Validate that the dicoms are of expected DCE sequence by checking for dicom header tags
        #return: a boolean
        ref_dicom = self.ref_dicom  # And this ref_dicom is obtained from...?
        return self.__NumberofTemporalPositions__ in ref_dicom and self.__ImagesinAcquisition__ in ref_dicom #and len(self.volumes) == self.__NUM_ECHOS__   

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
        # Edgar - The model.h5 saves the model architecture, right?

        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        s = Lambda(lambda x: x / 255) (inputs)

        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
        c1 = Dropout(0.1) (c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
        c2 = Dropout(0.1) (c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
        c3 = Dropout(0.2) (c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
        c4 = Dropout(0.2) (c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
        c5 = Dropout(0.3) (c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
        c6 = Dropout(0.2) (c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
        c7 = Dropout(0.2) (c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
        c8 = Dropout(0.1) (c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
        c9 = Dropout(0.1) (c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

        # Fit model
        # ...

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

        # reshape volumes to be (slice, y, x, 1)
        v = self.SHIFT_DIMENSIONS_XYZ2ZXYC(vol, 50)                                       # Needed?

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

        # Not a lot of preprocessing needed 
        # - Maybe reduce number of phases (future use)
        # - Maybe shift dimensions, but depends on how the data is loaded, whihc I think happened already
        # - PROBALY NEED TO RESIZE...

        #model = load_model(OUTPUTMODEL_PATH, custom_objects={'mean_iou': mean_iou})

        #Volume_MRI = GET_ONE_NIIGZ(FoldersPaths[i].replace('Output/','')+ TESTINPUT_FILENAME)       # Arjun already took care of loading the dicoms/niftys right?
        Volume_MRI = volume
        Volume_MRI = self.SHIFT_DIMENSIONS_XYZ2ZXYC(Volume_MRI, 50)                                       # Needed?
        
        # For future use for when using only a few phases
        #if CHANNELS2TAKE != 50: Volume_MRI = Volume_MRI[:,:,:,0:CHANNELS2TAKE]
        
        # Getting the predictions                             # replacement --> generate_mask(self, volume: MedicalVolume)
        # PROBMAP3D = model.predict(Volume_MRI, verbose=1)  
        # PROBMAP3D = SHIFT_DIMENSIONS_ZXYC2XYZ(PROBMAP3D)
        # PROBMAP3D = np.fliplr(PROBMAP3D)                      # Needed?
        # PROBMAP3D = np.rot90(PROBMAP3D)                       # Needed?
        # BINARYV3D = BINARIZE_VOLUME(PROBMAP3D, 0.4)
        
        # Saving result                                       # replacement --> ...
        # PROBMAP3D = nib.Nifti1Image(PROBMAP3D, np.eye(4))
        # nib.save(PROBMAP3D, PredictedVolumePath)
        # BINARYV3D = nib.Nifti1Image(BINARYV3D, np.eye(4)) 
        # nib.save(BINARYV3D, PredictedVolumePath2)

        return Volume_MRI

    # def mean_iou(self, y_true, y_pred):
    
    #     # Define IoU metric
    #     prec = []
    #     for t in np.arange(0.5, 1.0, 0.05):
    #         y_pred_ = tf.to_int32(y_pred > t)
    #         score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
    #         K.get_session().run(tf.local_variables_initializer())
    #         with tf.control_dependencies([up_opt]):
    #             score = tf.identity(score)
    #         prec.append(score)
    #     return K.mean(K.stack(prec), axis=0)


    def SHIFT_DIMENSIONS_XYZ2ZXYC(self, Volume3D, Channels):
   
        # I have a X Y Z volume and UNET asks for a Z X Y C volume (C is for image_channels, like in RGB)
        [X,Y,Z] = Volume3D.shape
        C = Channels
        Z = int(Z/C)             # int(Z/C) dividing the number of new slices by the num of channels (phases) -->  5kimgs n 50 phases, 100 slices per vol, if C=1 no pasa naa'
        OutVol = np.zeros(shape = (Z, X, Y, C))   
        for c in range(C):
            c = c*Z #c*100
            for z in range(Z):
                temp = Volume3D[:,:,c+z]
                OutVol[z,:,:,int(c/100)] = temp
        return OutVol