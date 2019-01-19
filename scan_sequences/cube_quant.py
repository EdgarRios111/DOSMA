import os

from natsort import natsorted
from nipype.interfaces.elastix import Registration

import file_constants as fc
from scan_sequences.scans import NonTargetSequence
from utils import io_utils
from utils import quant_vals as qv
from utils.fits import MonoExponentialFit

__EXPECTED_NUM_SPIN_LOCK_TIMES__ = 4
__R_SQUARED_THRESHOLD__ = 0.9
__INITIAL_T1_RHO_VAL__ = 70.0

__T1_RHO_LOWER_BOUND__ = 0
__T1_RHO_UPPER_BOUND__ = 500
__T1_RHO_DECIMAL_PRECISION__ = 3


class CubeQuant(NonTargetSequence):
    NAME = 'cubequant'

    def __init__(self, dicom_path=None, dicom_ext=None, load_path=None):
        super().__init__(dicom_path, dicom_ext, load_path=load_path)

        self.subvolumes = None

        # Only load data if dicom path is not given, else assume user wants to rewrite information
        if load_path and dicom_path is None:
            self.load_data(load_path)

        if dicom_path is not None:
            self.subvolumes, self.spin_lock_times = self.__split_volumes__(__EXPECTED_NUM_SPIN_LOCK_TIMES__)
            self.intraregistered_data = self.__intraregister__(self.subvolumes)

        if self.subvolumes is None:
            raise ValueError('Either dicom_path or load_path must be specified')

    def interregister(self, target_path, mask_path=None):
        base_spin_lock_time, base_image = self.intraregistered_data['BASE']
        files = self.intraregistered_data['FILES']

        temp_interregistered_dirpath = io_utils.check_dir(os.path.join(self.temp_path, 'interregistered'))

        print('')
        print('==' * 40)
        print('Interregistering...')
        print('Target: %s' % target_path)
        if mask_path is not None:
            print('Mask: %s' % mask_path)
        print('==' * 40)

        if not mask_path:
            parameter_files = [fc.ELASTIX_RIGID_PARAMS_FILE, fc.ELASTIX_AFFINE_PARAMS_FILE]
        else:
            parameter_files = [fc.ELASTIX_RIGID_INTERREGISTER_PARAMS_FILE, fc.ELASTIX_AFFINE_INTERREGISTER_PARAMS_FILE]

        warped_file, transformation_files = self.__interregister_base_file__((base_image, base_spin_lock_time),
                                                                             target_path,
                                                                             temp_interregistered_dirpath,
                                                                             mask_path=mask_path,
                                                                             parameter_files=parameter_files)
        warped_files = [(base_spin_lock_time, warped_file)]

        # Load the transformation file. Apply same transform to the remaining images
        for spin_lock_time, filename in files:
            warped_file = self.__apply_transform__((filename, spin_lock_time),
                                                   transformation_files,
                                                   temp_interregistered_dirpath)
            # append the last warped file - this has all the transforms applied
            warped_files.append((spin_lock_time, warped_file))

        # copy each of the interregistered warped files to their own output
        subvolumes = dict()
        for spin_lock_time, warped_file in warped_files:
            subvolumes[spin_lock_time] = io_utils.load_nifti(warped_file)

        self.subvolumes = subvolumes

    def generate_t1_rho_map(self, tissues=None):
        """Generate 3D T1-rho map and r2 fit map using monoexponential fit across subvolumes acquired at different
                echo times
        :param tissues: A list of Tissue instances specifying which tissue to examine
                        if None, use list of tissues class initialized with
        :return: a list of T1Rho instances
        """

        if tissues is None:
            tissues = self.tissues

        quant_maps = []
        for tissue in tissues:
            spin_lock_times = []
            subvolumes_list = []

            # only calculate for focused region if a mask is available, this speeds up computation
            mask = tissue.get_mask()
            import pdb; pdb.set_trace()
            sorted_keys = natsorted(list(self.subvolumes.keys()))
            for spin_lock_time_index in sorted_keys:
                subvolumes_list.append(self.subvolumes[spin_lock_time_index])
                spin_lock_times.append(self.spin_lock_times[spin_lock_time_index])

            mef = MonoExponentialFit(spin_lock_times, subvolumes_list,
                                     mask=mask,
                                     bounds=(__T1_RHO_LOWER_BOUND__, __T1_RHO_UPPER_BOUND__),
                                     tc0=__INITIAL_T1_RHO_VAL__,
                                     decimal_precision=__T1_RHO_DECIMAL_PRECISION__)

            t1rho_map, r2 = mef.fit()

            quant_val_map = qv.T1Rho(t1rho_map)
            quant_val_map.add_additional_volume('r2', r2)

            quant_maps.append(quant_val_map)

            tissue.add_quantitative_value(quant_val_map)

        return quant_maps

    def __intraregister__(self, subvolumes):
        """Intraregister cubequant subvolumes to each other
        Patient could have moved between acquisition of different subvolumes, so different subvolumes of cubequant scan
            have to be registered with each other

        The first spin lock time has the highest SNR, so it is used as the target
        Subvolumes corresponding to the other spin lock times are registered to the target

        Affine registration is done using elastix

        :param subvolumes: dictionary of subvolumes mapping spin lock time index --> MedicalVolume
                                (e.g. {0 --> MedicalVolume A, 1 --> MedicalVolume B}
        :return: a dictionary of base, other files spin-lock index --> nifti filepath
        """

        if subvolumes is None:
            raise ValueError('subvolumes must be dict()')

        print('')
        print('==' * 40)
        print('Intraregistering...')
        print('==' * 40)

        # temporarily save subvolumes as nifti file
        ordered_spin_lock_time_indices = natsorted(list(subvolumes.keys()))
        raw_volumes_base_path = io_utils.check_dir(os.path.join(self.temp_path, 'raw'))

        # Use first spin lock time as a basis for registration
        spin_lock_nii_files = []
        for spin_lock_time_index in ordered_spin_lock_time_indices:
            filepath = os.path.join(raw_volumes_base_path, '%03d' % spin_lock_time_index + '.nii.gz')
            spin_lock_nii_files.append(filepath)

            subvolumes[spin_lock_time_index].save_volume(filepath)

        target_filepath = spin_lock_nii_files[0]

        intraregistered_files = []
        for i in range(1, len(spin_lock_nii_files)):
            spin_file = spin_lock_nii_files[i]
            spin_lock_time_index = ordered_spin_lock_time_indices[i]

            reg = Registration()
            reg.inputs.fixed_image = target_filepath
            reg.inputs.moving_image = spin_file
            reg.inputs.output_path = io_utils.check_dir(os.path.join(self.temp_path,
                                                                     'intraregistered',
                                                                     '%03d' % spin_lock_time_index))
            reg.inputs.parameters = [fc.ELASTIX_AFFINE_PARAMS_FILE]
            reg.terminal_output = fc.NIPYPE_LOGGING
            print('Registering %s --> %s' % (str(spin_lock_time_index), str(ordered_spin_lock_time_indices[0])))
            tmp = reg.run()

            warped_file = tmp.outputs.warped_file
            intraregistered_files.append((spin_lock_time_index, warped_file))

        return {'BASE': (ordered_spin_lock_time_indices[0], spin_lock_nii_files[0]),
                'FILES': intraregistered_files}

    def save_data(self, base_save_dirpath):
        super().save_data(base_save_dirpath)
        base_save_dirpath = self.__save_dir__(base_save_dirpath)

        # Save interregistered files
        interregistered_dirpath = os.path.join(base_save_dirpath, 'interregistered')

        for spin_lock_time_index in self.subvolumes.keys():
            filepath = os.path.join(interregistered_dirpath, '%03d.nii.gz' % spin_lock_time_index)
            self.subvolumes[spin_lock_time_index].save_volume(filepath)

    def load_data(self, base_load_dirpath):
        super().load_data(base_load_dirpath)
        base_load_dirpath = self.__save_dir__(base_load_dirpath, create_dir=False)

        interregistered_dirpath = os.path.join(base_load_dirpath, 'interregistered')

        self.subvolumes = self.__load_interregistered_files__(interregistered_dirpath)

    def __serializable_variables__(self):
        var_names = super().__serializable_variables__()
        var_names.extend(['spin_lock_times'])
        return var_names
