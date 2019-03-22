from tissues.tissue import Tissue


class Kidney(Tissue):
    ID = 1001  # should be unique to all tissues, and should not change - replace with a unique identifier
    STR_ID = 'kidney'  # short hand string id such as 'fc' for femoral cartilage
    FULL_NAME = 'kidney'  # full name of tissue 'femoral cartilage' for femoral cartilage

    def split_regions(self, base_map):
        """
        Split mask into anatomical regions
        :param base_map: a 3D numpy array
        :return: a 4D numpy array (region, height, width, depth) - save in variable self.regions
        """
        raise NotImplementedError('Cannot currently split %s into observable regions' % self.FULL_NAME)

    def __calc_quant_vals__(self, quant_map, map_type):
        """
        Private method to get quantitative values for tissue - implemented by tissue
        :param quant_map: a 3D numpy array for quantitative measures (t2, t2*, t1-rho, etc)
        :param map_type: an enum instance of QuantitativeValue
        :return: a dictionary of quantitative values, save in quant_vals
        """
        raise NotImplementedError('No quantitative value analysis currently supported for %s' % self.FULL_NAME)

    def __save_quant_data__(self, dirpath):
        """
        Save quantitative data generated for this tissue
        :param dirpath: Path to directory where to save quantitative information
        :return:
        """
        pass
