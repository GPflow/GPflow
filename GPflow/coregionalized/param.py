# -*- coding: utf-8 -*-
from ..param import AutoFlow
from .labeled_data import LabeledData


class LabeledAutoFlow(AutoFlow):
    """
    Modifited AutoFlow class, that can handle LabeledData
    """
    def get_a_data_holder(self, data_class, np_arg, instance=None):
        """
        For the case where data_class == LabeledData,
        np_array should be a tuple (data, index) with appropirate shapes.
        
        """
        if data_class == LabeledData:
            return LabeledData(np_arg, on_shape_change='pass', \
                                            num_labels=instance.num_labels)
        
        return AutoFlow.get_a_data_holder(self, data_class, np_arg, instance)
                
    
    def set_to_data_holder(self, data_holder, np_arg):
        """
        A method to set array into data holder.
        """
        if isinstance(data_holder, LabeledData):
            data_holder.set_data(np_arg)
        else:
            AutoFlow.set_to_data_holder(self, data_holder, np_arg)
            
            
    