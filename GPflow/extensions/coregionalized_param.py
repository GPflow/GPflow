# -*- coding: utf-8 -*-
import tensorflow as tf
from functools import wraps
from ..param import DictData
from .labeled_data import LabeledData


class AutoFlow_base:
    """
    This is an new AutoFlow implementation with DictData
    """
    def __init__(self, *data_classes):
        # NB. TF arg_tuples is a list of tuples, each of which can be used to
        # construct a tf placeholder.
        self.data_classes= data_classes

    def __call__(self, tf_method):
        @wraps(tf_method)
        def runnable(instance, *np_args):
            storage_name = '_' + tf_method.__name__ + '_AF_storage'
            if hasattr(instance, storage_name):
                # the method has been compiled already, get things out of storage
                storage = getattr(instance, storage_name)
            else:
                # the method needs to be compiled.
                storage = {}  # an empty dict to keep things in
                setattr(instance, storage_name, storage)
                # storage dict_data for the arguments
                storage['data_holders'] = []
                for data_class, np_arg in zip(self.data_classes, np_args):
                    storage['data_holders'].append(get_a_data_holder(incetance, data_class, np_arg))
                
                storage['free_vars'] = tf.placeholder(tf.float64, [None])
                instance.make_tf_array(storage['free_vars'])
                
                with instance.tf_mode():
                    storage['tf_result'] = tf_method(instance, storage['data_holders'])
                                
                storage['session'] = tf.Session()
                storage['session'].run(tf.initialize_all_variables(), feed_dict=instance.get_feed_dict())
            
            # set array into dict_data
            for d, np_arg in zip(stroage['dict_data'], np_args):
                d.set_data(np_arg)
            feed_dict[storage['free_vars']] = instance.get_free_state()
            feed_dict.update(instance.get_feed_dict())
            for d in stroage['dict_data']:
                feed_dict.update(d.get_feed_dict())
            return storage['session'].run(storage['tf_result'], feed_dict=feed_dict)

        return runnable


    def get_a_data_holder(self, incetance, data_class, np_array):
        """
        A method to create a DataHolder from data_class and np_arg.
        
        data_class should be a class that inheriting from DataHolder.
        
        np_array is an np.array that will be stored in the DataHolder
        """
        if data_class == DictData:
            data_holders.append(DictData(np_arg), on_shape_change_array='pass')
        else:
            raise Exception('%s Data_Holder class is not implemented.', data_class)
        return data_holders
        
    def set_data(self, data_hoders, np_args):
        for (data_class, data_holder, np_arg) in zip(self.data_classes, data_holders, np_args):
            if data_class == DictData:
                data_
        
class LabeledAutoFlow:
    """
    This is an DataHolder version of AutoFlow. 
    """
    def __init__(self, *tf_arg_tuples):
        # NB. TF arg_tuples is a list of tuples, each of which can be used to
        # construct a tf placeholder.
        self.tf_arg_tuples = tf_arg_tuples

    def __call__(self, tf_method):
        @wraps(tf_method)
        def runnable(instance, *np_args):
            storage_name = '_' + tf_method.__name__ + '_AF_storage'
            if hasattr(instance, storage_name):
                # the method has been compiled already, get things out of storage
                storage = getattr(instance, storage_name)
            else:
                # the method needs to be compiled.
                storage = {}  # an empty dict to keep things in
                setattr(instance, storage_name, storage)
                storage['free_vars'] = tf.placeholder(tf.float64, [None])
                instance.make_tf_array(storage['free_vars'])
                
                if not hasattr(instance, 'num_labels'):
                    raise Exception('CoregionalizedAutoFlow instance must be coregionalized_model')
            
                # construct LabelData for the prediction.
                # TODO np_arg should be appropriately provided.
                label = LabeledData(np_arg, instance.num_labels)
                storage['tf_args'] = [tf.placeholder(*a) for a in self.tf_arg_tuples if ***]
                storage['tf_args'].append(label)
                with instance.tf_mode(), label.tf_mode():
                    storage['tf_result'] = tf_method(instance, *storage['tf_args'])
                storage['session'] = tf.Session()
                storage['session'].run(tf.initialize_all_variables(), feed_dict=instance.get_feed_dict())
            feed_dict = dict(zip(storage['tf_args'], np_args))
            feed_dict[storage['free_vars']] = instance.get_free_state()
            feed_dict.update(instance.get_feed_dict())
            return storage['session'].run(storage['tf_result'], feed_dict=feed_dict)

        return runnable
        