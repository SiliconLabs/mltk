from __future__ import annotations
import copy
from typing import List, Dict

import numpy as np

from .dataset_mixin import DatasetMixin

from ..model_attributes import MltkModelAttributesDecorator




@MltkModelAttributesDecorator()
class DataGeneratorDatasetMixin(DatasetMixin):
    """Provides generic data generator properties to the base :py:class:`~MltkModel`
    
    .. seealso::
       - :py:class:`~ImageDatasetMixin`
       - :py:class:`~AudioDatasetMixin`
    """

    @property
    def datagen_context(self) -> DataGeneratorContext:
        """Loaded data generator's context"""
        return self._attributes.get_value('datagen.context', default=None)
    @datagen_context.setter
    def datagen_context(self, v: DataGeneratorContext):
        self._attributes['datagen.context'] = v

    def get_datagen_creator(self, subset: str):
        """Return an object that creates a data generator for the given subset"""
        retval = None 
        # if we want the training datagen or the model doesn't specify a validation_datagen
        if subset == 'training' or not getattr(self, 'validation_datagen'):
            if not hasattr(self, 'datagen') or self.datagen is None:
                raise Exception('Must specify the models datagen field')

            # Then return the datagen
            retval = self.datagen 
        else:
            # Otherwise return the validation_datagen
            retval = self.validation_datagen

        retval = copy.deepcopy(retval)
        if self.test_mode_enabled:
            retval.debug = True
        return retval


    def unload_dataset(self):
        """Unload the dataset"""
        super(DataGeneratorDatasetMixin, self).unload_dataset()
        try:
            self.datagen_context.shutdown()
        except:
            pass 
        self.datagen_context = None


    def summarize_dataset(self) -> str:
        datagen_context = self.datagen_context
        if datagen_context is None:
            return 'No dataset loaded'
        return f'{datagen_context}'
    
    
    def get_shuffle_index_dir(self) -> str:
        """The ParallelImageGenerator and ParallelImageGenerator have the option to shuffle the dataset
        entries once before they're used. The shuffled indices are then saved
        to a file. The saved indices file is added to the generated model archive.
        This function loads the indices file from the archive during evaluation
        and validation.
        
        .. note:: 
           We do NOT want to shuffle during eval/validation so that results are reproducible
           (hence we use the one-time-generated indices file)
        """
        #pylint: disable=no-member
        if self.loaded_subset in ('evaluation', 'validation'):
            try:
                return self.get_archive_dir('dataset') 
            except:
                pass 
        return self.create_log_dir('dataset')


    def _register_attributes(self):
        self._attributes.register('datagen.context', dtype=DataGeneratorContext)






class DataGeneratorContext:
    """Loaded data generator context"""

    def __init__(
        self, 
        subset:str,
        train_datagen, 
        validation_datagen, 
        train_class_counts:Dict[str,int],
        validation_class_counts:Dict[str,int]
    ):
        self.subset = subset
        """Data subset"""

        self.train_datagen = train_datagen
        """Training data generator"""

        self.validation_datagen = validation_datagen
        """Validation data generator"""

        self.evaluation_datagen =  train_datagen if validation_datagen is None else validation_datagen
        """Evaluation data generator which is the same as validation_datagen is provided else train_datagen"""

        self.train_class_counts = train_class_counts
        """Dictionary containing the number of samples for each class in the training subset"""

        self.validation_class_counts = validation_class_counts
        """Dictionary containing the number of samples for each class in the validation subset"""

        self.evaluation_class_counts = validation_class_counts or train_class_counts
        """Dictionary containing the number of samples for each class in the evaluation subset"""


    def shutdown(self):
        """Shutdown the data generators (if necessary)"""
        try:
            self.train_datagen.shutdown()
        except :
            pass
        try:
            self.validation_datagen.shutdown()
        except:
            pass


    def __str__(self):
        """Return a printable summary of the dataset"""

        s = ''
        if self.subset == 'training':
            s += self._get_datagen_summary('Training', self.train_class_counts)

        if self.subset == 'training' or self.subset == 'validation':
            s += self._get_datagen_summary('Validation', self.validation_class_counts)

        if self.subset == 'evaluation':
            s += self._get_datagen_summary('Evaluation', self.evaluation_class_counts)

        return s.strip()


    def _get_datagen_summary(self, name:str, class_counts:Dict[str,int]):
        try:
            if not class_counts:
                return ''

            total_samples = sum(x for x in class_counts.values())
            s = f'{name.title()} dataset: Found {total_samples} samples belonging to {len(class_counts)} classes:\n'
            for class_label,sample_count in class_counts.items():
                s += f'{class_label.rjust(10, " ")} = {sample_count}\n'
        except:
            # If something fails, then silently ignore the error
            s = ''

        return s

