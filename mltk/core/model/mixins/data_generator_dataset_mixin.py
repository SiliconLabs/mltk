import copy
from typing import List

import numpy as np

from .dataset_mixin import DatasetMixin

from ..model_attributes import MltkModelAttributesDecorator





class DataGeneratorContext(object):
    """Loaded data generator context"""

    def __init__(
        self, 
        classes: List[str], 
        class_mode:str,
        subset:str,
        train_datagen, 
        train_labels, 
        validation_datagen, 
        validation_labels
    ):
        self.classes = copy.deepcopy(classes)
        """List of string labels for each class in dataset"""
        self.class_mode = class_mode
        """How the label data if formatted"""

        self.subset = subset
        """Data subset"""

        self.train_datagen = train_datagen
        """Training data generator"""
        self.train_labels = train_labels
        """Array or list of "ground-truth" label for each training sample"""

        self.validation_datagen = validation_datagen
        """Validation data generator"""
        self.validation_labels = validation_labels
        """Array or list of "ground-truth" label for each validation sample"""

        self.evaluation_datagen =  train_datagen if validation_datagen is None else validation_datagen
        """Evaluation data generator which is the same as validation_datagen is provided else train_datagen"""
        self.evaluation_labels = train_labels if validation_labels is None else  validation_labels
        """Array or list of "ground-truth" label for each evaluation sample"""


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
            s += self._get_datagen_summary('Training', self.train_labels)

        if self.subset == 'training' or self.subset == 'validation':
            s += self._get_datagen_summary('Validation', self.validation_labels)

        if self.subset == 'evaluation':
            s += self._get_datagen_summary('Evaluation', self.evaluation_labels)

        return s.strip()


    def _get_datagen_summary(self, name, labels):
        if self.class_mode == 'categorical':
            _, class_counts = np.unique(labels, return_counts=True)
            s = f'{name.title()} dataset: Found {len(labels)} samples belonging to {len(self.classes)} classes:\n'
            for i, count in enumerate(class_counts):
                class_label = self.classes[i]
                s += f'{class_label.rjust(10, " ")} = {count}\n'
        else:
            s = f'{name.title()} dataset: Found {len(labels)} samples belonging to {len(self.classes)} classes\n'

        return s




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