from typing import Tuple, List
from .model.model_utils import KerasModel 



class TrainingResults:
    """Container for the model training results"""
    def __init__(self, mltk_model, keras_model:KerasModel, training_history):
        self.mltk_model = mltk_model
        """The MltkModel uses for training"""
        
        self.keras_model:KerasModel = keras_model
        """The trained KerasModel"""

        self.epochs:List[int] = training_history.epoch 
        """List of integers corresponding to each epoch"""

        self.params:dict = training_history.params 
        """Dictionary of parameters uses for training"""

        self.history = {}
        """Dictionary of metrics recorded for each epoch"""

        for key, value in training_history.history.items():
            if isinstance(value, list):
                self.history[key] = [float(x) for x in value]
            else:
                self.history[key] = value


    @property
    def model_archive_path(self) -> str:
        """File path to model archive which contains the model training output including trained model file"""
        return self.mltk_model.archive_path


    def asdict(self) -> dict:
        """Return the results as a dictionary"""
        return dict(
            epochs=self.epochs,
            params=self.params,
            history=self.history
        )

    def get_best_metric(self) -> Tuple[str, float]:
        """Return the best metric from training
        
        Returns:
            Tuple(Name of metric, best metric value)
        """
        max_val_metrics = ['accuracy']
        min_val_metrics = [
            'mse', 'mean_squared_error', 
            'mae', 'mean_absolute_error', 
            'mape', 'mean_absolute_percentage_error', 
            'msle', 'mean_squared_logarithmic_error'
        ]

        for metric in max_val_metrics:
            val_metric = f'val_{metric}'
            if val_metric in self.history:
                return val_metric, max(self.history[val_metric]) 
            if metric in self.history:
                return metric, max(self.history[metric]) 

        for metric in min_val_metrics:
            val_metric = f'val_{metric}'
            if val_metric in self.history:
                return val_metric, min(self.history[val_metric]) 
            if metric in self.history:
                return metric, min(self.history[metric]) 

        return None, 0


    def __str__(self) -> str:
       name, value = self.get_best_metric()
       return f'Best training {name} = {value}'

