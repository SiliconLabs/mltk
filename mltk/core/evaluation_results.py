
import logging


class EvaluationResults(dict):
    """Holds model evaluation results
    
    .. note:: The Implementation details are specific to the model type

    .. seealso::

       - :py:class:`mltk.core.ClassifierEvaluationResults` 
       - :py:class:`mltk.core.AutoEncoderEvaluationResults` 

    """
    def __init__(self, name:str, model_type:str='generic', **kwargs):
        super().__init__()
        self['name'] = name 
        self['model_type'] = model_type
        self.update(**kwargs)


    @property
    def name(self) -> str:
        """The name of the evaluated model"""
        return self['name']

    @property
    def model_type(self) -> str:
        """The type of the evaluated model (e.g. classification, autoencoder, etc.)"""
        return self['model_type']



    def generate_summary(self, include_all=True) -> str:
        """Generate and return a summary of the results as a string"""
        # This should be implemented by a subclass
        s = f'Name: {self.name}\n'
        s += f'Model Type: {self.model_type}\n'
        if include_all:
            for key, value in self.items():
                if key in ('name', 'model_type'):
                    continue
            s += f'{key}: {value}\n'
        
        return s.strip()


    def generate_plots(
        self, 
        show=True, 
        output_dir:str=None, 
        logger: logging.Logger=None
    ):
        """Generate plots of the evaluation results
        
        Args:
            show: Display the generated plots
            output_dir: Generate the plots at the specified directory. If omitted, generated in the model's logging directory
            logger: Optional logger
        """
        # This should be implemented by a subclass
        raise NotImplementedError


    def __str__(self) -> str:
        return self.generate_summary()

