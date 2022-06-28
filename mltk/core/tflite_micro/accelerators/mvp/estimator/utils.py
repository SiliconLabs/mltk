
import gzip
import os
import numpy as np
from urllib.parse import urlparse
import yaml
import onnx
import onnxruntime
import onnxruntime.backend as backend

from mltk.core.utils import get_mltk_logger
from mltk.utils.archive_downloader import download_verify_extract


onnxruntime.set_default_logger_severity(3)




class _DataList(list):
    def __init__(self, params):
        super().__init__()
        self.columns = []
        row = []
        for key in sorted(params):
            if key == 'self':
                continue
            row.append(params[key])
            self.columns.append(key)
        self.append(row)

    def tonumpy(self) -> np.ndarray:
        return np.asarray(self, dtype=np.int64)


class MetricBaseEstimator(object):
    def __init__(self, onnx_model_file):
        onnx_model = onnx.load(onnx_model_file)
        self.onnx_model_backend = backend.prepare(onnx_model, 'CPU')
        meta = onnx_model.metadata_props[0]
        self.feature_names = meta.value.split(',')


    def predict(self, **kwargs):
        X = _DataList(kwargs)  
        y = self.onnx_model_backend.run(X.tonumpy())
        return float(y[0])



def download_estimators() -> str:
    curdir = os.path.dirname(os.path.abspath(__file__))
    logger = get_mltk_logger()

    try:
        from mltk.cli import is_command_active
        show_progress=is_command_active()
    except:
        show_progress = False
    
    with open(f'{curdir}/estimators_url.yaml', 'r') as fp:
        estimators_url_obj = yaml.load(fp, Loader=yaml.SafeLoader)

    parsed_url = urlparse(estimators_url_obj['url'])
    url_name, _ = os.path.splitext(os.path.basename(parsed_url.path))

    # NOTE: If the estimators already exists then this doesn't do anything
    dest_dir = download_verify_extract(
        url=estimators_url_obj['url'],
        dest_subdir=f'accelerators/mvp/estimators/{url_name}',
        show_progress=show_progress,
        file_hash=estimators_url_obj['sha1'],
        file_hash_algorithm='sha1',
        logger=logger
    )

    return dest_dir


def load_model(name:str, accelerator:str, metric:str) -> MetricBaseEstimator:
    logger = get_mltk_logger()

    estimator_name = f'{name}.{accelerator}.{metric}.onnx.gz'

     # First see if the estimator archive exists in the local "generated" directory
    curdir = os.path.dirname(os.path.abspath(__file__))
    generated_dir = f'{curdir}/generated'
    if os.path.exists(f'{generated_dir}/__init__.py'):
        estimator_path = f'{generated_dir}/{estimator_name}'
   
    else:
        # Otherwise we need to download all the estimators
        try:
            estimators_dir = download_estimators()
        except Exception as e:
            logger.warning(f'Failed to download profiling estimators, err: {e}')
            return None

        estimator_path = f'{estimators_dir}/{estimator_name}'

    try:
        if os.path.exists(estimator_path):
            with gzip.open(estimator_path, 'rb') as fp:
                return MetricBaseEstimator(fp)
    except Exception as e:
        logger.warning(f'Failed to load profiling estimator: {estimator_path}, err: {e}')

    return None


def activation_to_int(activation:str) -> int:
    activation = activation.lower()
    if activation == 'none':
        return 0
    elif activation == 'relu':
        return 1
    elif activation == 'relu_n1_to_1':
        return 2 
    elif activation == 'relu6':
        return 3 
    elif activation == 'tanh':
        return 4 
    elif activation == 'sign_bit':
        return 5 
    else:
        return -1


def use_activation(activation:str) -> int:
    return bool_to_int(activation.lower() != 'none')


def padding_to_int(padding:str) -> int:
    padding = padding.lower()
    if padding == 'same':
        return 1
    elif padding == 'valid':
        return 2
    else:
        return 0 

def bool_to_int(value:bool) -> int:
    return 1 if value else 0
