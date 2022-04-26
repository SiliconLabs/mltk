"""

"""

import os
import math
import json
import itertools
from typing import Union, Tuple, List
import random
import logging
import tqdm
from numba import jit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from math import ceil
import scipy.signal as sps
import matplotlib.pyplot as plt
from keras_preprocessing.image.utils import (img_to_array, load_img, array_to_img)
from mltk.core.model import (
    MltkModel,
    TrainMixin,
    ImageDatasetMixin,
    EvaluateMixin
)
from mltk.core import (
    load_tflite_or_keras_model,
    get_mltk_logger,
    KerasModel,
    TfliteModel,
    EvaluationResults,
    ClassifierEvaluationResults,
    TrainingResults
)
from mltk.core.tflite_model.tflite_model import TfliteModel
from mltk.utils.archive_downloader import download_verify_extract
from mltk.utils.hasher import generate_hash
from mltk.core.preprocess.image.parallel_generator import ParallelImageDataGenerator, ParallelProcessParams


# @mltk_model # NOTE: This tag is required for this model be discoverable
class MyModel(
    MltkModel, 
    TrainMixin, 
    ImageDatasetMixin,
    EvaluateMixin
):
    pass
my_model = MyModel()



###############################################################################
# Data Preprocessing
#

@jit
def balance_colorspace(
    x:np.ndarray,
    imax = 0,
    imin = 255,
    border = 32,
    threshold_min = 0,
    threshold_max = 240,
) -> np.ndarray:
    """Simple statistical centering, remove outliers
        Both brightness and space
    """
    height, width = x.shape
    out = np.zeros_like(x)

    for i in range(border, height-border):
        for j in range(border, width-border):
            if x[i,j] < threshold_max:
                imax = max(imax, int(x[i,j]))
            if x[i,j] > threshold_min: 
                imin = min(imin, int(x[i,j]))
  
    for i in range(height):
        for j in range(width):
            val_norm = (255/(imax-imin))*(int(x[i,j])-imin)
            out[i,j] = min(255, max(0, val_norm))
        
    return out


def generate_gaussian_filter(filter_size:int, sigma) -> np.ndarray:
    """Calculate gaussian filter"""
    g = np.zeros((filter_size,filter_size)); #2D filter matrix
    
    #gaussian filter
    for i in range(-(filter_size-1)//2,(filter_size-1)//2+1):
        for j in range(-(filter_size-1)//2,(filter_size-1)//2+1):
            x0 = (filter_size+1)//2; #center
            y0 = (filter_size+1)//2; #center
            x=i+x0; #row
            y=j+y0; #col
            g[y-1,x-1] = np.exp(-((x-x0)**2+(y-y0)**2)/2/sigma/sigma)
    
    # normalize gaussian filter
    sum = np.sum(g)
    g = g/sum
    return g


def generate_sharpening_filter(g_filter:np.ndarray, amount:float)-> Tuple[np.ndarray, int]:
    """harpening filter, original + (original ? blurred) × amount"""
    filter_size = len(g_filter)
    
    original = np.zeros((filter_size,filter_size))
    original[(filter_size-1)//2,(filter_size-1)//2] = 1

    s_filter = original + (original - g_filter) * amount
    
    # Normalize to int8
    gain = 128/s_filter[(len(s_filter)-1)//2,(len(s_filter)-1)//2]
    gain = 2**(np.floor(np.log2(gain)))
    s_filter = np.round(s_filter*gain)
    s_filter[(len(s_filter)-1)//2,(len(s_filter)-1)//2] += gain - np.sum(s_filter)
    
    return s_filter.astype(np.int8), gain.astype(np.int8)


@jit
def conv2d(input:np.ndarray, filter:np.ndarray, gain):
    input_w, input_h = input.shape[1], input.shape[0]      # input width and input height
    filter_w, filter_h = filter.shape[1], filter.shape[0]  # filter width and filter height

    output_h = input_h
    output_w = input_w

    pad_along_height = filter_h - 1
    pad_along_width = filter_w - 1

    pad_height = pad_along_height // 2
    pad_width = pad_along_width // 2

    input = input.flatten()
    filter = filter.flatten()
    output = np.zeros((output_h*output_w,), dtype=np.uint8)  # convolution output
    out_i = 0

    for out_y in range(output_h):
        in_y_origin = out_y - pad_height
        for out_x in range(output_w):
            in_x_origin = out_x - pad_width
            acc = 0 
            filter_i = 0
            for filter_y in range(filter_h):
                in_y = in_y_origin + filter_y
                if in_y < 0 or in_y >= input_h:
                    filter_i += filter_w
                    continue 
                for filter_x in range(filter_w):
                    in_x = in_x_origin + filter_x
                    if in_x < 0 or in_x >= input_w:
                        filter_i += 1
                        continue 

                    acc += input[in_y*input_w + in_x] * filter[filter_i]
                    filter_i += 1

            norm_val = acc // gain

            output[out_i] = min(255, max(0, norm_val))
            out_i += 1

    output = output.reshape((output_h, output_w))
                
    return output

def sharpen_image(image:np.ndarray, filter:np.ndarray, gain:float) -> np.ndarray:
    """ Sharpen image, conv2d followed by saturation"""
    # Pure conv phase
    image_sharp = sps.convolve2d(image.astype(np.int16), filter ,mode='same') // gain
    image_sharp = np.clip(image_sharp, 0, 255)
    image_sharp = image_sharp.astype(np.uint8)

    # image_shape2 = conv2d(image, filter, gain)
    # assert np.allclose(image_sharp, image_shape2)
 
    return image_sharp


_verify_msg = ''
def verify_sample(
    x:np.ndarray,
    imin=32,
    imax=224,
    border=32,
    full_threshold=4,
    center_threshold=3,
) -> bool:
    """Return if the given sample is of poor quality or not"""
    global _verify_msg
    height, width = x.shape
    dark_full = 0
    light_full = 0
    for i in range(height):
        for j in range(width):
            if x[i,j] < imin:
                dark_full += 1
            elif x[i,j] >= imax: 
                light_full += 1
    
    dark_center = 0
    light_center = 0
    for i in range(border, height-border):
        for j in range(border, width-border):
            if x[i,j] < imin:
                dark_center += 1
            elif x[i,j] >= imax: 
                light_center += 1

    _verify_msg = f'dark_full={dark_full} light_full={light_full}\n'
    _verify_msg += f'dark_center={dark_center} light_center={light_center}\n'
    _verify_msg += f'abs(dark_full-light_full)={abs(dark_full-light_full)}\n'
    _verify_msg += f'(dark_full+light_full)/full_threshold={(dark_full+light_full)/full_threshold}\n'
    _verify_msg += f'abs(dark_center-light_center)={abs(dark_center-light_center)}\n'
    _verify_msg += f'(dark_center+light_center)/center_threshold={(dark_center+light_center)/center_threshold}\n'

    if( (abs(dark_full-light_full) > (dark_full+light_full)/full_threshold) or \
        (abs(dark_center-light_center) > (dark_center+light_center)/center_threshold)):
        return False 
    
    return True


###############################################################################
# Dataset generation
#

class MyDataset:
    def __init__(
        self,
        nomatch_multiplier:int=10,
        preprocess_samples=True,
        g_filter_size=5, # approximates radius of 2.5
        g_filter_sigma=8,
        contrast=2.0,
        border=32,
        balance_threshold_max=240,
        balance_threshold_min=0,
        verify_imin=32,
        verify_imax=224,
        verify_full_threshold=3,
        verify_center_threshold=2
    ):
        self.nomatch_multiplier = nomatch_multiplier
        """The number of pairs of non-matching fingerprints to generate for each fingerprint"""

        self.preprocess_sample_enabled = preprocess_samples

        self.preprocess_params = dict(
            g_filter_size = g_filter_size,
            g_filter_sigma = g_filter_sigma,
            contrast = contrast,
            border = border,
            balance_threshold_min = balance_threshold_min,
            balance_threshold_max = balance_threshold_max,
            verify_imin = verify_imin,
            verify_imax = verify_imax,
            verify_full_threshold = verify_full_threshold,
            verify_center_threshold = verify_center_threshold
        )

        g_filter = generate_gaussian_filter(
            self.preprocess_params['g_filter_size'],
            self.preprocess_params['g_filter_sigma']
        )
        self.sharpen_filter, self.sharpen_gain = generate_sharpening_filter(
            g_filter, 
            self.preprocess_params['contrast']
        )
        
    
    def preprocess_sample(self, x:np.ndarray) -> np.ndarray:
        if len(x.shape) == 3:
            x = np.squeeze(x, axis=-1)
        x = balance_colorspace(x, 
            threshold_min=self.preprocess_params['balance_threshold_min'],
            threshold_max=self.preprocess_params['balance_threshold_max'],
            border=self.preprocess_params['border']
        )
        x = sharpen_image(x,
            filter=self.sharpen_filter,
            gain=self.sharpen_gain
        )
        return x


    def verify_sample(self, x:np.ndarray) -> bool:
        return verify_sample(
            x, 
            imin=self.preprocess_params['verify_imin'],
            imax=self.preprocess_params['verify_imax'],
            border=self.preprocess_params['border'],
            full_threshold=self.preprocess_params['verify_full_threshold'],
            center_threshold=self.preprocess_params['verify_center_threshold']
        )


    def load_data(self) -> str:
        """Download and extract the dataset and return the path to the extract directory"""
        DATASET_ARCHIVE_URL = 'https://www.dropbox.com/s/l9hpstr04nvuqs4/silabs_fingerprints_v2.zip?dl=1'
        DATASET_SHA1 = '001B3F7278FE9BA12FFFEF43A22EE53525EC792E'

        unprocessed_dir = download_verify_extract(
            url=DATASET_ARCHIVE_URL,
            dest_subdir='datasets/silabs_fingerprints/v2/unprocessed',
            file_hash=DATASET_SHA1,
            file_hash_algorithm='sha1',
            remove_root_dir=True
        )

        if not self.preprocess_sample_enabled:
            return unprocessed_dir

        processed_base_dir = os.path.dirname(unprocessed_dir).replace('\\', '/') + '/processed'

        params_hash = generate_hash(self.preprocess_params)[:8]
        processed_dir = f'{processed_base_dir}/{params_hash}'
        processed_params_path = f'{processed_dir}/params.json'
        dropped_samples_path = f'{processed_dir}/dropped_samples.txt'

        os.makedirs(processed_dir, exist_ok=True)
        if not os.path.exists(processed_params_path):
            print(f'Generating {processed_dir} ...')
            all_samples = self.list_all_samples(unprocessed_dir, flatten=True)
            dropped_samples = []
            sampled_count = 0
            with tqdm.tqdm(total=len(all_samples), unit='sample', desc='Preprocessing samples') as progbar:
                for fn in all_samples:
                    sample_path = f'{unprocessed_dir}/{fn}'
                    img = load_img(sample_path, color_mode='grayscale')
                    x = img_to_array(img, dtype='uint8')
                    img.close()

                    x_norm = self.preprocess_sample(x)
                    if not self.verify_sample(x_norm):
                        dropped_samples.append(fn)
                        progbar.update()
                        continue

                    dst_path = f'{processed_dir}/{fn}'
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    img = array_to_img(x, scale=False, dtype=np.uint8)
                    img.save(dst_path)
                    sampled_count += 1
                    progbar.update()

            print(f'Generating {dropped_samples_path}')
            with open(dropped_samples_path, 'w') as f:
                for fn in dropped_samples:
                    f.write(fn + '\n')
            
            print(f'Generating {processed_params_path}')
            with open(processed_params_path, 'w') as f:
                json.dump(self.preprocess_params, f, indent=3)

            print(f'  Total samples dropped: {len(dropped_samples)}')
            print(f'  Total samples used: {sampled_count}')

        return processed_dir


    def list_valid_filenames_in_directory(
        self,
        base_directory:str, 
        search_class:str, 
        white_list_formats:List[str], 
        split:float, 
        follow_links:bool, 
        shuffle_index_directory:str
    ) -> Tuple[str, List[str]]:
        """Return a list of the filenames for the given search class""" 
        if search_class == 'match':
            fp_pairs = self._generate_match_pairs(base_directory)
        else:
            fp_pairs = self._generate_nomatch_pairs(base_directory)
        
        if split:
            num_files = len(fp_pairs)
            if split[0] == 0:
                start = 0
                stop = math.ceil(split[1] * num_files)

                # We want to ensure that we do NOT split
                # on the same person's fingerprints

                # Get the directory name that starts on the other side of the split
                other_subset = os.path.dirname(fp_pairs[stop][0])

                # Keep moving the 'stop' index back while
                # it points to the same person's fingerprints
                while stop > 0 and fp_pairs[stop-1][0].startswith(other_subset):
                    stop -= -1

            else:
                start = math.ceil(split[0] * num_files)
                stop = num_files

                other_subset = os.path.dirname(fp_pairs[start][0])
                while start > 0 and fp_pairs[start-1][0].startswith(other_subset):
                    start -= 1

                
            fp_pairs = fp_pairs[start:stop] 
        
        return search_class, fp_pairs


    def list_all_samples(self, base_directory:str, flatten:bool):
        """Return a list of all the samples in the dataset"""
        retval = []
        for dn in os.listdir(base_directory):
            dir_path = f'{base_directory}/{dn}'
            if not os.path.isdir(dir_path):
                continue
            filenames = []
            for fn in os.listdir(dir_path):
                if not fn.startswith('raw_') or not fn.endswith('.jpg'):
                    continue

                sample_base_path = f'{dn}/{fn}'

                if flatten:
                    retval.append(sample_base_path)
                else:
                    filenames.append(sample_base_path)

            if filenames:
                retval.append(filenames)
        
        return retval
    

    def _generate_match_pairs(self, base_directory:str):
        """Generate a list of all possible matching fingerprint pairs"""
        index_path = f'{base_directory}/.index/match.txt'

        fp_pairs = []
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                for line in f:
                    fp_pairs.append(tuple(line.strip().split(',')))
            return fp_pairs

        sample_batches = self.list_all_samples(base_directory, flatten=False)
        for sample_batch in sample_batches:
            for perm in itertools.permutations(sample_batch, 2):
                fp_pairs.append(perm)

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, 'w') as f:
            for p in fp_pairs:
                f.write(','.join(p) + '\n')

        return fp_pairs


    def _generate_nomatch_pairs(self, base_directory:str):
        """Generate a list of pairs of each fp and a randomly chosen non-matching fp"""
        index_path = f'{base_directory}/.index/nomatch-{self.nomatch_multiplier}.txt'

        fp_pairs = []
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                for line in f:
                    fp_pairs.append(tuple(line.strip().split(',')))
            return fp_pairs

        all_samples = self.list_all_samples(base_directory, flatten=True)

        for sample in all_samples:
            sample_dir = os.path.dirname(sample)
            for _ in range(self.nomatch_multiplier):
                while True:
                    nonmatch_sample = random.choice(all_samples)
                    # Ensure the nomatch_sample is not in the current sample's batch
                    if not nonmatch_sample.startswith(sample_dir):
                        break 
                fp_pairs.append((sample, nonmatch_sample))


        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, 'w') as f:
            for p in fp_pairs:
                f.write(','.join(p) + '\n')

        return fp_pairs



###############################################################################
# Model Loss Function
#


class ContrastiveLoss(keras.losses.Loss):
    def __init__(
        self, 
        margin=1, 
        reduction=keras.losses.Reduction.AUTO,
        name='contrastive_loss'
    ):
        """Calculates the contrastive loss.

        Contrastive loss = mean( (1-true_value) * square(prediction) +
        true_value * square( max(margin-prediction, 0) ))

        Arguments:

        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

        Returns:
            A tensor containing contrastive loss as floating point value.
        """
        super(ContrastiveLoss, self).__init__(
            name=name,
            reduction=reduction
        )
        self.margin = margin

    def call(self, y_true, y_pred):
        """
        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.
        """
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(self.margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    def get_config(self):
        """Returns the config dictionary for a `Loss` instance."""
        config =  super(ContrastiveLoss, self).get_config()
        config['margin'] = self.margin
        return config


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Euclidean distance = sqrt(sum(square(t1-t2)))

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x = vects[0]
    y = vects[1]
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


###############################################################################
# Model Builder
#


def my_model_builder(my_model: MyModel) -> KerasModel:
    """Build the siamese network model """
    input_1 = layers.Input(my_model.input_shape)
    input_2 = layers.Input(my_model.input_shape)

    # As mentioned above, Siamese Network share weights between
    # tower networks (sister networks). To allow this, we will use
    # same embedding network for both tower networks.
    embedding_network = build_model_tower(my_model)
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)

    output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
    keras_model = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

    keras_model.compile(
        loss=my_model.loss, 
        optimizer=my_model.optimizer, 
        metrics=my_model.metrics
    )
    return keras_model




###############################################################################
# Model Saver
#


def my_keras_model_saver(
    mltk_model:MyModel,
    keras_model:KerasModel,
    logger:logging.Logger
) -> KerasModel:
    """This is invoked after training successfully completes
    
    Here want to just save one of the "towers"
    as that is what is used to generate the fingerprint signature
    on the device
    """
    # The given keras_model contains the full siamese network
    # Save it to the model's log dir
    h5_path = mltk_model.h5_log_dir_path
    siamese_network_h5_path = h5_path[:-len('.h5')] + '.siamese.h5'
    logger.debug(f'Saving {siamese_network_h5_path}')
    keras_model.save(siamese_network_h5_path, save_format='tf')

    # Extract the embedding network from the siamese network
    embedding_network = None
    for layer in keras_model.layers:
        if layer.name == 'model':
            embedding_network = layer
            break
    if embedding_network is None:
        raise RuntimeError('Failed to find embedding model in siamese network model, does the embedding model have the name "model" ?')

    # Save the tower as the .h5 model file for this model
    logger.debug(f'Saving {h5_path}')
    embedding_network.save(h5_path, save_format='tf')

    # Return the keras model
    return embedding_network



###############################################################################
# Model Evaluation
#

def generate_predictions(
    mltk_model:MyModel, 
    built_model:Union[KerasModel, TfliteModel],
    threshold:float,
    x=None
) -> Tuple[np.ndarray,np.ndarray]:
    """Generate predictions using the dataset and trained model
    
    A "prediction" is the euclidean distance between two fingerprint images.
    If the distance is less than threshold then the fingerprints are considered
    a match, otherwise they're not matching (i.e. they're not the same finger)
    """
    def _compare_signatures(s1, s2) -> float:
        # Calculate the distance (i.e. similarity)
        # between the two fingerprint signature vectors
        dis = np.sqrt(np.sum(np.square(s1 - s2)))
        # If the distance is less than the threshold
        # then the two signatures are considered a match
        # Normalize the distance to be between 0,1
        # where <0.5 maps to < threshold
        return min((0.5/threshold) * dis, 1.0), dis

    y_dis = []
    y_pred = []
    y_label = []

    desc = '    Generating .h5 predictions' if isinstance(built_model, KerasModel) else 'Generating .tflite predictions'

    # If this a .tflite model
    # then we need to dequantize the model output.
    # The input to the .tflite should have a scaler of 1 and zeropoint of 0
    # (i.e. the model input expects the full int8 range)
    # However, the model output does NOT use the full int8.
    # Thus we need to use the output tensor's scaler and zeropoint to convert to the int8 range.
    # HINT: look at the the .tflite in https://netron.app
    #      and view the quantization params for the input and output tensors.
    # The TfliiteModel.predict() API will automatically do the de-quantization if
    # we force the output dtype to be float32
    kwargs = dict()
    if isinstance(built_model, TfliteModel):
        kwargs['y_dtype'] = np.float32

    if x is not None:
        with tqdm.tqdm(total=len(x), unit='prediction', desc=desc) as progbar:
            for x0, x1 in x:
                # For each fingerprint sample pair
                # generate a "signature"
                s0 = built_model.predict(np.expand_dims(x0, axis=0), **kwargs)[0]
                s1 = built_model.predict(np.expand_dims(x1, axis=0), **kwargs)[0]
                pred, dis = _compare_signatures(s0, s1)
                y_pred.append(pred)
                y_dis.append(dis)
                progbar.update()
    else:
        with tqdm.tqdm(total=mltk_model.x.n, unit='prediction', desc=desc) as progbar:
            for batch_x, batch_y in mltk_model.x:
                batch_x0 = batch_x[0]
                batch_x1 = batch_x[1]
                batch_s0 = built_model.predict(batch_x0, **kwargs)
                batch_s1 = built_model.predict(batch_x1, **kwargs)
                for s0, s1, y in zip(batch_s0, batch_s1, batch_y):
                    pred, dis = _compare_signatures(s0, s1)
                    y_pred.append(pred)
                    y_dis.append(dis)
                    y_label.append(y)
                progbar.update(mltk_model.x.batch_size)

    y_pred = np.asarray(y_pred)
    y_label = np.asarray(y_label)  

    return y_pred, y_label, y_dis


def collect_samples(my_model:MyModel, count:int) -> Tuple[list, list]:
    """Collect the specified number of samples from the dataset"""
    my_model.datagen.debug = True
    my_model.datagen.cores = 1
    my_model.datagen.validation_split = None
    my_model.load_dataset(subset='training')

    if count == -1:
        count = 1e12

    match_samples = []
    nomatch_samples = []
    for batch_x, batch_y in my_model.x:
        if len(match_samples) + len(nomatch_samples) >= count:
            break 
        for x0, x1, y in zip(batch_x[0], batch_x[1], batch_y):
            if y == 0 and len(match_samples) < count/2:
                match_samples.append((x0, x1))
            elif y == 1 and len(nomatch_samples) < count/2:
                nomatch_samples.append((x0, x1))

    my_model.unload_dataset()

    all_x = match_samples + nomatch_samples
    all_y = [0] * len(match_samples) + [1] * len(nomatch_samples)

    return all_x, all_y


def my_model_evaluator(
    mltk_model:MyModel, 
    built_model:Union[KerasModel, TfliteModel],
    eval_dir:str,
    logger:logging.Logger,
    show:bool
) -> EvaluationResults:
    """Custom callback to evaluate the trained model
    
    The model is effectively a classifier, but we need to do
    a special step to compare the signatures in the dataset.
    """
    results = ClassifierEvaluationResults(
        name=mltk_model.name,
        classes=mltk_model.classes
    ) 

    threshold = my_model.model_parameters['threshold']
    logger.error(f'Using model threshold: {threshold}')

    y_pred, y_label, y_dis = generate_predictions( 
        mltk_model,
        built_model,
        threshold
    )

    results.calculate(
        y=y_label,
        y_pred=y_pred,
    )

    results.generate_plots(
        logger=logger, 
        output_dir=eval_dir, 
        show=show
    )

    match_dis = []
    nomatch_dis = []

    for y, dis in zip(y_label, y_dis):
        if y == 0:
            match_dis.append(dis)
        else:
            nomatch_dis.append(dis)

    match_dis = sorted(match_dis)
    match_dis_x = [i for i in range(len(match_dis))]
    nomatch_dis = sorted(nomatch_dis)
    nomatch_dis_x = [i for i in range(len(nomatch_dis))]

    step = (match_dis[-1] - match_dis[0]) / 100
    thresholds = np.arange(match_dis[0], match_dis[-1], step)

    match_acc = []
    nomatch_acc = []

    for thres in thresholds:
        valid_count = sum(x < thres for x in match_dis)
        match_acc.append(valid_count / len(match_dis))
        valid_count = sum(x > thres for x in nomatch_dis)
        nomatch_acc.append(valid_count / len(nomatch_dis))

    fig = plt.figure('Threshold vs Accuracy')

    plt.plot(match_acc, thresholds, label='Match')
    plt.plot(nomatch_acc, thresholds, label='Non-match')

    #plt.ylim([0.0, 0.01])
    plt.legend(loc="lower right")
    plt.xlabel('Accuracy')
    plt.ylabel('Threshold')
    plt.title('Threshold vs Accuracy')
    plt.grid(which='major')

    if eval_dir:
        output_path = f'{eval_dir}/threshold_vs_accuracy.png'
        plt.savefig(output_path)
        logger.info(f'Generated {output_path}')
    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)
    

    fig = plt.figure('Euclidean Distance')

    plt.plot(match_dis_x, match_dis, label='Match')
    plt.plot(nomatch_dis_x, nomatch_dis, label='Non-match')

    plt.legend(loc="lower right")
    plt.xlabel('Index')
    plt.ylabel('Distance')
    plt.title('Euclidean Distance')
    plt.grid(which='major')

    if eval_dir:
        output_path = f'{eval_dir}/eclidean_distance.png'
        plt.savefig(output_path)
        logger.info(f'Generated {output_path}')
    if show:
        plt.show(block=False)
    else:
        fig.clear()
        plt.close(fig)

    return results


#######################################################
# Custom command: datagen_dump
import typer
@my_model.cli.command('datagen_dump')
def datagen_dump_custom_command(
    tflite:bool = typer.Option(False, '--tflite',
        help='Include the trained .tflite model predictions in the displayed results'
    ),
    h5:bool = typer.Option(False, '--h5',
        help='Include the trained .h5 model predictions in the displayed results'
    ),
    count:int = typer.Option(100, '--count',
        help='Number of samples to dump, -1 to dump all'
    ),
    threshold:float = typer.Option(None, '--threshold',
        help='Comparsion threshold. If omitted then use the threshold from the model'
    ),
):
    """Custom command to dump the dataset
    
    \b
    Invoke this command with:
    mltk custom fp_siamese2 datagen_dump
    mltk custom fp_siamese2 datagen_dump --tflite --h5 --count 200
    """
    threshold = threshold or my_model.model_parameters['threshold']


    dump_dir = my_model.create_log_dir('datagen_dump', delete_existing=True)

    x_samples, y_samples = collect_samples(my_model, count=count)

    tflite_y_pred = None
    if tflite:
        tflite_model = load_tflite_or_keras_model(my_model, model_type='tflite')
        tflite_y_pred, _, _ = generate_predictions(
            my_model,
            tflite_model,
            threshold=threshold,
            x=x_samples
        )

    h5_y_pred = None
    if h5:
        keras_model = load_tflite_or_keras_model(my_model, model_type='h5')
        h5_y_pred, _, _ = generate_predictions(
            my_model,
            keras_model,
            threshold=threshold,
            x=x_samples
        )

    with tqdm.tqdm(total=len(x_samples), unit='sample', desc='               Dumping samples') as progbar:
        for i, x in enumerate(x_samples):
            fig = plt.figure(figsize=(4, 2))
            plt.axis('off')

            label = 'match' if y_samples[i] == 0 else 'nomatch'
            title = 'Match:' if y_samples[i] == 0 else 'Non-match:'
            if tflite_y_pred is not None:
                title += f' tflite={tflite_y_pred[i]:.3f},'
            if h5_y_pred is not None:
                title += f' keras={h5_y_pred[i]:.3f},'
            plt.title(title[:-1])

            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(x[0], cmap="gray")
            ax.axis('off')
           
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(x[1], cmap="gray")
            ax.axis('off')
           
            fig.tight_layout()
            plt.savefig(f'{dump_dir}/{label}-{i}.png')
            plt.close(fig)
            progbar.update()


    print(f'Images dumped to {dump_dir}')
   


#######################################################
# Custom command: compare_preprocess
import typer
@my_model.cli.command('compare_preprocess')
def datagen_dump_custom_command(
    count:int = typer.Option(100, '--count',
        help='Number of samples to dump, -1 to use all images'
    ),
):
    """Compare raw samples vs preprocessed samples
    
    \b
    Invoke this command with:
    mltk custom fp_siamese2 compare_preprocess
    mltk custom fp_siamese2 compare_preprocess --count 200
    """

    dump_dir = my_model.create_log_dir('compare_preprocess', delete_existing=True)
    dataset = MyDataset(
       preprocess_samples=False
    )
    unprocessed_dir = dataset.load_data()
    all_samples = dataset.list_all_samples(unprocessed_dir, flatten=True)
    if count == -1:
        count = len(all_samples)
    all_samples = all_samples[:min(count, len(all_samples))]


    with tqdm.tqdm(total=len(all_samples), unit='sample', desc='Comparing samples') as progbar:
        for fn in all_samples:
            img = load_img(f'{unprocessed_dir}/{fn}', color_mode='grayscale')
            unprocessed_img = img_to_array(img, dtype='uint8')
            unprocessed_img = np.squeeze(unprocessed_img, axis=-1)
            img.close() 

            processed_img = dataset.preprocess_sample(unprocessed_img)

            img_valid = dataset.verify_sample(processed_img)

            fig = plt.figure(figsize=(4, 4))
            plt.axis('off')

            name = os.path.basename(fn)
            plt.title(name)

            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(unprocessed_img, cmap="gray")
            ax.axis('off')
           
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(processed_img, cmap="gray")
            ax.axis('off')
           
            #fig.tight_layout()
            fig.text(.1, 0, _verify_msg)
            plt.savefig(f'{dump_dir}/{"" if img_valid else "droppped-"}{name}')
            plt.close(fig)
            progbar.update()


    print(f'Images dumped to {dump_dir}')
   



################################################################
# Model Specification



def build_model_tower(my_model: MyModel):
    """Build the one of the "tower's" of the siamese network
    
    This is the ML model that is deployed to the device.
    It takes a fingerprint grayscale image as an input
    and returns a (hopefully) unique signature of the image.
    """
    input = layers.Input(my_model.input_shape)

    #x = layers.Rescaling(scale=1. / 127.5, offset=-128)(input)
    x = layers.Conv2D(8, (5, 5), strides=(2,2))(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(16, (3, 3), strides=(1,1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # x = layers.Conv2D(24, (3, 3), strides=(2,2))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)

    #x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    #x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    # 
    # x = layers.Flatten()(x)
    # x = layers.Dropout(.3)(x)

    # x = tf.keras.layers.BatchNormalization()(x)
    # x = layers.Dense(256)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    x = layers.Dense(16)(x) # This determines how long the fingerprint signature vector is

    return keras.Model(input, x)


my_model.version = 1
my_model.description = 'Fingerprint similarity estimation using a Siamese Network with a contrastive loss'
my_model.epochs = 100
my_model.batch_size = 16
my_model.loss = ContrastiveLoss(margin=1.0)
my_model.metrics = ['accuracy']
my_model.optimizer = 'adam'
my_model.reduce_lr_on_plateau = dict(
  monitor='loss',
  factor = 0.95,
  patience = 1,
  min_delta=1e-7,
  verbose=1
)

# https://keras.io/api/callbacks/early_stopping/
# If the validation accuracy doesn't improve after 'patience' epochs then stop training
# my_model.early_stopping = dict( 
#   monitor = 'val_accuracy',
#   patience = 15,
#   verbose=1,
#   min_delta=1e-3,
#   restore_best_weights=True
# )



my_model.build_model_function = my_model_builder
my_model.on_save_keras_model = my_keras_model_saver
my_model.eval_custom_function = my_model_evaluator

# For every fingerprint in the datset, 
# generate this many non-matching pairs
nomatch_multiplier = 5
my_model.dataset = MyDataset(nomatch_multiplier=nomatch_multiplier)
my_model.class_mode = 'binary' # we have a signal sigmoid output, so must use a binary "class mode"
my_model.classes = ['match', 'no-match']
my_model.input_shape = (180, 180, 1) # We manually crop the image in convert_img_from_uint8_to_int8()
my_model.target_size = (192, 192, 1) # Ths is the size of the images in the dataset, 
                                     # We use the native image size to do all augmentations
                                     # Then in the preprocessing_function() callback we crop the image border
my_model.class_weights = 'balanced'



# The maximum "distance" between two signature vectors to be considered
# the same fingerprint
# Refer to the <model log dir>/eval/h5/threshold_vs_accuracy.png
# to get an idea of what this valid should be
my_model.model_parameters['threshold'] = 0.18

# Also add the preprocessing settings to the model parameters
preprocess_params = my_model.dataset.preprocess_params
my_model.model_parameters['sharpen_filter'] = my_model.dataset.sharpen_filter.flatten().tobytes()
my_model.model_parameters['sharpen_filter_width'] = my_model.dataset.sharpen_filter.shape[1]
my_model.model_parameters['sharpen_filter_height'] = my_model.dataset.sharpen_filter.shape[0]
my_model.model_parameters['sharpen_gain'] = my_model.dataset.sharpen_gain
my_model.model_parameters['balance_threshold_max'] = preprocess_params['balance_threshold_max']
my_model.model_parameters['balance_threshold_min'] = preprocess_params['balance_threshold_min']
my_model.model_parameters['border'] = preprocess_params['border']
my_model.model_parameters['verify_imin'] = preprocess_params['verify_imin']
my_model.model_parameters['verify_imax'] = preprocess_params['verify_imax']
my_model.model_parameters['verify_full_threshold'] = preprocess_params['verify_full_threshold']
my_model.model_parameters['verify_center_threshold'] = preprocess_params['verify_center_threshold']




def convert_img_from_uint8_to_int8(params:ParallelProcessParams, x:np.ndarray) -> np.ndarray:
    # x is a float32 dtype but has an uint8 range
    x = np.clip(x, 0, 255) # The data should already been in the uint8 range, but clip it just to be sure
    x = x - 128 # Convert from uint8 to int8
    x = x.astype(np.int8)
    # Crop 6 pixels from the image border
    x = x[6:192-6, 6:192-6]
    return np.nan_to_num(x)


my_model.datagen = ParallelImageDataGenerator(
    cores=0.65,
    debug=False,
    dtype=np.float32, # NOTE: The dtype is float32 but the range is int8,
    max_batches_pending=48, 
    validation_split= 0.1,
    validation_augmentation_enabled=True,
    preprocessing_function=convert_img_from_uint8_to_int8,
    #save_to_dir=my_model.create_log_dir('datagen_dump', delete_existing=True),
    rotation_range=10,
    width_shift_range=15,
    height_shift_range=15,
    #brightness_range=(0.50, 1.70),
    #contrast_range=(0.50, 1.70),
    fill_mode='constant',
    cval=0xff,
    noise=['gauss', 'poisson', 's&p'],
    #zoom_range=(0.95, 1.05),
    # samplewise_center=True,
    # samplewise_std_normalization=True,
    rescale=None,
    horizontal_flip=False,
    vertical_flip=False
)


# We need to save reference to the custom loss function
# so that we can load the .h5 file
# See https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object
my_model.keras_custom_objects['ContrastiveLoss'] = ContrastiveLoss
my_model.keras_custom_objects['euclidean_distance'] = euclidean_distance



##################################
# TF-Lite converter settings

def my_representative_dataset_generator():
    # The data generator returns tuples of fingerprints
    # which is what is required to train the siamese network.
    # However, in my_keras_model_saver() we only save one of the "towers"
    # of the network (which is used to convert the fingerprint into a signature).
    # As such, to quantize the model we need to return a list of fingerprints (not tuples)
    for i, batch in enumerate(my_model.validation_data):
        batch_x, batch_y, _ = keras.utils.unpack_x_y_sample_weight(batch)
        batch_x0 = batch_x[0]
        batch_x1 = batch_x[1]
        # The TF-Lite converter expects 1 sample batches
        for x0 in batch_x0:
            yield [np.expand_dims(x0, axis=0)]
        for x1 in batch_x1:
            yield [np.expand_dims(x1, axis=0)]
        if i > 50:
            break


my_model.tflite_converter = dict( 
    optimizations=[tf.lite.Optimize.DEFAULT],
    supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
    inference_input_type=tf.int8,
    inference_output_type=tf.int8,
    representative_dataset=my_representative_dataset_generator,
    experimental_new_converter =False,
    experimental_new_quantizer =False
)


