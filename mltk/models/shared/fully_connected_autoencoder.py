
from functools  import reduce
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Dense, Activation, Flatten, BatchNormalization


def FullyConnectedAutoEncoder(
    input_shape:tuple=(5,128,1), # Default parameters (see ToyADMOS paper: https://arxiv.org/abs/1908.03299)
    dense_units:int=128, 
    latent_units:int=8
) -> Model:
    """Fully Connected Auto-encoder

    .. seealso::
       * http://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds
       * https://github.com/y-kawagu/dcase2020_task2_baseline
       * https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/ToyADMOS_FC_AE/toyadmos_autoencoder_eembc.py
    """


    # Input layer
    input_img = Input(shape=input_shape)
    
    # Flatten input image
    x = Flatten()(input_img)

    # First encoder layer
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    
    # Second encoder layer
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    
    # Third encoder layer
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    
    # Fourth encoder layer
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    
    # Latent layer
    x = Dense(latent_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 

    # First decoder layer
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 

    # Second decoder layer
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 

    # Third decoder layer
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 

    # Fourth decoder layer
    x = Dense(dense_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 

    # Output layer
    output_units = reduce(lambda x,y: x*y, input_shape)
    x = Dense(output_units)(x)
    decoded = Reshape(input_shape)(x)

    # Build model
    autoencoder = Model(input_img, decoded)
    return autoencoder