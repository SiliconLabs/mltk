"""Utilities for normalizing data"""
import numpy as np



def normalize(
    x:np.ndarray,
    rescale:float=None,
    samplewise_center=False,
    samplewise_std_normalization=False,
    samplewise_normalize_range=None,
    dtype:np.dtype=None
) -> np.ndarray:
    """Applies the normalization configuration in-place to a batch of inputs.

    Args:
        x: Input sample to normalize
        rescale: ``x *= rescale``
        samplewise_center: ``x -= np.mean(x, keepdims=True)``
        samplewise_std_normalization: ``x /= (np.std(x, keepdims=True) + 1e-6)``
        samplewise_normalize_range: ``x = diff * (x - np.min(x)) / np.ptp(x) + lower``
        dtype: The output dtype, if not dtype if given then x is converted to float32
    Returns:
        The normalized value of x
    """

    # If we're not doing standardization
    if not (rescale or \
            samplewise_center or \
            samplewise_std_normalization or \
            samplewise_normalize_range):
        # Ensure the x's data-type is as expected
        # and convert if it's not
        if x.dtype !=  dtype:
            x = x.astype(dtype)

        # Return the non-standardized x
        return x

    # Otherwise, convert the x to float32 before
    # doing the standardization (if necessary)
    if x.dtype != np.floating:
        x = x.astype(np.float32)

    if rescale:
        x *= rescale
    if samplewise_center:
        x -= np.mean(x, keepdims=True)
    if samplewise_std_normalization:
        x /= (np.std(x, keepdims=True) + 1e-6)
    if samplewise_normalize_range:
        lower = float(samplewise_normalize_range[0])
        upper = float(samplewise_normalize_range[1])
        diff = upper - lower
        x = diff * (x - np.min(x)) / np.ptp(x) + lower

    if dtype is not None:
        x = x.astype(dtype)

    return x

