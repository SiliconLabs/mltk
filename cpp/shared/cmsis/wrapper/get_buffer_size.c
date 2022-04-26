
#include "arm_nn_types.h"

#define ARM_MATH_DSP


int32_t arm_convolve_1x1_s8_fast_get_buffer_size(const cmsis_nn_dims *input_dims)
{
    (void)input_dims;
    return 0;
}


int32_t arm_convolve_1_x_n_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    return (2 * input_dims->c * filter_dims->w * filter_dims->h) * sizeof(int16_t);
#else
    (void)input_dims;
    (void)filter_dims;
    return 0;
#endif
}

int32_t arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if defined(ARM_MATH_MVEI)
    int32_t col_length = input_dims->c * filter_dims->w * filter_dims->h;
    // Get number of complete int16 lanes(multiple of 8) for given col_length. This is dependent on
    // implementation of  arm_nn_mat_mult_s8
    col_length = (col_length + 7) / 8;
    // 4 -> number of im2col buffers, 8 -> 8 elements per Q register
    return 4 * col_length * 8 * (int32_t)sizeof(int8_t);
#elif defined(ARM_MATH_DSP)
    return (2 * input_dims->c * filter_dims->w * filter_dims->h) * (int32_t)sizeof(int16_t);
#else
    (void)input_dims;
    (void)filter_dims;
    return 0;
#endif
}


int32_t arm_convolve_wrapper_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims)
{
    if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) && (input_dims->c % 4 == 0) &&
        (conv_params->stride.w == 1) && (conv_params->stride.h == 1) && (filter_dims->w == 1) && (filter_dims->h == 1))
    {
        return arm_convolve_1x1_s8_fast_get_buffer_size(input_dims);
    }
    else if ((output_dims->h == 1) && (input_dims->h == 1) && (filter_dims->h == 1) && (output_dims->w % 4 == 0) &&
             (input_dims->n == 1))
    {
        return arm_convolve_1_x_n_s8_get_buffer_size(input_dims, filter_dims);
    }
    else
    {
        return arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
    }
}

int32_t arm_depthwise_conv_s8_opt_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if defined(ARM_MATH_MVEI)
    /* The + 4 accounts for out of bounds read of the lhs buffers in the *_nt_t_* functions.  */
    return (2 * input_dims->c * filter_dims->w * filter_dims->h) * (int32_t)sizeof(int16_t) + 4;
#elif defined(ARM_MATH_DSP)
    return (input_dims->c * filter_dims->w * filter_dims->h) * sizeof(int16_t);
#else
    (void)input_dims;
    (void)filter_dims;
    return 0;
#endif
}


int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size(const cmsis_nn_dw_conv_params *dw_conv_params,
                                                      const cmsis_nn_dims *input_dims,
                                                      const cmsis_nn_dims *filter_dims,
                                                      const cmsis_nn_dims *output_dims)
{
    (void)dw_conv_params;
    int32_t size = 0;

    if (input_dims->c == output_dims->c && input_dims->n == 1)
    {
        size = arm_depthwise_conv_s8_opt_get_buffer_size(input_dims, filter_dims);
    }

    return size;
}

int32_t arm_avgpool_s8_get_buffer_size(const int output_x, const int ch_src)
{
    (void)output_x;

#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    return (ch_src * sizeof(int32_t));
#else
    (void)ch_src;
    return 0;
#endif
}