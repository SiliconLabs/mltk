

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "microfrontend/lib/frontend.h"

namespace py = pybind11;

namespace mltk
{


class AudioFeatureGeneratorWrapper
{
public:
    AudioFeatureGeneratorWrapper(const py::dict& settings);
    ~AudioFeatureGeneratorWrapper();
    void process_sample(const py::array_t<int16_t>& input, py::array& output);
    bool activity_was_detected();

    FrontendState _frontend_state;
    int _sample_length;
    int _n_channels;
    int _n_features;
    int _window_step;
    int _window_size;
    int _dynamic_quantize_range;
};


} // namespace mltk