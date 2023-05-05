#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "hashmap.h"
#include "sampler.h"
#include "sparsedense_grid.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<HashMap> hashmap(m, "HashMap");
    hashmap.def(py::init<int, int, at::Tensor &, c10::Device &>());
    hashmap.def("load_states", &HashMap::load_states);

    hashmap.def("find", &HashMap::find);

    hashmap.def("insert_keys", &HashMap::insert_keys);
    hashmap.def("insert", &HashMap::insert);

    hashmap.def("erase", &HashMap::erase);
    hashmap.def("clear", &HashMap::clear);

    hashmap.def("items", &HashMap::items);
    hashmap.def("size", &HashMap::size);
    hashmap.def("device", &HashMap::device);

    m.def("query_forward", &query_forward, "Query forward");
    m.def("query_backward_forward", &query_backward_forward,
          "Query backward forward");
    m.def("query_backward_backward", &query_backward_backward,
          "Query backward backward");
    m.def("isosurface_extraction", &isosurface_extraction,
          "Isosurface extraction");
    m.def("marching_cubes", &marching_cubes, "Marching cubes");

    m.def("ray_sample", &ray_sample, "Ray sample");
}
