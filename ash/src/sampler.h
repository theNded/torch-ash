#pragma once

#include <torch/extension.h>

#include "hashmap.h"

std::tuple<at::Tensor, at::Tensor> ray_find_near_far(
        const HashMap& hashmap,
        const at::Tensor& ray_origins,
        const at::Tensor& ray_directions,
        const at::Tensor& bbox_min,
        const at::Tensor& bbox_max,
        const float t_min,
        const float t_max,
        const float t_step,
        const float empty_space_step_multiplier);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ray_sample(
        const HashMap& hashmap,
        const at::Tensor& ray_origins,
        const at::Tensor& ray_directions,
        const at::Tensor& ray_nears,
        const at::Tensor& ray_fars,
        const at::Tensor& bbox_min,
        const at::Tensor& bbox_max,
        const int64_t max_samples_per_ray,
        const float empty_space_step_multiplier);
