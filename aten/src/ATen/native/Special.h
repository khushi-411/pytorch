#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

using structured_quad_fn = void(*)(TensorIteratorBase);

DECLARE_DISPATCH(structured_quad_fn, special_hyp2f1_stub);

} // namespace native
} // namespace at
