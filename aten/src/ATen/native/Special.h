#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

// TODO
// TODO: void (*) mean?
using structured_quad_fn = void (*)(
    at::TensorIteratorBase& iter);

DECLARE_DISPATCH(structured_quad_fn, special_hyp2f1_stub);

} // namespace native
} // namespace at
