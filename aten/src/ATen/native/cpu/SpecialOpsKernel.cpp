#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/SpecialOps.h>

#include <cmath>
#include <iostream>

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Math.h>

namespace at {
namespace native {

namespace {

void hyp2f1_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "hyp2f1_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t self, scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return calc_hyp2f1(self, a, b, c);
    });
  });
}

} // namespace

REGISTER_DISPATCH(hyp2f1_stub, &hyp2f1_kernel);

} // namespace native
} // namespace at
