#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Special.h>

#include <cmath>
#include <iostream>

#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Math.h>

namespace at {
namespace native {

namespace {

// TODO: error: ‘cpu_kernel’ was not declared in this scope
// try with at::native::cpu_kernel -------- not working
void hyp2f1_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "hyp2f1_cpu", [&]() {
    // TODO: Why to declare as Scalar?: Used to create python binding
    // TODO: [] means?
    cpu_kernel(iter, [](scalar_t self, scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return calc_hyp2f1(self, a, b, c);  // call for gauss hypergeometric function availabe at Math.h
    });
  });
}

} // anonymous namespace

// TODO: Why pass kernel by address?
REGISTER_DISPATCH(special_hyp2f1_stub, &hyp2f1_kernel);

} // namespace native
} // namespace at
