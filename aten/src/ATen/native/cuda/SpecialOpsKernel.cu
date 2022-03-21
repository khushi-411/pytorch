#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/Special.h>

#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>

#include <ATen/Math.cuh>

namespace at {
namespace native {

namespace {

void hyp2f1_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hyp2f1_cuda", [&]() {
    gpu_kerenel(iter, []GPU_LAMBDA(scalar_t self, scalar_t a, scalar_b, scalar_t c) -> scalar_t {
      return calc_hyp2f1(self, a, b, c);
     });
  });
};

} // anonymous namespace

REGISTER_DISPATCH(special_hyp2f1_stub, &hyp2f1_kernel_cuda);

} // namespace native
} // namespace at
