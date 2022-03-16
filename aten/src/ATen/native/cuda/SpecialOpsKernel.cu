#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/Math.h>

namespace at {
namespace native {

void hyp2f1_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "hyp2f1_cuda", [&]() {
    gpu_kerenel(iter, []GPU_LAMBDA(scalar_t self, scalar_t a, scalar_b, scalar_t c) -> scalar_t {
      return hyp2f1(self, a, b, c);
     });
  });
};

REGISTER_DISPATCH(hyp2f1_stub, &hyp2f1_kernel_cuda);

} // namespace native
} // namespace at
