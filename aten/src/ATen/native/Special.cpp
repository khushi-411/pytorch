#include <ATen/native/Special.h>

namespace at {

namespace meta {

TORCH_META_FUNC(special_hyp2f1)(
    const Tensor& self, const Tensor& a, const Tensor& b, const Tensor& c) {
  build(
    at::TensorIteratorConfig()
    .set_check_mem_overlap(true)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .promote_integer_inputs_to_float(true)
    .add_borrowed_output(maybe_get_output())  // add_borrowed_output()
    .add_borrowed_input(self)
    .add_borrowed_input(a)
    .add_borrowed_input(b)
    .add_borrowed_input(c)
  );
}

} // namespace meta

namespace native {

TORCH_IMPL_FUNC(special_hyp2f1_out)
(const Tensor& self, const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& result) {
    special_hyp2f1_stub(device_type(), *this);
}

// TODO
Tensor special_hyp2f1(const Scalar& self, const Tensor& a, const Tensor& b, const Tensor& c) {
  return at::special_hyp2f1(wrapped_scalar_tensor(self), a, b, c);
}

Tensor special_hyp2f1(const Tensor& self, const Scalar& a, const Tensor& b, const Tensor& c) {
  return at::special_hyp2f1(self, wrapped_scalar_tensor(a), b, c);
}

Tensor special_hyp2f1(const Tensor& self, const Tensor& a, const Scalar& b, const Tensor& c) {
  return at::special_hyp2f1(self, a, wrapped_scalar_tensor(b), c);
}

Tensor special_hyp2f1(const Tensor& self, const Tensor& a, const Tensor& b, const Scalar& c) {
  return at::special_hyp2f1(self, a, b, wrapped_scalar_tensor(c));
}

Tensor special_hyp2f1(const Tensor& self, const Scalar& a, const Scalar& b, const Tensor& c) {
  return at::special_hyp2f1(self, wrapped_scalar_tensor(a), wrapped_scalar_tensor(b), c);
}

Tensor special_hyp2f1(const Tensor& self, const Tensor& a, const Scalar& b, const Scalar& c) {
  return at::special_hyp2f1(self, a, wrapped_scalar_tensor(b), wrapped_scalar_tensor(c));
}

Tensor special_hyp2f1(const Tensor& self, const Scalar& a, const Tensor& b, const Scalar& c) {
  return at::special_hyp2f1(self, wrapped_scalar_tensor(a), b, wrapped_scalar_tensor(c));
}

// TODO: why & ?
Tensor& special_hyp2f1_out(const Tensor& self, const Scalar& a, const Tensor& b, const Tensor& c, Tensor& result) {
  return at::special_hyp2f1(self, wrapped_scalar_tensor(a), b, c);
}

Tensor& special_hyp2f1_out(const Scalar& self, const Tensor& a, const Tensor& b, const Tensor& c, Tensor& result) {
  return at::special_hyp2f1(wrapped_scalar_tensor(self), a, b, c);
}

Tensor& special_hyp2f1_out(const Tensor& self, const Tensor& a, const Scalar& b, const Tensor& c, Tensor& result) {
  return at::special_hyp2f1(self, a, wrapped_scalar_tensor(b), c);
}

Tensor& special_hyp2f1_out(const Tensor& self, const Tensor& a, const Tensor& b, const Scalar& c, Tensor& result) {
  return at::special_hyp2f1(self, a, b, wrapped_scalar_tensor(c));
}

Tensor& special_hyp2f1_out(const Tensor& self, const Scalar& a, const Scalar& b, const Tensor& c, Tensor& result) {
  return at::special_hyp2f1(self, wrapped_scalar_tensor(a), wrapped_scalar_tensor(b), c);
}

Tensor& special_hyp2f1_out(const Tensor& self, const Tensor& a, const Scalar& b, const Scalar& c, Tensor& result) {
  return at::special_hyp2f1(self, a, wrapped_scalar_tensor(b), wrapped_scalar_tensor(c));
}

Tensor& special_hyp2f1_out(const Tensor& self, const Scalar& a, const Tensor& b, const Scalar& c, Tensor& result) {
  return at::special_hyp2f1(self, wrapped_scalar_tensor(a), b, wrapped_scalar_tensor(c));
}

DEFINE_DIPATCH(special_hyp2f1_stub);

} // namespace native
} // namespace at
