#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/irange.h>

#include <complex>

namespace at {
namespace native {

namespace {

template<bool inplace>
using Ctype = typename std::conditional<inplace, Tensor&, Tensor>::type;

template<bool inplace>
using ComplexType = typename std::conditional<inplace, c10::complex<float>&, c10::complex<float> >::type;

Tensor make_feature_noise(const Tensor& input) {
  auto input_sizes = input.sizes();
  TORCH_CHECK(input.dim() >= 2, "Feature dropout requires at least 2 dimensions in the input");
  std::vector<int64_t> sizes;
  sizes.reserve(input.dim());
  sizes.push_back(input_sizes[0]);
  sizes.push_back(input_sizes[1]);
  for (const auto i : c10::irange(2, input.dim())) {
    (void)i; //Suppress unused variable warning
    sizes.push_back(1);
  }
  return input.new_empty(sizes);
}

bool is_fused_kernel_acceptable(const Tensor& input, double p) {
  return (input.is_cuda() || input.is_xpu() || input.is_lazy()) && p > 0 && p < 1 && input.sym_numel() > 0;
}

// NB: sure, we could have used different overloads here, but I would feel insecure
// knowing that this dispatch depends only on the constness of the references
template<bool inplace>
Tensor& multiply(Tensor& input, const Tensor& noise) {
  static_assert(inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul_(noise);
}

template<bool inplace>
Tensor multiply(const Tensor& input, const Tensor& noise) {
  static_assert(!inplace, "Wrong multiply overload triggered in Dropout.cpp");
  return input.mul(noise);
}

template<bool inplace>
Tensor& add(Tensor& input1, const Tensor& input2) {
  static_assert(inplace, "Wrong add overload triggered in Dropout.cpp");
  return input1.add_(input2);
}

template<bool inplace>
Tensor add(const Tensor& input1, const Tensor& input2) {
  static_assert(!inplace, "Wrong add overload triggered in Dropout.cpp");
  return input1.add(input2);
}

std::pair<Tensor, Tensor> complex_to_real(const Tensor& inp) {
  auto inp_view_as_complex = at::view_as_real(inp);
  auto dim_i = inp_view_as_complex.dim() - 1;
  auto i_r = inp_view_as_complex.select(dim_i, 0);
  auto i_i = inp_view_as_complex.select(dim_i, 1);
  return std::make_pair(i_r, i_i);
}

template<bool feature_dropout, bool alpha_dropout, bool inplace, typename T>
Ctype<inplace> _complex_dropout_impl(T& input, double p, bool train) {
  TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);

  Tensor i_r, i_i;
  std::tie(i_r, i_i) = complex_to_real(input.resolve_conj());

  if (p == 0 || !train || input.numel() == 0) {
    auto i = c10::Scalar(c10::complex<double>(0, 1));
    return add<inplace>(i_r, i * i_i);
  }

  Tensor k, s;
  if (p == 1) {
    k = multiply<inplace>(i_r, at::zeros({}, i_r.options())); // TODO: options() ???
    s = multiply<inplace>(i_i, at::zeros({}, i_i.options()));
    auto i = c10::Scalar(c10::complex<double>(0, 1));
    return add<inplace>(k, i * s);
  }

  at::Tensor b_r;  // used for alpha dropout only
  auto noise_r = feature_dropout ? make_feature_noise(i_r) : at::empty_like(i_r, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::manual_seed(0); // TODO: Is this correct ???
  noise_r.bernoulli_(1 - p);
  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    b_r = noise_r.add(-1).mul_(alpha * a).add_(alpha * a * p);
   noise_r.mul_(a);
  } else {
    noise_r.div_(1 - p);
  }

  at::Tensor b_i;  // used for alpha dropout only
  auto noise_i = feature_dropout ? make_feature_noise(i_i) : at::empty_like(i_i, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::manual_seed(0);
  noise_i.bernoulli_(1 - p);
  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    b_i = noise_i.add(-1).mul_(alpha * a).add_(alpha * a * p);
    noise_i.mul_(a);
  } else {
    noise_i.div_(1 - p);
  }

  Tensor k2, s2;
  if (!alpha_dropout) {
    k2 = multiply<inplace>(i_r, noise_r);
    s2 = multiply<inplace>(i_i, noise_i);
    auto i = c10::Scalar(c10::complex<double>(0, 1));
    return add<inplace>(k2, i * s2);
  } else {
    k2 = multiply<inplace>(i_r, noise_r).add_(b_r);
    s2 = multiply<inplace>(i_i, noise_i).add_(b_i);
    auto i = c10::Scalar(c10::complex<double>(0, 1));
    return add<inplace>(k2, i * s2);
  }
}

template<bool feature_dropout, bool alpha_dropout, bool inplace, typename T>
Ctype<inplace> _dropout_impl(T& input, double p, bool train) {
  TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
  if (p == 0 || !train || input.numel() == 0) {
    return input;
  }

  if (p == 1) {
    return multiply<inplace>(input, at::zeros({}, input.options()));
  }

  at::Tensor b; // used for alpha_dropout only
  auto noise = feature_dropout ? make_feature_noise(input) : at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  noise.bernoulli_(1 - p);
  if (alpha_dropout) {
    constexpr double alpha = 1.7580993408473766;
    double a = 1. / std::sqrt((alpha * alpha * p + 1) * (1 - p));
    b = noise.add(-1).mul_(alpha * a).add_(alpha * a * p);
    noise.mul_(a);
  } else {
    noise.div_(1 - p);
  }

  if (!alpha_dropout) {
    return multiply<inplace>(input, noise);
  } else {
    return multiply<inplace>(input, noise).add_(b);
  }
}

#define ALIAS_SPECIALIZATION(ALIAS_NAME, IS_FEATURE, IS_ALPHA)                      \
template <bool inplace, typename... Args>                                           \
Ctype<inplace> ALIAS_NAME(Args&&... args) {                                         \
  return _dropout_impl<IS_FEATURE, IS_ALPHA, inplace>(std::forward<Args>(args)...); \
}

ALIAS_SPECIALIZATION(_dropout,               false, false)
ALIAS_SPECIALIZATION(_feature_dropout,       true,  false)
ALIAS_SPECIALIZATION(_alpha_dropout,         false, true )
ALIAS_SPECIALIZATION(_feature_alpha_dropout, true,  true )

#define ALIAS_SPECIALIZATION_COMPLEX(ALIAS_NAME, IS_FEATURE, IS_ALPHA)                      \
template <bool inplace, typename... Args>                                                   \
Ctype<inplace> ALIAS_NAME(Args&&... args) {                                                 \
  return _complex_dropout_impl<IS_FEATURE, IS_ALPHA, inplace>(std::forward<Args>(args)...); \
}

ALIAS_SPECIALIZATION_COMPLEX(_complex_dropout,               false, false)

} // anomymous namepsace

std::tuple<Tensor,Tensor>
native_dropout_cpu(const Tensor& input, double p, c10::optional<bool> train) {
  if (input.numel() == 0) {
    return std::make_tuple(input, at::empty_like(input, input.options()));
  }

  Tensor mask;
  Tensor output;

  if (!train.has_value() || *train) {
    double p1m = 1. - p;
    // Check for probability of zero to avoid divide by zero and NaN results
    double scale = p1m == 0 ? 0. : 1. / p1m;
    mask = at::empty_like(input, input.options().dtype(c10::CppTypeToScalarType<bool>::value));
    mask.bernoulli_(p1m);
    output = input.mul(mask).mul_(scale);
  } else {
    mask = at::ones_like(input, input.options().dtype(c10::CppTypeToScalarType<bool>::value));
    output = input.clone();
  }
  return std::make_tuple(output, mask);
}

Tensor native_dropout_backward(const Tensor& grad, const Tensor& mask, double scale) {
  Tensor result = grad * mask * scale;
  return result;
}

Tensor dropout(const Tensor& input, double p, bool train) {
  auto result = [&]() {
    NoNamesGuard guard;
    // TODO: we can remove this is_nested() code smell in the future
    //       if we find a way to support _dropout for nested tensor
    //       e.g. make it an op (at::_dropout) to use dispatcher?
    if (input.is_nested() || (train && is_fused_kernel_acceptable(input, p))) {
      return std::get<0>(at::native_dropout(input, p, train));
    }
    if (at::isComplexType(input.scalar_type())) {
      return _complex_dropout<true>(input, p, train);
    }
    return _dropout<false>(input, p, train);
  }();
  namedinference::propagate_names(result, input);
  return result;
}

Tensor& dropout_(Tensor& input, double p, bool train) {
  //const Tensor k = input.scalar_type();
  if (at::isComplexType(input.scalar_type())) {
    Tensor i_r, i_i;
    std::tie(i_r, i_i) = complex_to_real(input.resolve_conj());
    auto i = c10::Scalar(c10::complex<double>(0, 1));
    at::manual_seed(0);
    return _dropout<true>(i_r, p, train).add_(i * _dropout<true>(i_i, p, train));
    // return _complex_dropout<true>(input, p, train);
  }
  return _dropout<true>(input, p, train);
}

Tensor feature_dropout(const Tensor& input, double p, bool train) {
  return _feature_dropout<false>(input, p, train);
}

Tensor& feature_dropout_(Tensor& input, double p, bool train) {
  return _feature_dropout<true>(input, p, train);
}

Tensor alpha_dropout(const Tensor& input, double p, bool train) {
  return _alpha_dropout<false>(input, p, train);
}

Tensor& alpha_dropout_(Tensor& input, double p, bool train) {
  return _alpha_dropout<true>(input, p, train);
}

Tensor feature_alpha_dropout(const Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<false>(input, p, train);
}

Tensor& feature_alpha_dropout_(Tensor& input, double p, bool train) {
  return _feature_alpha_dropout<true>(input, p, train);
}

} // namespace native
} // namespace at
