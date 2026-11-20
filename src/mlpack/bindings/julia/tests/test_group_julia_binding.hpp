/**
 * @file bindings/julia/tests/test_group_julia_binding.hpp
 * @author Ryan Curtin
 *
 * Utility "model" class for grouped test binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_TESTS_TEST_GROUP_JULIA_BINDING_HPP
#define MLPACK_BINDINGS_JULIA_TESTS_TEST_GROUP_JULIA_BINDING_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/kernels/cauchy_kernel.hpp>

namespace mlpack {

// Very simple test class type that just holds a Cauchy kernel internally.
class TestGroupJuliaBinding
{
 public:
  // Create the wrapped Cauchy kernel with the given bandwidth.
  TestGroupJuliaBinding(const double bw = 1.0) : ck(bw) { }

  // Get the kernel.
  CauchyKernel& Kernel() { return ck; }

  // Serialize the object.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar(CEREAL_NVP(ck));
  }

 private:
  CauchyKernel ck;
};

} // namespace mlpack

#endif // MLPACK_BINDINGS_JULIA_TESTS_TEST_GROUP_JULIA_BINDING_HPP
