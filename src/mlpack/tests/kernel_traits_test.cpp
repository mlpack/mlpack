/**
 * @file kernel_traits_test.cpp
 * @author Ryan Curtin
 *
 * Test the KernelTraits class.  Because all of the values are known at compile
 * time, this test is meant to ensure that uses of KernelTraits still compile
 * okay and react as expected.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::kernel;

BOOST_AUTO_TEST_SUITE(KernelTraitsTest);

BOOST_AUTO_TEST_CASE(IsNormalizedTest)
{
  // Reason number ten billion why macros are bad:
  //
  // The Boost unit test framework is built on macros.  When I write
  // BOOST_REQUIRE_EQUAL(KernelTraits<int>::IsNormalized, false), what actually
  // happens (in gcc at least) is that the 'false' gets implicitly converted to
  // an int; then, the compiler goes looking for an int IsNormalized variable in
  // KernelTraits.  But this doesn't exist, so we get this error at linker time:
  //
  // kernel_traits_test.cpp:(.text+0xb86): undefined reference to
  // `mlpack::kernel::KernelTraits<mlpack::kernel::LinearKernel>::IsNormalized'
  //
  // and this actually tells us nothing about the error.  When you dig deep
  // enough or get frustrated enough, you end up realizing it's a macro problem
  // and now have to *explicitly* cast to bool.  Yes, really; try it if you
  // don't believe me.

  // Test each kernel individually.
  // If the type is not a valid kernel, it should be false (default value).
  BOOST_REQUIRE_EQUAL((bool) KernelTraits<int>::IsNormalized, false);

  // Normalized kernels.
  BOOST_REQUIRE_EQUAL((bool) KernelTraits<CosineDistance>::IsNormalized, true);
  BOOST_REQUIRE_EQUAL((bool) KernelTraits<EpanechnikovKernel>::IsNormalized,
      true);
  BOOST_REQUIRE_EQUAL((bool) KernelTraits<GaussianKernel>::IsNormalized, true);
  BOOST_REQUIRE_EQUAL((bool) KernelTraits<LaplacianKernel>::IsNormalized, true);
  BOOST_REQUIRE_EQUAL((bool) KernelTraits<SphericalKernel>::IsNormalized, true);
  BOOST_REQUIRE_EQUAL((bool) KernelTraits<TriangularKernel>::IsNormalized,
      true);

  // Unnormalized kernels.
  BOOST_REQUIRE_EQUAL((bool) KernelTraits<LinearKernel>::IsNormalized, false);
  BOOST_REQUIRE_EQUAL((bool) KernelTraits<PolynomialKernel>::IsNormalized,
      false);
  BOOST_REQUIRE_EQUAL((bool) KernelTraits<PSpectrumStringKernel>::IsNormalized,
      false);
}

BOOST_AUTO_TEST_SUITE_END();
