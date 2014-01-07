/**
 * @file kernel_traits_test.cpp
 * @author Ryan Curtin
 *
 * Test the KernelTraits class.  Because all of the values are known at compile
 * time, this test is meant to ensure that uses of KernelTraits still compile
 * okay and react as expected.
 *
 * This file is part of MLPACK 1.0.8.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

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
