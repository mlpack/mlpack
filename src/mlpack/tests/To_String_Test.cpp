/**
 * @file ToStringTest.cpp
 * @author Ryan Birmingham
 *
 * Test of the AugmentedLagrangian class using the test functions defined in
 * aug_lagrangian_test_functions.hpp.
 **/

#include <mlpack/core.hpp>
#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

#include <mlpack/core/metrics/ip_metric.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/metrics/mahalanobis_distance.hpp>
#include <mlpack/core/kernels/pspectrum_string_kernel.hpp>

#include <mlpack/core/kernels/pspectrum_string_kernel.hpp>

#include <mlpack/core/kernels/example_kernel.hpp>

using namespace mlpack;
using namespace mlpack::kernel;
using namespace mlpack::distribution;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(ToStringTest);

BOOST_AUTO_TEST_CASE(DiscreteDistributionString)
{
  DiscreteDistribution d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(GaussianDistributionString)
{
  GaussianDistribution d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(CosineDistanceString)
{
  CosineDistance d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(EpanechnikovKernelString)
{
  EpanechnikovKernel d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(ExampleKernelString)
{
  ExampleKernel d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(GaussianKernelString)
{
  GaussianKernel d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(HyperbolicTangentKernelString)
{
  HyperbolicTangentKernel d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(LaplacianKernelString)
{
  LaplacianKernel d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(LinearKernelString)
{
  LinearKernel d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(PolynomialKernelString)
{
  PolynomialKernel d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(PSpectrumStringKernelString)
{
  const std::vector<std::vector<std::string> > s;
  const size_t t=1;
  PSpectrumStringKernel d(s,t);
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(SphericalKernelString)
{
  SphericalKernel d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(TriangularKernelString)
{
  TriangularKernel d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(IPMetricString)
{
  IPMetric<TriangularKernel> d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(LMetricString)
{
  LMetric<1> d;
  Log::Info << d;
}

BOOST_AUTO_TEST_CASE(MahalanobisDistanceString)
{
  MahalanobisDistance<> d;
  Log::Info << d;
}

BOOST_AUTO_TEST_SUITE_END();
