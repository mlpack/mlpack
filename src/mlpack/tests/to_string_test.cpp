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

#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
//#include <mlpack/core/optimizers/lrsdp/lrsdp.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/methods/nca/nca_softmax_error_function.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian_test_functions.hpp>

#include <mlpack/core/tree/ballbound.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/tree/mrkd_statistic.hpp>
#include <mlpack/core/tree/hrectbound.hpp>
#include <mlpack/core/tree/periodichrectbound.hpp>
#include <mlpack/core/tree/statistic.hpp>
#include <mlpack/methods/cf/cf.hpp>
#include <mlpack/methods/det/dtree.hpp>
#include <mlpack/methods/emst/dtb.hpp>
#include <mlpack/methods/fastmks/fastmks.hpp>
#include <mlpack/methods/gmm/gmm.hpp>

using namespace mlpack;
using namespace mlpack::kernel;
using namespace mlpack::distribution;
using namespace mlpack::metric;
using namespace mlpack::nca;
using namespace mlpack::bound;
using namespace mlpack::tree;

//using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(ToStringTest);

BOOST_AUTO_TEST_CASE(DiscreteDistributionString)
{
  DiscreteDistribution d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(GaussianDistributionString)
{
  GaussianDistribution d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(CosineDistanceString)
{
  CosineDistance d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(EpanechnikovKernelString)
{
  EpanechnikovKernel d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(ExampleKernelString)
{
  ExampleKernel d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(GaussianKernelString)
{
  GaussianKernel d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(HyperbolicTangentKernelString)
{
  HyperbolicTangentKernel d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LaplacianKernelString)
{
  LaplacianKernel d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LinearKernelString)
{
  LinearKernel d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(PolynomialKernelString)
{
  PolynomialKernel d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(PSpectrumStringKernelString)
{
  const std::vector<std::vector<std::string> > s;
  const size_t t=1;
  PSpectrumStringKernel d(s,t);
  Log::Debug << d;
  std::string sttm = d.ToString();
  BOOST_REQUIRE_NE(sttm, "");
}

BOOST_AUTO_TEST_CASE(SphericalKernelString)
{
  SphericalKernel d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(TriangularKernelString)
{
  TriangularKernel d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(IPMetricString)
{
  IPMetric<TriangularKernel> d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LMetricString)
{
  LMetric<1> d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(MahalanobisDistanceString)
{
  MahalanobisDistance<> d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(SGDString)
{
  const arma::mat g(2,2);
  const arma::Col<size_t> v(2);
  SoftmaxErrorFunction<> a(g,v);
	mlpack::optimization::SGD<SoftmaxErrorFunction<> > d(a);
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(L_BFGSString)
{
  const arma::mat g(2,2);
  const arma::Col<size_t> v(2);
  SoftmaxErrorFunction<> a(g,v);
	mlpack::optimization::L_BFGS<SoftmaxErrorFunction<> > d(a);
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(AugLagString)
{
	mlpack::optimization::AugLagrangianTestFunction a;
  mlpack::optimization::AugLagrangianFunction 
      <mlpack::optimization::AugLagrangianTestFunction> q(a);
	mlpack::optimization::AugLagrangian 
      <mlpack::optimization::AugLagrangianTestFunction> d(a);
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(BallBoundString)
{
  BallBound<> d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(BinSpaceString)
{  
  arma::mat q(2,2);
  q.randu();
  BinarySpaceTree<HRectBound<1> > d(q);
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

/**
BOOST_AUTO_TEST_CASE(CFString)
{ 
  arma::mat c(10,10);
  mlpack::cf::CF d(c);
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}
**/
BOOST_AUTO_TEST_CASE(DetString)
{ 
  arma::mat c(4,4);
  c.randn(); 
  mlpack::det::DTree d(c);
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(EmstString)
{ 
  arma::mat c(4,4);
  c.randn(); 
  mlpack::emst::DualTreeBoruvka<> d(c);
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(FastMKSString)
{ 
  arma::mat c(4,4);
  c.randn();
  mlpack::fastmks::FastMKS<LinearKernel> d(c);
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(GMMString)
{ 
  arma::mat c(400,40);
  c.randn();
  mlpack::gmm::GMM<> d(5, 4);
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_SUITE_END();
