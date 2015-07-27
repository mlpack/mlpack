/**
 * @file to_string_test.cpp
 * @author Ryan Birmingham
 *
 * Test of the toString functionality.
 **/

#include <mlpack/core.hpp>
#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

#include <mlpack/core/metrics/ip_metric.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/metrics/mahalanobis_distance.hpp>

#include <mlpack/core/kernels/pspectrum_string_kernel.hpp>
#include <mlpack/core/kernels/example_kernel.hpp>

#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
#include <mlpack/core/optimizers/sdp/lrsdp.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/methods/nca/nca_softmax_error_function.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian_test_functions.hpp>

#include <mlpack/core/tree/ballbound.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/bounds.hpp>
#include <mlpack/core/tree/hrectbound.hpp>
#include <mlpack/core/tree/statistic.hpp>

#include <mlpack/methods/cf/cf.hpp>
#include <mlpack/methods/det/dtree.hpp>
#include <mlpack/methods/emst/dtb.hpp>
#include <mlpack/methods/fastmks/fastmks.hpp>
#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/kernel_pca/kernel_pca.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/local_coordinate_coding/lcc.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/methods/lsh/lsh_search.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/nca/nca.hpp>
#include <mlpack/methods/pca/pca.hpp>
#include <mlpack/methods/radical/radical.hpp>
#include <mlpack/methods/range_search/range_search.hpp>
#include <mlpack/methods/rann/ra_search.hpp>
#include <mlpack/methods/sparse_coding/sparse_coding.hpp>

using namespace mlpack;
using namespace mlpack::kernel;
using namespace mlpack::distribution;
using namespace mlpack::metric;
using namespace mlpack::nca;
using namespace mlpack::bound;
using namespace mlpack::tree;
using namespace mlpack::neighbor;

//using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(ToStringTest);

std::ostringstream testOstream;

BOOST_AUTO_TEST_CASE(DiscreteDistributionString)
{
  DiscreteDistribution d("0.4 0.5 0.1");
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(GaussianDistributionString)
{
  GaussianDistribution d("0.1 0.3", "1.0 0.1; 0.1 1.0");
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(CosineDistanceString)
{
  CosineDistance d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(EpanechnikovKernelString)
{
  EpanechnikovKernel d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(ExampleKernelString)
{
  ExampleKernel d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(GaussianKernelString)
{
  GaussianKernel d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(HyperbolicTangentKernelString)
{
  HyperbolicTangentKernel d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LaplacianKernelString)
{
  LaplacianKernel d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LinearKernelString)
{
  LinearKernel d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(PolynomialKernelString)
{
  PolynomialKernel d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(PSpectrumStringKernelString)
{
  const std::vector<std::vector<std::string> > s;
  const size_t t = 1;
  PSpectrumStringKernel d(s, t);
  Log::Debug << d;
  testOstream << d;
  std::string sttm = d.ToString();
  BOOST_REQUIRE_NE(sttm, "");
}

BOOST_AUTO_TEST_CASE(SphericalKernelString)
{
  SphericalKernel d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(TriangularKernelString)
{
  TriangularKernel d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(IPMetricString)
{
  IPMetric<TriangularKernel> d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LMetricString)
{
  LMetric<1> d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(MahalanobisDistanceString)
{
  MahalanobisDistance<> d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(SGDString)
{
  const arma::mat g(2, 2);
  const arma::Col<size_t> v(2);
  SoftmaxErrorFunction<> a(g, v);
  mlpack::optimization::SGD<SoftmaxErrorFunction<> > d(a);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(L_BFGSString)
{
  const arma::mat g(2, 2);
  const arma::Col<size_t> v(2);
  SoftmaxErrorFunction<> a(g, v);
  mlpack::optimization::L_BFGS<SoftmaxErrorFunction<> > d(a);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(AugLagString)
{
  mlpack::optimization::AugLagrangianTestFunction a;
  mlpack::optimization::AugLagrangian<
      mlpack::optimization::AugLagrangianTestFunction> d(a);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LRSDPString)
{
  arma::mat c(40, 40);
  c.randn();
  const size_t b=3;
  mlpack::optimization::LRSDP<mlpack::optimization::SDP<arma::sp_mat>> d(b,b,c);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(BallBoundString)
{
  BallBound<> d(3.5, "1.0 2.0");
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(BinSpaceString)
{
  arma::mat q(2, 50);
  q.randu();
  KDTree<ManhattanDistance, EmptyStatistic, arma::mat> d(q);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(CoverTreeString)
{
  arma::mat q(2, 50);
  q.randu();
  StandardCoverTree<EuclideanDistance, EmptyStatistic, arma::mat> d(q);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(CFString)
{
  arma::mat c(3, 3);
  c(0, 0) = 1;
  c(1, 0) = 2;
  c(2, 0) = 1.5;
  c(0, 1) = 2;
  c(1, 1) = 3;
  c(2, 1) = 2.0;
  c(0, 2) = 1;
  c(1, 2) = 3;
  c(2, 2) = 0.7;
  mlpack::cf::CF<> d(c);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(DetString)
{
  arma::mat c(4, 4);
  c.randn();
  mlpack::det::DTree d(c);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(EmstString)
{
  arma::mat c(4, 4);
  c.randu();
  mlpack::emst::DualTreeBoruvka<> d(c);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(FastMKSString)
{
  arma::mat c(4, 4);
  c.randn();
  mlpack::fastmks::FastMKS<LinearKernel> d(c);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(GMMString)
{
  arma::mat c(400, 40);
  c.randn();
  mlpack::gmm::GMM<> d(5, 4);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(HMMString)
{
  mlpack::hmm::HMM<> d(5, 4);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(KPCAString)
{
  LinearKernel k;
  mlpack::kpca::KernelPCA<LinearKernel> d(k, false);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(KMeansString)
{
  mlpack::kmeans::KMeans<metric::ManhattanDistance> d(100);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LarsString)
{
  mlpack::regression::LARS d(false);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LinRegString)
{
  arma::mat c(40, 40);
  arma::mat b(40, 1);
  c.randn();
  b.randn();
  mlpack::regression::LinearRegression d(c, b);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LCCString)
{
  arma::mat c(40,40);
  const size_t b=3;
  const double a=1;
  c.randn();
  mlpack::lcc::LocalCoordinateCoding<> d(c, b, a);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LogRegString)
{
  arma::mat c(40, 40);
  arma::mat b(40, 1);
  c.randn();
  b.randn();
  mlpack::regression::LogisticRegression<> d(c, b);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(LSHString)
{
  arma::mat c(40, 40);
  const size_t b=3;
  c.randn();
  mlpack::neighbor::LSHSearch<NearestNeighborSort> d(c, b, b);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(NeighborString)
{
  arma::mat c(40, 40);
  c.randn();
  mlpack::neighbor::NeighborSearch<> d(c);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

/*
BOOST_AUTO_TEST_CASE(NMFString)
{
  arma::mat c(40, 40);
  c.randn();
  mlpack::amf::AMF<> d;
  Log::Debug << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}
*/

BOOST_AUTO_TEST_CASE(NCAString)
{
  arma::mat c(40, 40);
  arma::Col<size_t> b(3);
  c.randn();
  mlpack::nca::NCA<> d(c, b);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(PCAString)
{
  mlpack::pca::PCA d(true);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(RadicalString)
{
  mlpack::radical::Radical d;
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(RangeSearchString)
{
  arma::mat c(40, 40);
  c.randn();
  mlpack::range::RangeSearch<> d(c);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(RannString)
{
  arma::mat c(40, 40);
  c.randn();
  mlpack::neighbor::RASearch<> d(c);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_CASE(SparseCodingString)
{
  arma::mat c(40, 40);
  c.randn();
  const size_t b=3;
  double a=0.1;
  mlpack::sparse_coding::SparseCoding<> d(c,b,a);
  Log::Debug << d;
  testOstream << d;
  std::string s = d.ToString();
  BOOST_REQUIRE_NE(s, "");
}

BOOST_AUTO_TEST_SUITE_END();

