/**
 * @file kernel_test.cpp
 * @author Ryan Curtin
 * @author Ajinkya Kale
 *
 * Tests for the various kernel classes.
 */
#include <mlpack/core/kernels/cosine_distance.hpp>
#include <mlpack/core/kernels/epanechnikov_kernel.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/hyperbolic_tangent_kernel.hpp>
#include <mlpack/core/kernels/laplacian_kernel.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/polynomial_kernel.hpp>
#include <mlpack/core/kernels/spherical_kernel.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/metrics/mahalanobis_distance.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace mlpack::kernel;
using namespace mlpack::metric;

BOOST_AUTO_TEST_SUITE(KernelTest);

/**
 * Basic test of the Manhattan distance.
 */
BOOST_AUTO_TEST_CASE(manhattan_distance)
{
  // A couple quick tests.
  arma::vec a = "1.0 3.0 4.0";
  arma::vec b = "3.0 3.0 5.0";

  BOOST_REQUIRE_CLOSE(ManhattanDistance::Evaluate(a, b), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE(ManhattanDistance::Evaluate(b, a), 3.0, 1e-5);

  // Check also for when the root is taken (should be the same).
  BOOST_REQUIRE_CLOSE((LMetric<1, true>::Evaluate(a, b)), 3.0, 1e-5);
  BOOST_REQUIRE_CLOSE((LMetric<1, true>::Evaluate(b, a)), 3.0, 1e-5);
}

/**
 * Basic test of squared Euclidean distance.
 */
BOOST_AUTO_TEST_CASE(squared_euclidean_distance)
{
  // Sample 2-dimensional vectors.
  arma::vec a = "1.0  2.0";
  arma::vec b = "0.0 -2.0";

  BOOST_REQUIRE_CLOSE(SquaredEuclideanDistance::Evaluate(a, b), 17.0, 1e-5);
  BOOST_REQUIRE_CLOSE(SquaredEuclideanDistance::Evaluate(b, a), 17.0, 1e-5);
}

/**
 * Basic test of Euclidean distance.
 */
BOOST_AUTO_TEST_CASE(euclidean_distance)
{
  arma::vec a = "1.0 3.0 5.0 7.0";
  arma::vec b = "4.0 0.0 2.0 0.0";

  BOOST_REQUIRE_CLOSE(EuclideanDistance::Evaluate(a, b), sqrt(76.0), 1e-5);
  BOOST_REQUIRE_CLOSE(EuclideanDistance::Evaluate(b, a), sqrt(76.0), 1e-5);
}

/**
 * Arbitrary test case for coverage.
 */
BOOST_AUTO_TEST_CASE(arbitrary_case)
{
  arma::vec a = "3.0 5.0 6.0 7.0";
  arma::vec b = "1.0 2.0 1.0 0.0";

  BOOST_REQUIRE_CLOSE(LMetric<3>::Evaluate(a, b), 503.0, 1e-5);
  BOOST_REQUIRE_CLOSE(LMetric<3>::Evaluate(b, a), 503.0, 1e-5);

  BOOST_REQUIRE_CLOSE((LMetric<3, true>::Evaluate(a, b)), 7.95284762, 1e-5);
  BOOST_REQUIRE_CLOSE((LMetric<3, true>::Evaluate(b, a)), 7.95284762, 1e-5);
}

/**
 * Make sure two vectors of all zeros return zero distance, for a few different
 * powers.
 */
BOOST_AUTO_TEST_CASE(lmetric_zeros)
{
  arma::vec a(250);
  a.fill(0.0);

  // We cannot use a loop because compilers seem to be unable to unroll the loop
  // and realize the variable actually is knowable at compile-time.
  BOOST_REQUIRE((LMetric<1, false>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<1, true>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<2, false>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<2, true>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<3, false>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<3, true>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<4, false>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<4, true>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<5, false>::Evaluate(a, a)) == 0);
  BOOST_REQUIRE((LMetric<5, true>::Evaluate(a, a)) == 0);
}

/**
 * Simple test of Mahalanobis distance with unset covariance matrix.
 */
BOOST_AUTO_TEST_CASE(md_unset_covariance)
{
  MahalanobisDistance<false> md;
  arma::vec a = "1.0 2.0 2.0 3.0";
  arma::vec b = "0.0 0.0 1.0 3.0";

  BOOST_REQUIRE_CLOSE(md.Evaluate(a, b), 6.0, 1e-5);
  BOOST_REQUIRE_CLOSE(md.Evaluate(b, a), 6.0, 1e-5);
}

/**
 * Simple test of Mahalanobis distance with unset covariance matrix and
 * t_take_root set to true.
 */
BOOST_AUTO_TEST_CASE(md_root_unset_covariance)
{
  MahalanobisDistance<true> md;
  arma::vec a = "1.0 2.0 2.5 5.0";
  arma::vec b = "0.0 2.0 0.5 8.0";

  BOOST_REQUIRE_CLOSE(md.Evaluate(a, b), sqrt(14.0), 1e-5);
  BOOST_REQUIRE_CLOSE(md.Evaluate(b, a), sqrt(14.0), 1e-5);
}

/**
 * Simple test with diagonal covariance matrix.
 */
BOOST_AUTO_TEST_CASE(md_diagonal_covariance)
{
  arma::mat cov = arma::eye<arma::mat>(5, 5);
  cov(0, 0) = 2.0;
  cov(1, 1) = 0.5;
  cov(2, 2) = 3.0;
  cov(3, 3) = 1.0;
  cov(4, 4) = 1.5;
  MahalanobisDistance<false> md(cov);

  arma::vec a = "1.0 2.0 2.0 4.0 5.0";
  arma::vec b = "2.0 3.0 1.0 1.0 0.0";

  BOOST_REQUIRE_CLOSE(md.Evaluate(a, b), 52.0, 1e-5);
  BOOST_REQUIRE_CLOSE(md.Evaluate(b, a), 52.0, 1e-5);
}

/**
 * More specific case with more difficult covariance matrix.
 */
BOOST_AUTO_TEST_CASE(md_full_covariance)
{
  arma::mat cov = "1.0 2.0 3.0 4.0;"
                  "0.5 0.6 0.7 0.1;"
                  "3.4 4.3 5.0 6.1;"
                  "1.0 2.0 4.0 1.0;";
  MahalanobisDistance<false> md(cov);

  arma::vec a = "1.0 2.0 2.0 4.0";
  arma::vec b = "2.0 3.0 1.0 1.0";

  BOOST_REQUIRE_CLOSE(md.Evaluate(a, b), 15.7, 1e-5);
  BOOST_REQUIRE_CLOSE(md.Evaluate(b, a), 15.7, 1e-5);
}

/**
 * Simple test case for the cosine distance.
 */
BOOST_AUTO_TEST_CASE(cosine_distance_same_angle)
{
  arma::vec a = "1.0 2.0 3.0";
  arma::vec b = "2.0 4.0 6.0";

  BOOST_REQUIRE_SMALL(CosineDistance::Evaluate(a, b), 1e-5);
  BOOST_REQUIRE_SMALL(CosineDistance::Evaluate(b, a), 1e-5);
}

/**
 * Now let's have them be orthogonal.
 */
BOOST_AUTO_TEST_CASE(cosine_distance_orthogonal)
{
  arma::vec a = "0.0 1.0";
  arma::vec b = "1.0 0.0";

  BOOST_REQUIRE_CLOSE(CosineDistance::Evaluate(a, b), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(CosineDistance::Evaluate(b, a), 1.0, 1e-5);
}

/**
 * Some random angle test.
 */
BOOST_AUTO_TEST_CASE(cosine_distance_random_test)
{
  arma::vec a = "0.1 0.2 0.3 0.4 0.5";
  arma::vec b = "1.2 1.0 0.8 -0.3 -0.5";

  BOOST_REQUIRE_CLOSE(CosineDistance::Evaluate(a, b), 1 - 0.1385349024, 1e-5);
  BOOST_REQUIRE_CLOSE(CosineDistance::Evaluate(b, a), 1 - 0.1385349024, 1e-5);
}

/**
 * Linear Kernel test.
 */
BOOST_AUTO_TEST_CASE(linear_kernel)
{
  arma::vec a = ".2 .3 .4 .1";
  arma::vec b = ".56 .21 .623 .82";

  LinearKernel lk;
  BOOST_REQUIRE_CLOSE(lk.Evaluate(a,b), .5062, 1e-5);
  BOOST_REQUIRE_CLOSE(lk.Evaluate(b,a), .5062, 1e-5);
}

/**
 * Linear Kernel test, orthogonal vectors.
 */
BOOST_AUTO_TEST_CASE(linear_kernel_orthogonal)
{
  arma::vec a = "1 0 0";
  arma::vec b = "0 0 1";

  LinearKernel lk;
  BOOST_REQUIRE_SMALL(lk.Evaluate(a,b), 1e-5);
  BOOST_REQUIRE_SMALL(lk.Evaluate(b,a), 1e-5);
}

BOOST_AUTO_TEST_CASE(gaussian_kernel)
{
  arma::vec a = "1 0 0";
  arma::vec b = "0 1 0";
  arma::vec c = "0 0 1";

  GaussianKernel gk(.5);
  BOOST_REQUIRE_CLOSE(gk.Evaluate(a,b), .018315638888734, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.Evaluate(b,a), .018315638888734, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.Evaluate(a,c), .018315638888734, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.Evaluate(c,a), .018315638888734, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.Evaluate(b,c), .018315638888734, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.Evaluate(c,b), .018315638888734, 1e-5);
  /* check the single dimension evaluate function */
  BOOST_REQUIRE_CLOSE(gk.Evaluate(1.0), 0.1353352832366127, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.Evaluate(2.0), 0.00033546262790251185, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.Evaluate(3.0), 1.5229979744712629e-08, 1e-5);
  /* check the normalization constant */
  BOOST_REQUIRE_CLOSE(gk.Normalizer(1), 1.2533141373155001, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.Normalizer(2), 1.5707963267948963, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.Normalizer(3), 1.9687012432153019, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.Normalizer(4), 2.4674011002723386, 1e-5);
  /* check the convolution integral */
  BOOST_REQUIRE_CLOSE(gk.ConvolutionIntegral(a,b), 0.024304474038457577, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.ConvolutionIntegral(a,c), 0.024304474038457577, 1e-5);
  BOOST_REQUIRE_CLOSE(gk.ConvolutionIntegral(b,c), 0.024304474038457577, 1e-5);

}

BOOST_AUTO_TEST_CASE(spherical_kernel)
{
  arma::vec a = "1.0 0.0";
  arma::vec b = "0.0 1.0";
  arma::vec c = "0.2 0.9";

  SphericalKernel sk(.5);
  BOOST_REQUIRE_CLOSE(sk.Evaluate(a,b), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(sk.Evaluate(a,c), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(sk.Evaluate(b,c), 1.0, 1e-5);
  /* check the single dimension evaluate function */
  BOOST_REQUIRE_CLOSE(sk.Evaluate(0.10), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(sk.Evaluate(0.25), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(sk.Evaluate(0.50), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(sk.Evaluate(1.00), 0.0, 1e-5);
  /* check the normalization constant */
  BOOST_REQUIRE_CLOSE(sk.Normalizer(1), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(sk.Normalizer(2), 0.78539816339744828, 1e-5);
  BOOST_REQUIRE_CLOSE(sk.Normalizer(3), 0.52359877559829893, 1e-5);
  BOOST_REQUIRE_CLOSE(sk.Normalizer(4), 0.30842513753404244, 1e-5);
  /* check the convolution integral */
  BOOST_REQUIRE_CLOSE(sk.ConvolutionIntegral(a,b), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(sk.ConvolutionIntegral(a,c), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(sk.ConvolutionIntegral(b,c), 1.0021155029652784, 1e-5);
}

BOOST_AUTO_TEST_CASE(epanechnikov_kernel)
{
  arma::vec a = "1.0 0.0";
  arma::vec b = "0.0 1.0";
  arma::vec c = "0.1 0.9";

  EpanechnikovKernel ek(.5);
  BOOST_REQUIRE_CLOSE(ek.Evaluate(a,b), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(ek.Evaluate(b,c), 0.92, 1e-5);
  BOOST_REQUIRE_CLOSE(ek.Evaluate(a,c), 0.0, 1e-5);
  /* check the single dimension evaluate function */
  BOOST_REQUIRE_CLOSE(ek.Evaluate(0.10), 0.96, 1e-5);
  BOOST_REQUIRE_CLOSE(ek.Evaluate(0.25), 0.75, 1e-5);
  BOOST_REQUIRE_CLOSE(ek.Evaluate(0.50), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(ek.Evaluate(1.00), 0.0, 1e-5);
  /* check the normalization constant */
  BOOST_REQUIRE_CLOSE(ek.Normalizer(1), 0.666666666666666, 1e-5);
  BOOST_REQUIRE_CLOSE(ek.Normalizer(2), 0.39269908169872414, 1e-5);
  BOOST_REQUIRE_CLOSE(ek.Normalizer(3), 0.20943951023931956, 1e-5);
  BOOST_REQUIRE_CLOSE(ek.Normalizer(4), 0.10280837917801415, 1e-5);
  /* check the convolution integral */
  BOOST_REQUIRE_CLOSE(ek.ConvolutionIntegral(a,b), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(ek.ConvolutionIntegral(a,c), 0.0, 1e-5);
  BOOST_REQUIRE_CLOSE(ek.ConvolutionIntegral(b,c), 1.5263455690698258, 1e-5);
}
BOOST_AUTO_TEST_CASE(polynomial_kernel)
{
  arma::vec a = "0 0 1";
  arma::vec b = "0 1 0";

  PolynomialKernel pk(5.0, 5.0);
  BOOST_REQUIRE_CLOSE(pk.Evaluate(a, b), 3125.0, 0);
  BOOST_REQUIRE_CLOSE(pk.Evaluate(b, a), 3125.0, 0);
}

BOOST_AUTO_TEST_CASE(hyperbolic_tangent_kernel)
{
  arma::vec a = "0 0 1";
  arma::vec b = "0 1 0";

  HyperbolicTangentKernel tk(5.0, 5.0);
  BOOST_REQUIRE_CLOSE(tk.Evaluate(a, b), 0.9999092, 1e-5);
  BOOST_REQUIRE_CLOSE(tk.Evaluate(b, a), 0.9999092, 1e-5);
}

BOOST_AUTO_TEST_CASE(laplacian_kernel)
{
  arma::vec a = "0 0 1";
  arma::vec b = "0 1 0";

  LaplacianKernel lk(1.0);
  BOOST_REQUIRE_CLOSE(lk.Evaluate(a, b), 0.243116734, 5e-5);
  BOOST_REQUIRE_CLOSE(lk.Evaluate(b, a), 0.243116734, 5e-5);
}

BOOST_AUTO_TEST_SUITE_END();
