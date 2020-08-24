/**
 * @file tests/kernel_test.cpp
 * @author Ryan Curtin
 * @author Ajinkya Kale
 *
 * Tests for the various kernel classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core/kernels/cosine_distance.hpp>
#include <mlpack/core/kernels/epanechnikov_kernel.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/hyperbolic_tangent_kernel.hpp>
#include <mlpack/core/kernels/laplacian_kernel.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/polynomial_kernel.hpp>
#include <mlpack/core/kernels/spherical_kernel.hpp>
#include <mlpack/core/kernels/pspectrum_string_kernel.hpp>
#include <mlpack/core/kernels/cauchy_kernel.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/core/metrics/mahalanobis_distance.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"
#include "serialization_catch.hpp"

using namespace mlpack;
using namespace mlpack::kernel;
using namespace mlpack::metric;

/**
 * Basic test of the Manhattan distance.
 */
TEST_CASE("ManhattanDistanceTest", "[KernelTest]")
{
  // A couple quick tests.
  arma::vec a = "1.0 3.0 4.0";
  arma::vec b = "3.0 3.0 5.0";

  REQUIRE(ManhattanDistance::Evaluate(a, b) == Approx(3.0).epsilon(1e-7));
  REQUIRE(ManhattanDistance::Evaluate(b, a) == Approx(3.0).epsilon(1e-7));

  // Check also for when the root is taken (should be the same).
  REQUIRE((LMetric<1, true>::Evaluate(a, b)) == Approx(3.0).epsilon(1e-7));
  REQUIRE((LMetric<1, true>::Evaluate(b, a)) == Approx(3.0).epsilon(1e-7));
}

/**
 * Basic test of squared Euclidean distance.
 */
TEST_CASE("SquaredEuclideanDistanceTest", "[KernelTest]")
{
  // Sample 2-dimensional vectors.
  arma::vec a = "1.0  2.0";
  arma::vec b = "0.0 -2.0";

  REQUIRE(SquaredEuclideanDistance::Evaluate(a, b) ==
      Approx(17.0).epsilon(1e-7));
  REQUIRE(SquaredEuclideanDistance::Evaluate(b, a) ==
      Approx(17.0).epsilon(1e-7));
}

/**
 * Basic test of Euclidean distance.
 */
TEST_CASE("EuclideanDistanceTest", "[KernelTest]")
{
  arma::vec a = "1.0 3.0 5.0 7.0";
  arma::vec b = "4.0 0.0 2.0 0.0";

  REQUIRE(EuclideanDistance::Evaluate(a, b) ==
      Approx(sqrt(76.0)).epsilon(1e-7));
  REQUIRE(EuclideanDistance::Evaluate(b, a) ==
      Approx(sqrt(76.0)).epsilon(1e-7));
}

/**
 * Arbitrary test case for coverage.
 */
TEST_CASE("ArbitraryCaseTest", "[KernelTest]")
{
  arma::vec a = "3.0 5.0 6.0 7.0";
  arma::vec b = "1.0 2.0 1.0 0.0";

  REQUIRE((LMetric<3, false>::Evaluate(a, b)) == Approx(503.0).epsilon(1e-7));
  REQUIRE((LMetric<3, false>::Evaluate(b, a)) == Approx(503.0).epsilon(1e-7));

  REQUIRE((LMetric<3, true>::Evaluate(a, b)) ==
      Approx(7.95284762).epsilon(1e-7));
  REQUIRE((LMetric<3, true>::Evaluate(b, a)) ==
      Approx(7.95284762).epsilon(1e-7));
}

/**
 * Make sure two vectors of all zeros return zero distance, for a few different
 * powers.
 */
TEST_CASE("LMetricZerosTest", "[KernelTest]")
{
  arma::vec a(250);
  a.fill(0.0);

  // We cannot use a loop because compilers seem to be unable to unroll the loop
  // and realize the variable actually is knowable at compile-time.
  REQUIRE(LMetric<1, false>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<1, true>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<2, false>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<2, true>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<3, false>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<3, true>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<4, false>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<4, true>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<5, false>::Evaluate(a, a) == 0);
  REQUIRE(LMetric<5, true>::Evaluate(a, a) == 0);
}

/**
 * Simple test of Mahalanobis distance with unset covariance matrix in
 * constructor.
 */
TEST_CASE("MDUnsetCovarianceTest", "[KernelTest]")
{
  MahalanobisDistance<false> md;
  md.Covariance() = arma::eye<arma::mat>(4, 4);
  arma::vec a = "1.0 2.0 2.0 3.0";
  arma::vec b = "0.0 0.0 1.0 3.0";

  REQUIRE(md.Evaluate(a, b) == Approx(6.0).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(6.0).epsilon(1e-7));
}

/**
 * Simple test of Mahalanobis distance with unset covariance matrix in
 * constructor and t_take_root set to true.
 */
TEST_CASE("MDRootUnsetCovarianceTest", "[KernelTest]")
{
  MahalanobisDistance<true> md;
  md.Covariance() = arma::eye<arma::mat>(4, 4);
  arma::vec a = "1.0 2.0 2.5 5.0";
  arma::vec b = "0.0 2.0 0.5 8.0";

  REQUIRE(md.Evaluate(a, b) == Approx(sqrt(14.0)).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(sqrt(14.0)).epsilon(1e-7));
}

/**
 * Simple test of Mahalanobis distance setting identity covariance in
 * constructor.
 */
TEST_CASE("MDEyeCovarianceTest", "[KernelTest]")
{
  MahalanobisDistance<false> md(4);
  arma::vec a = "1.0 2.0 2.0 3.0";
  arma::vec b = "0.0 0.0 1.0 3.0";

  REQUIRE(md.Evaluate(a, b) == Approx(6.0).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(6.0).epsilon(1e-7));
}

/**
 * Simple test of Mahalanobis distance setting identity covariance in
 * constructor and t_take_root set to true.
 */
TEST_CASE("MDRootEyeCovarianceTest", "[KernelTest]")
{
  MahalanobisDistance<true> md(4);
  arma::vec a = "1.0 2.0 2.5 5.0";
  arma::vec b = "0.0 2.0 0.5 8.0";

  REQUIRE(md.Evaluate(a, b) == Approx(sqrt(14.0)).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(sqrt(14.0)).epsilon(1e-7));
}

/**
 * Simple test with diagonal covariance matrix.
 */
TEST_CASE("MDDiagonalCovarianceTest", "[KernelTest]")
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

  REQUIRE(md.Evaluate(a, b) == Approx(52.0).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(52.0).epsilon(1e-7));
}

/**
 * More specific case with more difficult covariance matrix.
 */
TEST_CASE("MDFullCovarianceTest", "[KernelTest]")
{
  arma::mat cov = "1.0 2.0 3.0 4.0;"
                  "0.5 0.6 0.7 0.1;"
                  "3.4 4.3 5.0 6.1;"
                  "1.0 2.0 4.0 1.0;";
  MahalanobisDistance<false> md(cov);

  arma::vec a = "1.0 2.0 2.0 4.0";
  arma::vec b = "2.0 3.0 1.0 1.0";

  REQUIRE(md.Evaluate(a, b) == Approx(15.7).epsilon(1e-7));
  REQUIRE(md.Evaluate(b, a) == Approx(15.7).epsilon(1e-7));
}

/**
 * Simple test case for the cosine distance.
 */
TEST_CASE("CosineDistanceSameAngleTest", "[KernelTest]")
{
  arma::vec a = "1.0 2.0 3.0";
  arma::vec b = "2.0 4.0 6.0";

  REQUIRE(CosineDistance::Evaluate(a, b) == Approx(1.0).epsilon(1e-7));
  REQUIRE(CosineDistance::Evaluate(b, a) == Approx(1.0).epsilon(1e-7));
}

/**
 * Now let's have them be orthogonal.
 */
TEST_CASE("CosineDistanceOrthogonalTest", "[KernelTest]")
{
  arma::vec a = "0.0 1.0";
  arma::vec b = "1.0 0.0";

  REQUIRE(CosineDistance::Evaluate(a, b) == Approx(0.0).margin(1e-5));
  REQUIRE(CosineDistance::Evaluate(b, a) == Approx(0.0).margin(1e-5));
}

/**
 * Some random angle test.
 */
TEST_CASE("CosineDistanceRandomTest", "[KernelTest]")
{
  arma::vec a = "0.1 0.2 0.3 0.4 0.5";
  arma::vec b = "1.2 1.0 0.8 -0.3 -0.5";

  REQUIRE(CosineDistance::Evaluate(a, b) ==
      Approx(0.1385349024).epsilon(1e-7));
  REQUIRE(CosineDistance::Evaluate(b, a) ==
      Approx(0.1385349024).epsilon(1e-7));
}

/**
 * Linear Kernel test.
 */
TEST_CASE("LinearKernelTest", "[KernelTest]")
{
  arma::vec a = ".2 .3 .4 .1";
  arma::vec b = ".56 .21 .623 .82";

  LinearKernel lk;
  REQUIRE(lk.Evaluate(a, b) == Approx(.5062).epsilon(1e-7));
  REQUIRE(lk.Evaluate(b, a) == Approx(.5062).epsilon(1e-7));
}

/**
 * Linear Kernel test, orthogonal vectors.
 */
TEST_CASE("LinearKernelOrthogonalTest", "[KernelTest]")
{
  arma::vec a = "1 0 0";
  arma::vec b = "0 0 1";

  LinearKernel lk;
  REQUIRE(lk.Evaluate(a, b) == Approx(0.0).margin(1e-5));
  REQUIRE(lk.Evaluate(a, b) == Approx(0.0).margin(1e-5));
}

TEST_CASE("GaussianKernelTest", "[KernelTest]")
{
  arma::vec a = "1 0 0";
  arma::vec b = "0 1 0";
  arma::vec c = "0 0 1";

  GaussianKernel gk(.5);
  REQUIRE(gk.Evaluate(a, b) == Approx(.018315638888734).epsilon(1e-7));
  REQUIRE(gk.Evaluate(b, a) == Approx(.018315638888734).epsilon(1e-7));
  REQUIRE(gk.Evaluate(a, c) == Approx(.018315638888734).epsilon(1e-7));
  REQUIRE(gk.Evaluate(c, a) == Approx(.018315638888734).epsilon(1e-7));
  REQUIRE(gk.Evaluate(b, c) == Approx(.018315638888734).epsilon(1e-7));
  REQUIRE(gk.Evaluate(c, b) == Approx(.018315638888734).epsilon(1e-7));
  /* check the single dimension evaluate function */
  REQUIRE(gk.Evaluate(1.0) == Approx(0.1353352832366127).epsilon(1e-7));
  REQUIRE(gk.Evaluate(2.0) == Approx(0.00033546262790251185).epsilon(1e-7));
  REQUIRE(gk.Evaluate(3.0) == Approx(1.5229979744712629e-08).epsilon(1e-7));
  /* check the normalization constant */
  REQUIRE(gk.Normalizer(1) == Approx(1.2533141373155001).epsilon(1e-7));
  REQUIRE(gk.Normalizer(2) == Approx(1.5707963267948963).epsilon(1e-7));
  REQUIRE(gk.Normalizer(3) == Approx(1.9687012432153019).epsilon(1e-7));
  REQUIRE(gk.Normalizer(4) == Approx(2.4674011002723386).epsilon(1e-7));
  /* check the convolution integral */
  REQUIRE(gk.ConvolutionIntegral(a, b) ==
      Approx(0.024304474038457577).epsilon(1e-7));
  REQUIRE(gk.ConvolutionIntegral(a, c) ==
      Approx(0.024304474038457577).epsilon(1e-7));
  REQUIRE(gk.ConvolutionIntegral(b, c) ==
      Approx(0.024304474038457577).epsilon(1e-7));
}

TEST_CASE("GaussianKernelSerializationTest", "[KernelTest]")
{
  GaussianKernel gk(0.5);
  GaussianKernel xmlGk(1.5), textGk, binaryGk(15.0);

  // Serialize the kernels.
  SerializeObjectAll(gk, xmlGk, textGk, binaryGk);

  REQUIRE(gk.Bandwidth() == Approx(0.5).epsilon(1e-7));
  REQUIRE(xmlGk.Bandwidth() == Approx(0.5).epsilon(1e-7));
  REQUIRE(textGk.Bandwidth() == Approx(0.5).epsilon(1e-7));
  REQUIRE(binaryGk.Bandwidth() == Approx(0.5).epsilon(1e-7));
}

TEST_CASE("SphericalKernelTest", "[KernelTest]")
{
  arma::vec a = "1.0 0.0";
  arma::vec b = "0.0 1.0";
  arma::vec c = "0.2 0.9";

  SphericalKernel sk(.5);
  REQUIRE(sk.Evaluate(a, b) == Approx(0.0).epsilon(1e-7));
  REQUIRE(sk.Evaluate(a, c) == Approx(0.0).epsilon(1e-7));
  REQUIRE(sk.Evaluate(b, c) == Approx(1.0).epsilon(1e-7));
  /* check the single dimension evaluate function */
  REQUIRE(sk.Evaluate(0.10) == Approx(1.0).epsilon(1e-7));
  REQUIRE(sk.Evaluate(0.25) == Approx(1.0).epsilon(1e-7));
  REQUIRE(sk.Evaluate(0.50) == Approx(1.0).epsilon(1e-7));
  REQUIRE(sk.Evaluate(1.00) == Approx(0.0).epsilon(1e-7));
  /* check the normalization constant */
  REQUIRE(sk.Normalizer(1) == Approx(1.0).epsilon(1e-7));
  REQUIRE(sk.Normalizer(2) == Approx(0.78539816339744828).epsilon(1e-7));
  REQUIRE(sk.Normalizer(3) == Approx(0.52359877559829893).epsilon(1e-7));
  REQUIRE(sk.Normalizer(4) == Approx(0.30842513753404244).epsilon(1e-7));
  /* check the convolution integral */
  REQUIRE(sk.ConvolutionIntegral(a, b) == Approx(0.0).epsilon(1e-7));
  REQUIRE(sk.ConvolutionIntegral(a, c) == Approx(0.0).epsilon(1e-7));
  REQUIRE(sk.ConvolutionIntegral(b, c) ==
      Approx(1.0021155029652784).epsilon(1e-7));
}

TEST_CASE("EpanechnikovKernelTest", "[KernelTest]")
{
  arma::vec a = "1.0 0.0";
  arma::vec b = "0.0 1.0";
  arma::vec c = "0.1 0.9";

  EpanechnikovKernel ek(.5);
  REQUIRE(ek.Evaluate(a, b) == Approx(0.0).epsilon(1e-7));
  REQUIRE(ek.Evaluate(b, c) == Approx(0.92).epsilon(1e-7));
  REQUIRE(ek.Evaluate(a, c) == Approx(0.0).epsilon(1e-7));
  /* check the single dimension evaluate function */
  REQUIRE(ek.Evaluate(0.10) == Approx(0.96).epsilon(1e-7));
  REQUIRE(ek.Evaluate(0.25) == Approx(0.75).epsilon(1e-7));
  REQUIRE(ek.Evaluate(0.50) == Approx(0.0).epsilon(1e-7));
  REQUIRE(ek.Evaluate(1.00) == Approx(0.0).epsilon(1e-7));
  /* check the normalization constant */
  REQUIRE(ek.Normalizer(1) == Approx(0.666666666666666).epsilon(1e-7));
  REQUIRE(ek.Normalizer(2) == Approx(0.39269908169872414).epsilon(1e-7));
  REQUIRE(ek.Normalizer(3) == Approx(0.20943951023931956).epsilon(1e-7));
  REQUIRE(ek.Normalizer(4) == Approx(0.10280837917801415).epsilon(1e-7));
  /* check the convolution integral */
  REQUIRE(ek.ConvolutionIntegral(a, b) == Approx(0.0).epsilon(1e-7));
  REQUIRE(ek.ConvolutionIntegral(a, c) == Approx(0.0).epsilon(1e-7));
  REQUIRE(ek.ConvolutionIntegral(b, c) ==
      Approx(1.5263455690698258).epsilon(1e-7));
}

TEST_CASE("PolynomialKernelTest", "[KernelTest]")
{
  arma::vec a = "0 0 1";
  arma::vec b = "0 1 0";

  PolynomialKernel pk(5.0, 5.0);
  REQUIRE(pk.Evaluate(a, b) == Approx(3125).epsilon(0));
  REQUIRE(pk.Evaluate(b, a) == Approx(3125).epsilon(0));
}

TEST_CASE("HyperbolicTangentKernelTest", "[KernelTest]")
{
  arma::vec a = "0 0 1";
  arma::vec b = "0 1 0";

  HyperbolicTangentKernel tk(5.0, 5.0);
  REQUIRE(tk.Evaluate(a, b) == Approx(0.9999092).epsilon(1e-7));
  REQUIRE(tk.Evaluate(b, a) == Approx(0.9999092).epsilon(1e-7));
}

TEST_CASE("LaplacianKernelTest", "[KernelTest]")
{
  arma::vec a = "0 0 1";
  arma::vec b = "0 1 0";

  LaplacianKernel lk(1.0);
  REQUIRE(lk.Evaluate(a, b) == Approx(0.243116734).epsilon(5e-7));
  REQUIRE(lk.Evaluate(b, a) == Approx(0.243116734).epsilon(5e-7));
}

// Ensure that the p-spectrum kernel successfully extracts all length-p
// substrings from the data.
TEST_CASE("PSpectrumSubstringExtractionTest", "[KernelTest]")
{
  std::vector<std::vector<std::string> > datasets;

  datasets.push_back(std::vector<std::string>());

  datasets[0].push_back("herpgle");
  datasets[0].push_back("herpagkle");
  datasets[0].push_back("klunktor");
  datasets[0].push_back("flibbynopple");

  datasets.push_back(std::vector<std::string>());

  datasets[1].push_back("floggy3245");
  datasets[1].push_back("flippydopflip");
  datasets[1].push_back("stupid fricking cat");
  datasets[1].push_back("food time isn't until later");
  datasets[1].push_back("leave me alone until 6:00");
  datasets[1].push_back("only after that do you get any food.");
  datasets[1].push_back("obloblobloblobloblobloblob");

  PSpectrumStringKernel p(datasets, 3);

  // Ensure the sizes are correct.
  REQUIRE(p.Counts().size() == 2);
  REQUIRE(p.Counts()[0].size() == 4);
  REQUIRE(p.Counts()[1].size() == 7);

  // herpgle: her, erp, rpg, pgl, gle
  REQUIRE(p.Counts()[0][0].size() == 5);
  REQUIRE(p.Counts()[0][0]["her"] == 1);
  REQUIRE(p.Counts()[0][0]["erp"] == 1);
  REQUIRE(p.Counts()[0][0]["rpg"] == 1);
  REQUIRE(p.Counts()[0][0]["pgl"] == 1);
  REQUIRE(p.Counts()[0][0]["gle"] == 1);

  // herpagkle: her, erp, rpa, pag, agk, gkl, kle
  REQUIRE(p.Counts()[0][1].size() == 7);
  REQUIRE(p.Counts()[0][1]["her"] == 1);
  REQUIRE(p.Counts()[0][1]["erp"] == 1);
  REQUIRE(p.Counts()[0][1]["rpa"] == 1);
  REQUIRE(p.Counts()[0][1]["pag"] == 1);
  REQUIRE(p.Counts()[0][1]["agk"] == 1);
  REQUIRE(p.Counts()[0][1]["gkl"] == 1);
  REQUIRE(p.Counts()[0][1]["kle"] == 1);

  // klunktor: klu, lun, unk, nkt, kto, tor
  REQUIRE(p.Counts()[0][2].size() == 6);
  REQUIRE(p.Counts()[0][2]["klu"] == 1);
  REQUIRE(p.Counts()[0][2]["lun"] == 1);
  REQUIRE(p.Counts()[0][2]["unk"] == 1);
  REQUIRE(p.Counts()[0][2]["nkt"] == 1);
  REQUIRE(p.Counts()[0][2]["kto"] == 1);
  REQUIRE(p.Counts()[0][2]["tor"] == 1);

  // flibbynopple: fli lib ibb bby byn yno nop opp ppl ple
  REQUIRE(p.Counts()[0][3].size() == 10);
  REQUIRE(p.Counts()[0][3]["fli"] == 1);
  REQUIRE(p.Counts()[0][3]["lib"] == 1);
  REQUIRE(p.Counts()[0][3]["ibb"] == 1);
  REQUIRE(p.Counts()[0][3]["bby"] == 1);
  REQUIRE(p.Counts()[0][3]["byn"] == 1);
  REQUIRE(p.Counts()[0][3]["yno"] == 1);
  REQUIRE(p.Counts()[0][3]["nop"] == 1);
  REQUIRE(p.Counts()[0][3]["opp"] == 1);
  REQUIRE(p.Counts()[0][3]["ppl"] == 1);
  REQUIRE(p.Counts()[0][3]["ple"] == 1);

  // floggy3245: flo log ogg ggy gy3 y32 324 245
  REQUIRE(p.Counts()[1][0].size() == 8);
  REQUIRE(p.Counts()[1][0]["flo"] == 1);
  REQUIRE(p.Counts()[1][0]["log"] == 1);
  REQUIRE(p.Counts()[1][0]["ogg"] == 1);
  REQUIRE(p.Counts()[1][0]["ggy"] == 1);
  REQUIRE(p.Counts()[1][0]["gy3"] == 1);
  REQUIRE(p.Counts()[1][0]["y32"] == 1);
  REQUIRE(p.Counts()[1][0]["324"] == 1);
  REQUIRE(p.Counts()[1][0]["245"] == 1);

  // flippydopflip: fli lip ipp ppy pyd ydo dop opf pfl fli lip
  // fli(2) lip(2) ipp ppy pyd ydo dop opf pfl
  REQUIRE(p.Counts()[1][1].size() == 9);
  REQUIRE(p.Counts()[1][1]["fli"] == 2);
  REQUIRE(p.Counts()[1][1]["lip"] == 2);
  REQUIRE(p.Counts()[1][1]["ipp"] == 1);
  REQUIRE(p.Counts()[1][1]["ppy"] == 1);
  REQUIRE(p.Counts()[1][1]["pyd"] == 1);
  REQUIRE(p.Counts()[1][1]["ydo"] == 1);
  REQUIRE(p.Counts()[1][1]["dop"] == 1);
  REQUIRE(p.Counts()[1][1]["opf"] == 1);
  REQUIRE(p.Counts()[1][1]["pfl"] == 1);

  // stupid fricking cat: stu tup upi pid fri ric ick cki kin ing cat
  REQUIRE(p.Counts()[1][2].size() == 11);
  REQUIRE(p.Counts()[1][2]["stu"] == 1);
  REQUIRE(p.Counts()[1][2]["tup"] == 1);
  REQUIRE(p.Counts()[1][2]["upi"] == 1);
  REQUIRE(p.Counts()[1][2]["pid"] == 1);
  REQUIRE(p.Counts()[1][2]["fri"] == 1);
  REQUIRE(p.Counts()[1][2]["ric"] == 1);
  REQUIRE(p.Counts()[1][2]["ick"] == 1);
  REQUIRE(p.Counts()[1][2]["cki"] == 1);
  REQUIRE(p.Counts()[1][2]["kin"] == 1);
  REQUIRE(p.Counts()[1][2]["ing"] == 1);
  REQUIRE(p.Counts()[1][2]["cat"] == 1);

  // food time isn't until later: foo ood tim ime isn unt nti til lat ate ter
  REQUIRE(p.Counts()[1][3].size() == 11);
  REQUIRE(p.Counts()[1][3]["foo"] == 1);
  REQUIRE(p.Counts()[1][3]["ood"] == 1);
  REQUIRE(p.Counts()[1][3]["tim"] == 1);
  REQUIRE(p.Counts()[1][3]["ime"] == 1);
  REQUIRE(p.Counts()[1][3]["isn"] == 1);
  REQUIRE(p.Counts()[1][3]["unt"] == 1);
  REQUIRE(p.Counts()[1][3]["nti"] == 1);
  REQUIRE(p.Counts()[1][3]["til"] == 1);
  REQUIRE(p.Counts()[1][3]["lat"] == 1);
  REQUIRE(p.Counts()[1][3]["ate"] == 1);
  REQUIRE(p.Counts()[1][3]["ter"] == 1);

  // leave me alone until 6:00: lea eav ave alo lon one unt nti til
  REQUIRE(p.Counts()[1][4].size() == 9);
  REQUIRE(p.Counts()[1][4]["lea"] == 1);
  REQUIRE(p.Counts()[1][4]["eav"] == 1);
  REQUIRE(p.Counts()[1][4]["ave"] == 1);
  REQUIRE(p.Counts()[1][4]["alo"] == 1);
  REQUIRE(p.Counts()[1][4]["lon"] == 1);
  REQUIRE(p.Counts()[1][4]["one"] == 1);
  REQUIRE(p.Counts()[1][4]["unt"] == 1);
  REQUIRE(p.Counts()[1][4]["nti"] == 1);
  REQUIRE(p.Counts()[1][4]["til"] == 1);

  // only after that do you get any food.:
  // onl nly aft fte ter tha hat you get any foo ood
  REQUIRE(p.Counts()[1][5].size() == 12);
  REQUIRE(p.Counts()[1][5]["onl"] == 1);
  REQUIRE(p.Counts()[1][5]["nly"] == 1);
  REQUIRE(p.Counts()[1][5]["aft"] == 1);
  REQUIRE(p.Counts()[1][5]["fte"] == 1);
  REQUIRE(p.Counts()[1][5]["ter"] == 1);
  REQUIRE(p.Counts()[1][5]["tha"] == 1);
  REQUIRE(p.Counts()[1][5]["hat"] == 1);
  REQUIRE(p.Counts()[1][5]["you"] == 1);
  REQUIRE(p.Counts()[1][5]["get"] == 1);
  REQUIRE(p.Counts()[1][5]["any"] == 1);
  REQUIRE(p.Counts()[1][5]["foo"] == 1);
  REQUIRE(p.Counts()[1][5]["ood"] == 1);

  // obloblobloblobloblobloblob: obl(8) blo(8) lob(8)
  REQUIRE(p.Counts()[1][6].size() == 3);
  REQUIRE(p.Counts()[1][6]["obl"] == 8);
  REQUIRE(p.Counts()[1][6]["blo"] == 8);
  REQUIRE(p.Counts()[1][6]["lob"] == 8);
}

TEST_CASE("PSpectrumStringEvaluateTest", "[KernelTest]")
{
  // Construct simple dataset.
  std::vector<std::vector<std::string> > dataset;
  dataset.push_back(std::vector<std::string>());
  dataset[0].push_back("hello");
  dataset[0].push_back("jello");
  dataset[0].push_back("mellow");
  dataset[0].push_back("mellow jello");

  PSpectrumStringKernel p(dataset, 3);

  arma::vec a("0 0");
  arma::vec b("0 0");

  REQUIRE(p.Evaluate(a, b) == Approx(3.0).epsilon(1e-7));
  REQUIRE(p.Evaluate(b, a) == Approx(3.0).epsilon(1e-7));

  b = "0 1";
  REQUIRE(p.Evaluate(a, b) == Approx(2.0).epsilon(1e-7));
  REQUIRE(p.Evaluate(b, a) == Approx(2.0).epsilon(1e-7));

  b = "0 2";
  REQUIRE(p.Evaluate(a, b) == Approx(2.0).epsilon(1e-7));
  REQUIRE(p.Evaluate(b, a) == Approx(2.0).epsilon(1e-7));

  b = "0 3";
  REQUIRE(p.Evaluate(a, b) == Approx(4.0).epsilon(1e-7));
  REQUIRE(p.Evaluate(b, a) == Approx(4.0).epsilon(1e-7));

  a = "0 1";
  b = "0 1";
  REQUIRE(p.Evaluate(a, b) == Approx(3.0).epsilon(1e-7));
  REQUIRE(p.Evaluate(b, a) == Approx(3.0).epsilon(1e-7));

  b = "0 2";
  REQUIRE(p.Evaluate(a, b) == Approx(2.0).epsilon(1e-7));
  REQUIRE(p.Evaluate(b, a) == Approx(2.0).epsilon(1e-7));

  b = "0 3";
  REQUIRE(p.Evaluate(a, b) == Approx(5.0).epsilon(1e-7));
  REQUIRE(p.Evaluate(b, a) == Approx(5.0).epsilon(1e-7));

  a = "0 2";
  b = "0 2";
  REQUIRE(p.Evaluate(a, b) == Approx(4.0).epsilon(1e-7));
  REQUIRE(p.Evaluate(b, a) == Approx(4.0).epsilon(1e-7));

  b = "0 3";
  REQUIRE(p.Evaluate(a, b) == Approx(6.0).epsilon(1e-7));
  REQUIRE(p.Evaluate(b, a) == Approx(6.0).epsilon(1e-7));

  a = "0 3";
  REQUIRE(p.Evaluate(a, b) == Approx(11.0).epsilon(1e-7));
  REQUIRE(p.Evaluate(b, a) == Approx(11.0).epsilon(1e-7));
}

/**
 * Cauchy Kernel test.
 */
TEST_CASE("CauchyKernelTest", "[KernelTest]")
{
  arma::vec a = "0 0 1";
  arma::vec b = "0 1 0";

  CauchyKernel ck(5.0);
  REQUIRE(ck.Evaluate(a, b) == Approx(0.92592588).epsilon(1e-7));
  REQUIRE(ck.Evaluate(b, a) == Approx(0.92592588).epsilon(1e-7));
}
