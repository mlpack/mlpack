/**
 * @file nystroem_method_test.cpp
 * @author Ryan Curtin
 *
 * Test the NystroemMethod class and ensure that the reconstructed kernel matrix
 * errors are comparable with those in the literature.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

#include <mlpack/methods/nystroem_method/ordered_selection.hpp>
#include <mlpack/methods/nystroem_method/random_selection.hpp>
#include <mlpack/methods/nystroem_method/kmeans_selection.hpp>
#include <mlpack/methods/nystroem_method/nystroem_method.hpp>

using namespace mlpack;
using namespace mlpack::kernel;

BOOST_AUTO_TEST_SUITE(NystroemMethodTest);

/**
 * Make sure that if the rank is the same and we do a full-rank approximation,
 * the result is virtually identical (a little bit of tolerance for floating
 * point error).
 */
BOOST_AUTO_TEST_CASE(FullRankTest)
{
  // Run several trials.
  for (size_t trial = 0; trial < 5; ++trial)
  {
    arma::mat data;
    data.randu(5, trial * 200);

    GaussianKernel gk;
    NystroemMethod<GaussianKernel, OrderedSelection> nm(data, gk, trial * 200);

    arma::mat g;
    nm.Apply(g);

    // Construct exact kernel matrix.
    arma::mat kernel(trial * 200, trial * 200);
    for (size_t i = 0; i < trial * 200; ++i)
      for (size_t j = 0; j < trial * 200; ++j)
        kernel(i, j) = gk.Evaluate(data.col(i), data.col(j));

    // Reconstruct approximation.
    arma::mat approximation = g * g.t();

    // Check closeness.
    for (size_t i = 0; i < trial * 200; ++i)
    {
      for (size_t j = 0; j < trial * 200; ++j)
      {
        if (kernel(i, j) < 1e-5)
          BOOST_REQUIRE_SMALL(approximation(i, j), 1e-4);
        else
          BOOST_REQUIRE_CLOSE(kernel(i, j), approximation(i, j), 1e-5);
      }
    }
  }
}

/**
 * Can we accurately represent a rank-10 matrix?
 */
BOOST_AUTO_TEST_CASE(Rank10Test)
{
  arma::mat data;
  data.randu(500, 500); // Just so it's square.

  // Use SVD and only keep the first ten singular vectors.
  arma::mat U;
  arma::vec s;
  arma::mat V;
  arma::svd(U, s, V, data);

  // Don't set completely to 0; the hope is that K is still positive definite.
  s.subvec(0, 9) += 1.0; // Make sure the first 10 singular vectors are large.
  s.subvec(10, s.n_elem - 1).fill(1e-6);
  arma::mat dataMod = U * arma::diagmat(s) * V.t();

  // Add some noise.
  dataMod += 1e-5 * arma::randu<arma::mat>(dataMod.n_rows, dataMod.n_cols);

  // Calculate the true kernel matrix.
  LinearKernel lk;
  arma::mat kernel = dataMod.t() * dataMod;

  // Now use the linear kernel to get a Nystroem approximation; try this several
  // times.
  double normalizedFroAverage = 0.0;
  for (size_t trial = 0; trial < 20; ++trial)
  {
    LinearKernel lk;
    NystroemMethod<LinearKernel, RandomSelection> nm(dataMod, lk, 10);

    arma::mat g;
    nm.Apply(g);

    arma::mat approximation = g * g.t();

    // Check the normalized Frobenius norm.
    const double normalizedFro = arma::norm(kernel - approximation, "fro") /
        arma::norm(kernel, "fro");

    normalizedFroAverage += normalizedFro;
  }

  normalizedFroAverage /= 20;
  BOOST_REQUIRE_SMALL(normalizedFroAverage, 1e-3);
}

/**
 * Can we reproduce the results in Zhang, Tsang, and Kwok (2008)?
 * They provide the following test points (approximately) in their experiments
 * in Section 4.1, for the german dataset:
 *
 *  rank = 0.02n; approximation error: ~27
 *  rank = 0.04n; approximation error: ~15
 *  rank = 0.06n; approximation error: ~10
 *  rank = 0.08n; approximation error: ~7
 *  rank = 0.10n; approximation error: ~3
 */
BOOST_AUTO_TEST_CASE(GermanTest)
{
  // Load the dataset.
  arma::mat dataset;
  data::Load("german.csv", dataset, true);

  // These are our tolerance bounds.
  double results[5] = { 32.0, 20.0, 15.0, 12.0, 9.0 };

  // The bandwidth of the kernel is selected to be the half the average
  // distance between each point and the mean of the dataset.  This isn't
  // _exactly_ what the paper says, but I've modified what it said because our
  // formulation of what the Gaussian kernel is is different.
  GaussianKernel gk(16.461);

  // Calculate the true kernel matrix.
  arma::mat kernel(dataset.n_cols, dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
    for (size_t j = 0; j < dataset.n_cols; ++j)
      kernel(i, j) = gk.Evaluate(dataset.col(i), dataset.col(j));

  for (size_t trial = 0; trial < 5; ++trial)
  {
    // We will repeat each trial 20 times.
    double avgError = 0.0;
    for (size_t z = 0; z < 20; ++z)
    {
      NystroemMethod<GaussianKernel, KMeansSelection<> > nm(dataset, gk,
          size_t((double((trial + 1) * 2) / 100.0) * dataset.n_cols));
      arma::mat g;
      nm.Apply(g);

      // Reconstruct kernel matrix.
      arma::mat approximation = g * g.t();

      const double error = arma::norm(kernel - approximation, "fro");
      if (error != error)
      {
        // Sometimes K' is singular.  Unlucky.
        --z;
        continue;
      }
      else
      {
        Log::Debug << "Trial " << trial << ": error " << error << ".\n";
        avgError += arma::norm(kernel - approximation, "fro");
      }
    }

    avgError /= 20;

    // Ensure that this is within tolerance, which is at least as good as the
    // paper's results (plus a little bit for noise).
    BOOST_REQUIRE_SMALL(avgError, results[trial]);
  }
}

BOOST_AUTO_TEST_SUITE_END();
