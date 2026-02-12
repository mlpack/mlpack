/**
 * @file tests/tsne_test.cpp
 * @author Kiner Shah
 *
 * Test file for TSNE class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/tsne.hpp>

#include "catch.hpp"

using namespace arma;
using namespace mlpack;

/**
 * Test that t-SNE reduces dimensionality correctly.
 */
TEST_CASE("TSNEBasicTest", "[TSNETest]")
{
  // Create a simple dataset with 100 points in 10 dimensions.
  arma::mat data = arma::randu<arma::mat>(10, 100);
  
  TSNE<> tsne(30.0, 200.0, 100); // Fewer iterations for testing
  arma::mat output;
  
  tsne.Apply(data, output, 2);
  
  // Check output dimensionality.
  REQUIRE(output.n_rows == 2);
  REQUIRE(output.n_cols == 100);
  
  // Check that output contains valid numbers (not NaN or Inf).
  REQUIRE(output.is_finite());
}

/**
 * Test that different perplexity values produce different results.
 */
TEST_CASE("TSNEPerplexityTest", "[TSNETest]")
{
  arma::mat data = arma::randu<arma::mat>(5, 50);
  
  // Run with perplexity 5
  TSNE<> tsne1(5.0, 100.0, 100);
  arma::mat output1;
  tsne1.Apply(data, output1, 2);
  
  // Run with perplexity 15
  TSNE<> tsne2(15.0, 100.0, 100);
  arma::mat output2;
  tsne2.Apply(data, output2, 2);
  
  // Results should be different (not exactly the same)
  double diff = arma::norm(output1 - output2, "fro");
  REQUIRE(diff > 1e-5);
}

/**
 * Test that the same random seed produces the same results.
 */
TEST_CASE("TSNEDeterministicTest", "[TSNETest]")
{
  arma::mat data = arma::randu<arma::mat>(5, 50);
  
  // Run twice with the same seed
 arma::arma_rng::set_seed(42);
  TSNE<> tsne1(10.0, 100.0, 100);
  arma::mat output1;
  tsne1.Apply(data, output1, 2);
  
  arma::arma_rng::set_seed(42);
  TSNE<> tsne2(10.0, 100.0, 100);
  arma::mat output2;
  tsne2.Apply(data, output2, 2);
  
  // Results should be very similar (within tolerance)
  REQUIRE(arma::approx_equal(output1, output2, "both", 1e-5, 1e-5));
}

/**
 * Test that t-SNE separates distinct clusters.
 */
TEST_CASE("TSNEClusterSeparationTest", "[TSNETest]")
{
  // Create two well-separated clusters in high dimensions
  arma::mat cluster1 = arma::randn<arma::mat>(10, 25);
  cluster1.each_col() += arma::vec(10, arma::fill::zeros); // Center at origin
  
  arma::mat cluster2 = arma::randn<arma::mat>(10, 25);
  arma::vec offset(10, arma::fill::ones);
  offset *= 10.0; // Large offset
  cluster2.each_col() += offset;
  
  arma::mat data = arma::join_rows(cluster1, cluster2);
  
  // Run t-SNE
  TSNE<> tsne(10.0, 200.0, 500);
  arma::mat output;
  tsne.Apply(data, output, 2);
  
  // Check that points from the same cluster are closer together
  // than points from different clusters
  arma::vec center1 = arma::mean(output.cols(0, 24), 1);
  arma::vec center2 = arma::mean(output.cols(25, 49), 1);
  
  double interClusterDist = arma::norm(center1 - center2);
  
  // Average intra-cluster distance for cluster 1
  double intraClusterDist1 = 0.0;
  for (size_t i = 0; i < 25; ++i)
  {
    intraClusterDist1 += arma::norm(output.col(i) - center1);
  }
  intraClusterDist1 /= 25.0;
  
  // Inter-cluster distance should be larger than intra-cluster distance
  REQUIRE(interClusterDist > intraClusterDist1);
}

/**
 * Test that early exaggeration affects the embedding.
 */
TEST_CASE("TSNEEarlyExaggerationTest", "[TSNETest]")
{
  arma::mat data = arma::randu<arma::mat>(5, 50);
  
  // Run with different early exaggeration values
  arma::arma_rng::set_seed(42);
  TSNE<> tsne1(10.0, 100.0, 200, 4.0);
  arma::mat output1;
  tsne1.Apply(data, output1, 2);
  
  arma::arma_rng::set_seed(42);
  TSNE<> tsne2(10.0, 100.0, 200, 12.0);
  arma::mat output2;
  tsne2.Apply(data, output2, 2);
  
  // Results should be different
  double diff = arma::norm(output1 - output2, "fro");
  REQUIRE(diff > 1e-3);
}

/**
 * Test t-SNE on a very small dataset.
 */
TEST_CASE("TSNESmallDatasetTest", "[TSNETest]")
{
  // Create a tiny dataset with just 10 points
  arma::mat data = arma::randu<arma::mat>(5, 10);
  
  TSNE<> tsne(3.0, 100.0, 100); // Low perplexity for small dataset
  arma::mat output;
  
  tsne.Apply(data, output, 2);
  
  REQUIRE(output.n_rows == 2);
  REQUIRE(output.n_cols == 10);
  REQUIRE(output.is_finite());
}

/**
 * Test Student t-distribution probability calculations.
 */
TEST_CASE("StudentTDistributionTest", "[TSNETest]")
{
  StudentTDistribution<> dist(1.0); // 1 degree of freedom (Cauchy)
  
  // Test univariate probability
  double prob = dist.Probability(0.0, 0.0);
  REQUIRE(prob > 0.0);
  REQUIRE(std::isfinite(prob));
  
  // Test that probability decreases with distance
  double prob1 = dist.Probability(0.0, 0.0);
  double prob2 = dist.Probability(1.0, 0.0);
  REQUIRE(prob1 > prob2);
}

/**
 * Test pairwise probabilities computation.
 */
TEST_CASE("StudentTPairwiseProbabilitiesTest", "[TSNETest]")
{
  StudentTDistribution<> dist(1.0);
  
  arma::mat squaredDistances(3, 3);
  squaredDistances << 0.0 << 1.0 << 4.0 << arma::endr
                   << 1.0 << 0.0 << 9.0 << arma::endr
                   << 4.0 << 9.0 << 0.0 << arma::endr;
  
  arma::mat probabilities;
  dist.PairwiseProbabilities(squaredDistances, probabilities);
  
  REQUIRE(probabilities.n_rows == 3);
  REQUIRE(probabilities.n_cols == 3);
  
  // Diagonal should have highest probabilities
  REQUIRE(probabilities(0, 0) > probabilities(0, 1));
  REQUIRE(probabilities(1, 1) > probabilities(1, 2));
}
