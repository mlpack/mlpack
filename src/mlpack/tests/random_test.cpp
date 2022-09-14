/**
 * @file tests/random_test.cpp
 * @author Konstantin Sidorov
 *
 * Tests for generators of random numbers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include "catch.hpp"

using namespace mlpack;

// Test for RandInt() sampler from discrete uniform distribution.
TEST_CASE("DiscreteUniformRandomTest", "[RandomTest]")
{
  std::vector<std::pair<int, int>> ranges =
  {
    std::make_pair(0, 1),
    std::make_pair(0, 2),
    std::make_pair(-6, -2),
    std::make_pair(-3, 4),
    std::make_pair(13, 37)
  };
  const size_t iterations = 10000;
  for (std::pair<int, int> range : ranges)
  {
    int lo = range.first, hiExclusive = range.second;
    std::vector<int> count(hiExclusive - lo, 0);
    for (size_t iter = 0; iter < iterations; ++iter)
    {
      count[RandInt(lo, hiExclusive) - lo]++;
    }
    for (size_t i = 0; i < count.size(); ++i)
    {
      REQUIRE(1.0 / (hiExclusive - lo) - count[i] * 1.0 / iterations ==
          Approx(0.0).margin(0.15));
    }
  }

  // Here we also test RandInt(hiExclusive) overload.
  for (std::pair<int, int> range : ranges)
  {
    int lo = range.first, hiExclusive = range.second;
    if (lo != 0) continue;
    std::vector<int> count(hiExclusive - lo, 0);
    for (size_t iter = 0; iter < iterations; ++iter)
    {
      count[RandInt(lo, hiExclusive) - lo]++;
    }

    for (size_t i = 0; i < count.size(); ++i)
    {
      REQUIRE(1.0 / (hiExclusive - lo) - count[i] * 1.0 / iterations ==
          Approx(0.0).margin(0.15));
    }
  }
}

// Test for RandInt() sampler from discrete (possibly nonuniform) distribution.
TEST_CASE("WeightedRandomTest", "[RandomTest]")
{
  std::vector<std::vector<double>> weights = {
    {1},
    {0, 0, 1, 0, 0},
    {1, 0, 0},
    {0, 0, 1},
    {0.25, 0.25, 0.5},
    {0.9, 0.05, 0.05},
    {0.5, 0.1, 0.1, 0.1, 0.1, 0.1}
  };
  const size_t iterations = 50000;
  for (std::vector<double> weightSet : weights)
  {
    DiscreteDistribution d(1);
    d.Probabilities(0) =  arma::vec(weightSet);
    std::vector<int> count(weightSet.size(), 0);
    for (size_t iter = 0; iter < iterations; ++iter)
    {
      count[d.Random()(0)]++;
    }

    for (size_t i = 0; i < weightSet.size(); ++i)
    {
      REQUIRE(weightSet[i] - count[i] * 1.0 / iterations ==
          Approx(0.0).margin(0.15));
    }
  }
}
