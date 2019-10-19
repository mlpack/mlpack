/**
 * @file mock_categorical_data.hpp
 *
 * Generate categorical dataset for tests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_MOCK_CATEGORICAL_DATA_HPP
#define MLPACK_TESTS_MOCK_CATEGORICAL_DATA_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/dists/discrete_distribution.hpp>

/**
 * Create a mock categorical dataset for testing.
 */
inline void MockCategoricalData(arma::mat& d,
                                arma::Row<size_t>& l,
                                mlpack::data::DatasetInfo& datasetInfo)
{
  // We'll build a spiral dataset plus two noisy categorical features.  We need
  // to build the distributions for the categorical features (they'll be
  // discrete distributions).
  mlpack::distribution::DiscreteDistribution c1[5];
  // The distribution will be automatically normalized.
  for (size_t i = 0; i < 5; ++i)
  {
    std::vector<arma::vec> probs;
    probs.push_back(arma::vec(4, arma::fill::randu));
    c1[i] = mlpack::distribution::DiscreteDistribution(probs);
  }

  mlpack::distribution::DiscreteDistribution c2[5];
  for (size_t i = 0; i < 5; ++i)
  {
    std::vector<arma::vec> probs;
    probs.push_back(arma::vec(2, arma::fill::randu));
    c2[i] = mlpack::distribution::DiscreteDistribution(probs);
  }

  arma::mat spiralDataset(4, 4000);
  arma::Row<size_t> labels(4000);
  for (size_t i = 0; i < 4000; ++i)
  {
    // One circle every 2000 samples.  Plus some noise.
    const double magnitude = 2.0 + (double(i) / 200.0) +
        0.5 * mlpack::math::Random();
    const double angle = (i % 200) * (2 * M_PI) + mlpack::math::Random();

    const double x = magnitude * cos(angle);
    const double y = magnitude * sin(angle);

    spiralDataset(0, i) = x;
    spiralDataset(1, i) = y;

    // Set categorical features c1 and c2.
    if (i < 800)
    {
      spiralDataset(2, i) = c1[1].Random()[0];
      spiralDataset(3, i) = c2[1].Random()[0];
      labels[i] = 1;
    }
    else if (i < 1600)
    {
      spiralDataset(2, i) = c1[3].Random()[0];
      spiralDataset(3, i) = c2[3].Random()[0];
      labels[i] = 3;
    }
    else if (i < 2400)
    {
      spiralDataset(2, i) = c1[2].Random()[0];
      spiralDataset(3, i) = c2[2].Random()[0];
      labels[i] = 2;
    }
    else if (i < 3200)
    {
      spiralDataset(2, i) = c1[0].Random()[0];
      spiralDataset(3, i) = c2[0].Random()[0];
      labels[i] = 0;
    }
    else
    {
      spiralDataset(2, i) = c1[4].Random()[0];
      spiralDataset(3, i) = c2[4].Random()[0];
      labels[i] = 4;
    }
  }

  // Now create the dataset info.
  datasetInfo = mlpack::data::DatasetInfo(4);
  datasetInfo.Type(2) = mlpack::data::Datatype::categorical;
  datasetInfo.Type(3) = mlpack::data::Datatype::categorical;
  // Set mappings.
  datasetInfo.MapString<double>("0", 2);
  datasetInfo.MapString<double>("1", 2);
  datasetInfo.MapString<double>("2", 2);
  datasetInfo.MapString<double>("3", 2);
  datasetInfo.MapString<double>("0", 3);
  datasetInfo.MapString<double>("1", 3);

  // Now shuffle the dataset.
  arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0, 3999,
      4000));
  d = arma::mat(4, 4000);
  l = arma::Row<size_t>(4000);
  for (size_t i = 0; i < 4000; ++i)
  {
    d.col(i) = spiralDataset.col(indices[i]);
    l[i] = labels[indices[i]];
  }
}

#endif
