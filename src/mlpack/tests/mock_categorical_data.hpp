/**
 * @file tests/mock_categorical_data.hpp
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
 * Create a mock categorical dataset for testing classification.
 */
inline void MockCategoricalData(arma::mat& d,
                                arma::Row<size_t>& l,
                                mlpack::data::DatasetInfo& datasetInfo)
{
  // We'll build a spiral dataset plus two noisy categorical features.  We need
  // to build the distributions for the categorical features (they'll be
  // discrete distributions).
  mlpack::DiscreteDistribution c1[5];
  // The distribution will be automatically normalized.
  for (size_t i = 0; i < 5; ++i)
  {
    std::vector<arma::vec> probs;
    probs.push_back(arma::vec(4, arma::fill::randu));
    c1[i] = mlpack::DiscreteDistribution(probs);
  }

  mlpack::DiscreteDistribution c2[5];
  for (size_t i = 0; i < 5; ++i)
  {
    std::vector<arma::vec> probs;
    probs.push_back(arma::vec(2, arma::fill::randu));
    c2[i] = mlpack::DiscreteDistribution(probs);
  }

  arma::mat spiralDataset(4, 4000);
  arma::Row<size_t> labels(4000);
  for (size_t i = 0; i < 4000; ++i)
  {
    // One circle every 2000 samples.  Plus some noise.
    const double magnitude = 2.0 + (double(i) / 200.0) + 0.5 * mlpack::Random();
    const double angle = (i % 200) * (2 * M_PI) + mlpack::Random();

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

/**
 * Create a mock categorical dataset for testing regression.
 */
inline void MockCategoricalData(arma::mat& d,
                                arma::Row<double>& l,
                                mlpack::data::DatasetInfo& datasetInfo)
{
  // Dataset of size 4000.
  d.set_size(5, 4000);
  l.set_size(4000);

  for (size_t i = 0; i < 4000; ++i)
  {
    // Random numeric features.
    d(0, i) = mlpack::Random();
    d(1, i) = mlpack::Random(-1, 1);
    d(2, i) = mlpack::Random();

    // Binary feature.
    d(3, i) = mlpack::RandInt(0, 2);
    // 5-category categorical feature.
    d(4, i) = mlpack::RandInt(0, 5);

    // Mappings from categorical features to regression value.
    std::map<int, double> f;
    f[0] = 5.0;
    f[1] = -5.0;

    std::map<int, double> g;
    g[0] = 2.0;
    g[1] = 7.0;
    g[2] = -3.0;
    g[3] = 0.0;
    g[4] = 4.0;

    // Random noise in range [-0.5, 0.5).
    const double noise = mlpack::Random() - 0.5;

    // y = x1 + x2 + 3 * x3 + f(x4) + g(x5) + noise
    l[i] = d(0, i) + d(1, i) + 3 * d(2, i) + f[(int) d(3, i)] +
        g[(int) d(4, i)] + noise;
  }

  // Now create the dataset info.
  datasetInfo = mlpack::data::DatasetInfo(5);
  datasetInfo.Type(3) = mlpack::data::Datatype::categorical;
  datasetInfo.Type(4) = mlpack::data::Datatype::categorical;
  // Set mappings.
  datasetInfo.MapString<double>("0", 3);
  datasetInfo.MapString<double>("1", 3);

  datasetInfo.MapString<double>("0", 4);
  datasetInfo.MapString<double>("1", 4);
  datasetInfo.MapString<double>("2", 4);
  datasetInfo.MapString<double>("3", 4);
  datasetInfo.MapString<double>("4", 4);
}

#endif
