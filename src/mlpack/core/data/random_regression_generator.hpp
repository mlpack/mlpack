/**
 * @file random_regression_generator.hpp
 * @author Ali Hossam
 *
 * Implementation of a regression data generator with random features and error 
 * distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_DATA_RANDOM_REGRESSION_GENERATOR_HPP
#define MLPACK_CORE_DATA_RANDOM_REGRESSION_GENERATOR_HPP

#include <mlpack/core.hpp>
#include "mlpack/core/dists/dists.hpp"

namespace mlpack {
namespace data {

/**
 * @brief Class for generating linear regression data.
 *
 * This class provides a static function to generate synthetic regression data
 * with specified parameters.
 */
template<typename NoiseDistType = GaussianDistribution>
class RegressionDataGenerator 
{
 public:
  /**
   * @brief Constructor for RegressionDataGenerator.
   *
   * @param nSamples Number of samples.
   * @param nFeatures Number of features.
   * @param NoiseDistRule Noise distribution rule.
   * @param nTargets Number of target variables.
   * @param intercept Bias term.
   * @param sparsity Sparsity level of coefficients.
   * @param outliersFraction Fraction of samples with outliers.
   * @param outliersScale Scale of outliers.
   * @param randomSeed Seed for random number generation.
   */
  RegressionDataGenerator(
    int nSamples,
    int nFeatures,
    NoiseDistType NoiseDistRule = NoiseDistType(arma::vec("0"), arma::vec("1")),
    int nTargets = 1,
    float intercept = 0.0,
    float sparsity = 0,
    float outliersFraction = 0,
    float outliersScale = 10,
    int randomSeed = 0) : 
      nSamples(nSamples),
      nFeatures(nFeatures),
      NoiseDistRule(NoiseDistRule),
      nTargets(nTargets),
      intercept(intercept),
      sparsity(sparsity),
      outliersFraction(outliersFraction),
      outliersScale(outliersScale),
      randomSeed(randomSeed)
  {
    // Validation logic for input parameters
    if (nSamples <= 0 || nFeatures <= 0 || nTargets <= 0) 
    {
      throw std::invalid_argument("Invalid input: nSamples, nFeatures,"
          "and nTargets must be positive.");
    }

    if (sparsity < 0 || sparsity >= 1) 
    {
      throw std::invalid_argument("Invalid input: sparsity must be in" 
          "the range [0, 1).");
    }

    if (outliersFraction < 0 || outliersFraction >= 1) 
    {
      throw std::invalid_argument("Invalid input: outliersFraction must be"
          "in the range [0, 1).");
    }
  }

  /**
   * @brief Generate synthetic regression data.
   *
   * This function generates synthetic regression data with specified 
   * parameters.
   *
   * @param X Matrix to store the generated features.
   * @param y Matrix to store the generated response variable.
   */
  template<typename MatType>
  void GenerateData(MatType& X, MatType& y) 
  {
    // Set the seed if needed
    if (randomSeed)
      arma::arma_rng::set_seed(randomSeed);
    else
      arma::arma_rng::set_seed_random();

    // Generate random features with normal distribution
    X.randn(nSamples, nFeatures);

    // Generate coefficients with sparsity
    MatType coeff = arma::conv_to<MatType>::from(
        arma::mat(arma::sprandn<arma::sp_mat>(nFeatures,
                                              nTargets, 
                                              1 - sparsity)));

    y = X * coeff + intercept;

    // Add Noise
    MatType error(y.n_rows, y.n_cols);
    for (size_t j = 0; j < y.n_rows; ++j)
      for (size_t i = 0; i < y.n_cols; ++i)
        error(j, i) = NoiseDistRule.Random()[0];

    y += error;

    // Add outliers
    size_t numOutliers = static_cast<size_t>(outliersFraction * nSamples);
    arma::uvec outliersIndices = arma::randi<arma::uvec>(
        numOutliers,
        arma::distr_param(0, nSamples - 1));

    MatType outliersValues = arma::randn<MatType>(numOutliers) * outliersScale;

    y(outliersIndices) += outliersValues;

    // Transpose data (required for row-vector y in regression models)
    X = X.t();
    y = y.t();
  }

  template<typename Archive>
  void Serialize(Archive& ar, const uint32_t /* version */)
  {
    // We just need to serialize each of the members.
    ar(CEREAL_NVP(nSamples));
    ar(CEREAL_NVP(nFeatures));
    ar(CEREAL_NVP(NoiseDistRule));
    ar(CEREAL_NVP(nTargets));
    ar(CEREAL_NVP(intercept));
    ar(CEREAL_NVP(sparsity));
    ar(CEREAL_NVP(outliersFraction));
    ar(CEREAL_NVP(outliersScale));
    ar(CEREAL_NVP(randomSeed));
  }

 private:
  int nSamples;
  int nFeatures;
  NoiseDistType NoiseDistRule;  
  int nTargets;
  float intercept;
  float sparsity;
  float outliersFraction;
  float outliersScale;
  int randomSeed;
};

} // namespace data
} // namespace mlpack

#endif