/**
 * @file methods/random_forest/bootstrap.hpp
 * @author Ryan Curtin
 *
 * Implementation of the Bootstrap() function, which creates a bootstrapped
 * dataset from the given input dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANDOM_FOREST_BOOTSTRAP_HPP
#define MLPACK_METHODS_RANDOM_FOREST_BOOTSTRAP_HPP

#include <random>

namespace mlpack {

/**
 * Given a dataset, create another dataset via bootstrap sampling, with labels.
 */
template<bool UseWeights,
         typename MatType,
         typename LabelsType,
         typename WeightsType>
void Bootstrap(const MatType& dataset,
               const LabelsType& labels,
               const WeightsType& weights,
               MatType& bootstrapDataset,
               LabelsType& bootstrapLabels,
               WeightsType& bootstrapWeights)
{
  bootstrapDataset.set_size(dataset.n_rows, dataset.n_cols);
  bootstrapLabels.set_size(labels.n_elem);
  if (UseWeights)
    bootstrapWeights.set_size(weights.n_elem);

  // Random sampling with replacement.
  arma::uvec indices = randi<arma::uvec>(dataset.n_cols,
      arma::distr_param(0, dataset.n_cols - 1));
  bootstrapDataset = dataset.cols(indices);
  bootstrapLabels = labels.cols(indices);
  if (UseWeights)
    bootstrapWeights = weights.cols(indices);
}

/**
 * Default bootstrap strategy that uses the ::Bootstrap() function. 
 */
template<bool UseWeights>
class DefaultBootstrap {
public:
  template<typename MatType,
           typename LabelsType,
           typename WeightsType>
  void Bootstrap(const MatType& dataset,
                 const LabelsType& labels,
                 const WeightsType& weights,
                 MatType& bootstrapDataset,
                 LabelsType& bootstrapLabels,
                 WeightsType& bootstrapWeights)
  {
    mlpack::Bootstrap<UseWeights>(dataset,
                                  labels,
                                  weights,
                                  bootstrapDataset,
                                  bootstrapLabels,
                                  bootstrapWeights);
  }
};

/**
 * No-op bootstrap strategy that just copies the input to the output.
 */
template<bool UseWeights>
class IdentityBootstrap {
public:
  template<typename MatType,
           typename LabelsType,
           typename WeightsType>
  void Bootstrap(const MatType& dataset,
                 const LabelsType& labels,
                 const WeightsType& weights,
                 MatType& bootstrapDataset,
                 LabelsType& bootstrapLabels,
                 WeightsType& bootstrapWeights)
  {
    bootstrapDataset = dataset;
    bootstrapLabels = labels;
    if (UseWeights)
      bootstrapWeights = weights;
  }
};

/**
 * Sequential bootstrap.
 */
template<bool UseWeights, typename G = std::mt19937>
class SequentialBootstrap {
public:
  explicit SequentialBootstrap(const arma::mat& indicatorMatrix,
                               const G& generator = G(std::random_device()())) :
    SequentialBootstrap(indicatorMatrix, indicatorMatrix.n_rows, generator)
  {
  }

  explicit SequentialBootstrap(const arma::mat& indicatorMatrix,
                               arma::uword sampleCount,
                               const G& generator = G(std::random_device()())):
    indicatorMatrix(indicatorMatrix),
    colIndices(arma::linspace<arma::uvec>(0, indicatorMatrix.n_cols - 1)),
    sampleCount(sampleCount),
    generator(generator)
  {
  }

  template<typename MatType,
           typename LabelsType,
           typename WeightsType>
  void Bootstrap(const MatType& dataset,
                 const LabelsType& labels,
                 const WeightsType& weights,
                 MatType& bootstrapDataset,
                 LabelsType& bootstrapLabels,
                 WeightsType& bootstrapWeights)
  {
    if (labels.n_cols != indicatorMatrix.n_rows)
      throw std::invalid_argument("SequentialBootstrap::Bootstrap(): "
          "labels and indicatorMatrix n_rows differ!");

    // observations are stored as columns and dimensions 
    // (number of features) as rows.

    const arma::uvec phi(ComputeSamples());

    bootstrapDataset = dataset.cols(phi);
    bootstrapLabels = labels.cols(phi);
    if (UseWeights)
      bootstrapWeights = weights.rows(phi);
  }

  arma::uvec ComputeSamples() const
  {
    arma::uvec phi;

    while (phi.size() < sampleCount) {
      const auto prob(ComputeNextDrawProbabilities(phi, indicatorMatrix));
      std::discrete_distribution dist(prob.begin(), prob.end());

      phi = arma::join_cols(phi, arma::uvec{ colIndices[dist(generator)] });
    }

    return phi;
  }

  static arma::vec ComputeAverageUniqueness(const arma::mat& indicatorMatrix)
  {
    arma::rowvec concurrency(arma::sum(indicatorMatrix, 0)); // sum of each column
    auto uniqueness(
      indicatorMatrix.each_row()
      / concurrency.clamp(1.0,
        std::numeric_limits<arma::rowvec::elem_type>::max()));
    const auto n(arma::sum(indicatorMatrix, 1)); // sum of each row

    return arma::sum(uniqueness, 1) / n; // mean of each row
  }

  static arma::vec ComputeNextDrawProbabilities(const arma::uvec& phi,
                                                const arma::mat& indicatorMatrix)
  {
    arma::vec avg(indicatorMatrix.n_rows);
    arma::uvec rows(phi);

    rows.insert_rows(rows.n_rows, 1);
    for (arma::uword i(0); i < avg.size(); ++i) {
      rows.back() = i;
      avg[i] = ComputeAverageUniqueness(indicatorMatrix.rows(rows)).back();
    }

    return avg / arma::sum(avg);
  }

private:
  const arma::mat indicatorMatrix;
  const arma::uvec colIndices;
  const arma::uword sampleCount;
  G generator;
};

} // namespace mlpack

#endif
