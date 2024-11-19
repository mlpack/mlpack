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

namespace mlpack {

/**
 * Default bootstrap strategy that uses the ::Bootstrap() function. 
 */
class DefaultBootstrap
{
 public:
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
};

/**
 * No-op bootstrap strategy that just copies the input to the output.
 */
class IdentityBootstrap
{
 public:
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
    MakeAlias(bootstrapDataset, dataset, dataset.n_rows, dataset.n_cols);
    MakeAlias(bootstrapLabels, labels, labels.n_elem);
    if (UseWeights)
      MakeAlias(bootstrapWeights, weights, weights.n_elem);
  }
};

/**
 * Sequential bootstrap.
 */
template<typename MatType = arma::mat>
class SequentialBootstrap
{
 public:
  explicit SequentialBootstrap(const MatType& indicatorMatrix) :
    SequentialBootstrap(indicatorMatrix, indicatorMatrix.n_rows)
  {
  }

  SequentialBootstrap(const MatType& indicatorMatrix,
                      arma::uword sampleCount):
    indicatorMatrix(indicatorMatrix),
    colIndices(arma::linspace<arma::uvec>(0, indicatorMatrix.n_cols - 1)),
    sampleCount(sampleCount)
  {
  }

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

  /**
   * Compute the samples of the next draw.
   * 
   * @return A list of indices referring to the observations that should
   *         be sampled.
   */
  arma::uvec ComputeSamples() const
  {
    std::vector<arma::u64> phi;

    phi.reserve(sampleCount);
    while (phi.size() < sampleCount)
    {
      const arma::vec prob(ComputeNextDrawProbabilities(phi,
                                                        indicatorMatrix));
      DiscreteDistribution d =
          DiscreteDistribution(std::vector<arma::vec>(1u, prob));

      phi.push_back(colIndices[d.Random()[0]]);
    }

    return arma::conv_to<arma::uvec>::from(phi);
  }
 
  /**
   * Compute the average uniqueness of each event at any sampling time point.
   * The average uniqueness is a measure for how isolated an event is during
   * its lifetime from other events.
   *
   * @param[in] indicatorMatrix Is a sparse matrix with ones where an event is
   *                            active and zeros else. Each row is a event,
   *                            each column is a time point.
   * @return The average uniqueness of the events in @p indicatorMatrix.
   */
  static arma::vec ComputeAverageUniqueness(
      const MatType& indicatorMatrix)
  {
    // sum of each column
    arma::rowvec concurrency(arma::sum(indicatorMatrix, 0));

    return (arma::sum(indicatorMatrix.each_row()
      / concurrency.clamp(1.0,
        std::numeric_limits<arma::rowvec::elem_type>::max()), 1) /
          arma::sum(indicatorMatrix, 1)); // mean of each row.
  }

  /**
   * Compute probabilities of the next draw for each observation.
   * 
   * @param[in] phi List of previously drawn observations.
   * @param[in] indicatorMatrix See
   *                            SequentialBootstrap::ComputeAverageUniqueness.
   * @return The probabilities for each observation to be drawn in the next
   *         iteration.
   */
  static arma::vec ComputeNextDrawProbabilities(
      const std::vector<arma::u64>& phi,
      const arma::mat& indicatorMatrix)
  {
    arma::vec avg(indicatorMatrix.n_rows);
    arma::uvec rows(arma::conv_to<arma::uvec>::from(phi));

    rows.insert_rows(rows.n_rows, 1);
    for (arma::uword i(0); i < avg.size(); ++i) {
      rows.back() = i;
      avg[i] = ComputeAverageUniqueness(indicatorMatrix.rows(rows)).back();
    }

    return avg / arma::sum(avg);
  }

 private:
  const MatType indicatorMatrix;
  const arma::uvec colIndices;
  const arma::uword sampleCount;
};

} // namespace mlpack

#endif
