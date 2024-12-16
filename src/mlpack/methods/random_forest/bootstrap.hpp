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
 * Given a dataset, create another dataset via bootstrap sampling, with labels.
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
 *
 * @tparam IndMatType Indicator matrix type.
 */
template<typename IndMatType = arma::umat>
class SequentialBootstrap
{
public:
  /**
   * Constructor.
   *
   * @param[in] intervals See ComputeAverageUniqueness().
   * @param[in] colIndicesCount Number of data points in the dataset.
   */
  SequentialBootstrap(const IndMatType& intervals,
    arma::uword colIndicesCount) :
    intervals(intervals),
    sampleCount(colIndicesCount),
    colIndices(arma::linspace<arma::uvec>(0,
      colIndicesCount - 1,
      colIndicesCount))
  {
    if (intervals.n_cols != 2)
      throw std::invalid_argument(
        "SequentialBootstrap::SequentialBootstrap(): "
        "intervals must be a 2xm matrix!");
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
    if (labels.n_cols != intervals.n_rows)
      throw std::invalid_argument("SequentialBootstrap::Bootstrap(): "
        "labels n_cols and intervals n_rows differ!");

    // observations are stored as columns and dimensions
    // (number of features) as rows.
    if (colIndices.n_rows != dataset.n_cols) {
      throw std::invalid_argument("SequentialBootstrap::Bootstrap(): "
        "constructed with colIndicesCount different from "
        "dataset.n_cols!");
    }

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
        intervals, colIndices.n_elem));
      DiscreteDistribution d(
        DiscreteDistribution(std::vector<arma::vec>(1u, prob)));

      phi.push_back(colIndices[d.Random()[0]]);
    }

    return arma::conv_to<arma::uvec>::from(phi);
  }

  /**
   * Compute the average uniqueness of each event at any sampling time point.
   * The average uniqueness is a measure for how isolated an event is during
   * its lifetime from other events.
   *
   * @param[in] intervals Is a `2 x m` matrix, where each column has the start
   *                      sample and one past the end sample of an interval.
   * @param[in] from Start row to calculate the average uniqueness from.
   * @return The average uniqueness of the events in @p indicatorMatrix.
   */
  static arma::vec ComputeAverageUniqueness(
    const IndMatType& intervals,
    arma::uword       colIndicesCount,
    arma::uword       from)
  {
    arma::rowvec concurrency(colIndicesCount, arma::fill::zeros);

    for (arma::uword i(0); i < concurrency.n_cols; ++i) {
      for (arma::uword j(0); j < intervals.n_rows; ++j) {
        concurrency(i) += intervals(j, 0) <= i && i < intervals(j, 1) ? 1 : 0;
      }
    }

    arma::vec avg(intervals.n_rows - from, arma::fill::zeros);

    for (arma::uword i(from); i < intervals.n_rows; ++i) {
      for (arma::uword j(intervals(i, 0)); j < intervals(i, 1); ++j) {
        avg(i - from) += 1.0 / concurrency(j);
      }
      avg(i - from) /= intervals(i, 1) - intervals(i, 0);
    }

    return avg;
  }

  /**
   * Compute probabilities of the next draw for each observation.
   *
   * @param[in] phi List of previously drawn observations.
   * @param[in] intervals See ComputeAverageUniqueness().
   * @return The probabilities for each observation to be drawn in the next
   *         iteration.
   */
  static arma::vec ComputeNextDrawProbabilities(
    const std::vector<arma::u64>& phi,
    const IndMatType& intervals,
    arma::uword colIndicesCount)
  {
    arma::vec avg(intervals.n_rows);
    arma::uvec rows(arma::conv_to<arma::uvec>::from(phi));

    rows.insert_rows(rows.n_rows, 1);
    for (arma::uword i(0); i < avg.size(); ++i) {
      rows.back() = i;
      avg[i] = ComputeAverageUniqueness(intervals.rows(rows),
        colIndicesCount,
        rows.n_rows - 1).back();
    }

    return avg / arma::sum(avg);
  }

private:
  const IndMatType intervals;
  const arma::uword sampleCount;
  const arma::uvec colIndices;
};

} // namespace mlpack

#endif
