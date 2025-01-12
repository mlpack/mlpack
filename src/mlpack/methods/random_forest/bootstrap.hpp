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
   */
  SequentialBootstrap(const IndMatType& intervals) :
    intervals(intervals)
  {
    if (intervals.n_rows != 2)
      throw std::invalid_argument(
        "SequentialBootstrap::SequentialBootstrap(): "
        "intervals must be a 2 x m matrix!");
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
    if (labels.n_cols != intervals.n_cols)
      throw std::invalid_argument("SequentialBootstrap::Bootstrap(): "
        "labels n_cols and intervals n_cols differ!");

    // observations are stored as columns and dimensions
    // (number of features) as rows.
    const arma::uvec phi(arma::conv_to<arma::uvec>::from(
      ComputeSamples(dataset.n_cols)));

    bootstrapDataset = dataset.cols(phi);
    bootstrapLabels = labels.cols(phi);
    if (UseWeights)
      bootstrapWeights = weights.rows(phi);
  }

  /**
   * Compute the samples of the next draw.
   *
   * @param[in] colCount Number of data points in the dataset.
   *
   * @return A list of indices referring to the observations that should
   *         be sampled.
   */
  arma::uvec ComputeSamples(arma::uword colCount) const
  {
    DiscreteDistribution     d;
    arma::uvec               phi(colCount);

    for (arma::uword i(0); i < colCount; ++i) {
      d.Probabilities() = ComputeNextDrawProbabilities(phi, i, intervals);
      phi[i] = d.Random()[0];
    }

    return phi;
  }

  /**
   * Compute the average uniqueness of each event at any sampling time point.
   * The average uniqueness is a measure for how isolated an event is during
   * its lifetime from other events.
   *
   * @param[in] intervals Is a `2 x m` matrix, where each of the m columns has
   *                      the start sample and the end sample of an interval.
   * @param[in] indices Indices of the average uniqueness that is returned.
   *                    This parameter is used for optimization purposes so
   *                    that in the unit tests the average uniqueness for
   *                    all events can be accessed while in the production
   *                    code only a single element is accessed.
   * @return The average uniqueness of the events in @p intervals.
   */
  static arma::vec ComputeAverageUniqueness(
    const IndMatType& intervals,
    arma::uvec        indices)
  {
    arma::vec avg(indices.n_rows, arma::fill::zeros);
    arma::vec concurrency(intervals.max() + 1, arma::fill::zeros);

    for (arma::uword i(0); i < intervals.n_cols; ++i) {
      concurrency.subvec(intervals(0, i), intervals(1, i)) += 1.0;
    }

    const arma::vec invConcurrency(
      arma::vec(concurrency.n_rows, arma::fill::ones) / concurrency);

    // In production code this loop is only entered once.
    for (arma::uword i(0); i < indices.n_rows; ++i) {
      const arma::uword start(intervals(0, indices[i]));
      const arma::uword end(intervals(1, indices[i]));

      avg[i] = arma::accu(invConcurrency.rows(start, end)) /
        (end - start + 1);
    }

    return avg;
  }

  /**
   * Compute probabilities of the next draw for each observation.
   *
   * @param[in] phi List of previously drawn observations.
   * @param[in] phiSize Number of elements in use in @p phi.
   * @param[in] intervals See ComputeAverageUniqueness().
   * @return The probabilities for each observation to be drawn in the next
   *         iteration.
   */
  static arma::vec ComputeNextDrawProbabilities(
    arma::uvec& phi,
    arma::uword       phiSize,
    const IndMatType& intervals)
  {
    assert(phi.n_rows <= intervals.n_cols && phiSize < phi.n_rows);

    arma::vec avg(intervals.n_cols);

    for (arma::uword i(0); i < avg.size(); ++i) {
      phi[phiSize] = i;
      avg[i] = ComputeAverageUniqueness(
        intervals.cols(phi.rows(0, phiSize)),
        arma::uvec(1u, arma::fill::value(phiSize))).back();
    }

    return avg / arma::sum(avg);
  }

 private:
  const IndMatType intervals;
};

} // namespace mlpack

#endif
