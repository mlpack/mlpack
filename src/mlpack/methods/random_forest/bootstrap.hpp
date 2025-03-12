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
        DistrParam(0, dataset.n_cols - 1));
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
 * This sequential bootstrap is suitable for time-series that are not
 * independent and identically distributed (IID). The bootstrap will
 * generate variations of the data that are more IID. In order to do
 * that it takes an indicator matrix that is a measure for how
 * much informational overlap exists between each data point in the
 * data set. Then it will randomly draw from the dataset with
 * replacement but reducing the likelihood to draw data points that
 * have high informational overlap with already drawn samples. The
 * algorithm assumes that events are active between a start and end
 * point. This allows for an efficient coding of the indicator matrix
 * as a `2 x n` matrix where each column has the start and end point
 * of the event.
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
   * @param[in] intervals Is a `2 x n` matrix, where each of the n columns has
   *                      the start sample and the end sample of an interval.
   *                      This matrix is a space-efficient form of the
   *                      indicator matrix.
   */
  SequentialBootstrap(const IndMatType& intervals) :
      intervals(intervals)
  {
    if (intervals.n_rows != 2)
      throw std::invalid_argument(
          "SequentialBootstrap::SequentialBootstrap(): "
          "intervals must be a 2 x n matrix!");
  }

  /**
   * Compute the average uniqueness of some events in an interval.
   * 
   * The average uniqueness is a measure for how isolated an event is during
   * its lifetime from other events.
   * @param[in] start Beginning of the interval (inclusive).
   * @param[in] end End of the interval (inclusive).
   * @param[in] invConcurrency Inverse concurrency of the samples.
   * @return The average uniqueness of the events in [@p start, @p end].
   */
  static double ComputeAverageUniqueness(
      arma::uword      start,
      arma::uword      end,
      const arma::vec& invConcurrency)
  {
    return arma::accu(invConcurrency.rows(start, end)) / (end - start + 1);
  }

  /**
   * Compute probabilities of the next draw for each observation.
   *
   * @param[in] phi List of previously drawn observations.
   * @param[in] phiSize Number of elements in use in @p phi. To
   *                    avoid reallocations, @p phi has been
   *                    overallocated already.
   * @param[in] concurrency Concurrency of the samples selected
   *                        by @p phi so far.
   * @param[in] inConcurrency Inverse concurrency of the samples.
   * @param[in] intervals See SequentialBootstrap().
   * @param[out] avg The probabilities for each observation to be
   *                 drawn in the next iteration.
   */
  static void ComputeNextDrawProbabilities(
      arma::uvec&       phi,
      arma::uword       phiSize,
      arma::vec&        concurrency,
      arma::vec&        invConcurrency,
      const IndMatType& intervals,
      arma::vec&        avg)
  {
    // phi may have less rows than intervals has columns as
    // phi is grown in multiple steps.
    assert(phi.n_rows <= intervals.n_cols && phiSize < phi.n_rows);
    assert(avg.n_rows == intervals.n_cols);

    for (arma::uword i(0); i < avg.size(); ++i)
    {
      // Temporarily assume another concurrency for the current i.
      concurrency.subvec(intervals(0, i), intervals(1, i)) += 1.0;
      invConcurrency.subvec(intervals(0, i), intervals(1, i)) =
          1.0 / concurrency.subvec(intervals(0, i), intervals(1, i));

      phi[phiSize] = i;
      avg[i] = ComputeAverageUniqueness(
          intervals(0, phi[phiSize]),
          intervals(1, phi[phiSize]),
          invConcurrency);

      concurrency.subvec(intervals(0, i), intervals(1, i)) -= 1.0;
      invConcurrency.subvec(intervals(0, i), intervals(1, i)) =
          1.0 / concurrency.subvec(intervals(0, i), intervals(1, i));
    }

    avg /= arma::sum(avg);
  }

  /**
   * Compute the samples of the next draw.
   *
   * @param[in] colCount Number of data points in the dataset and
   *                     the number of samples to draw respectively.
   *
   * @return A list of indices referring to the observations that
   *         should be sampled.
   */
  arma::uvec ComputeSamples(arma::uword colCount) const
  {
    // Randomly sample columns from the intervals matrix.
    DiscreteDistribution d(intervals.n_cols);
    arma::uvec           phi(colCount);
    arma::vec            concurrency(intervals.max() + 1, arma::fill::zeros);
    arma::vec            invConcurrency(intervals.max() + 1, arma::fill::ones);

    for (arma::uword i(0); i < colCount; ++i)
    {
      ComputeNextDrawProbabilities(
          phi, i, concurrency, invConcurrency, intervals, d.Probabilities());
      assert(d.Probabilities().size() == intervals.n_cols);

      phi[i] = d.Random()[0];
      concurrency.subvec(intervals(0, phi[i]), intervals(1, phi[i])) += 1.0;
      invConcurrency.subvec(intervals(0, phi[i]), intervals(1, phi[i])) =
          1.0 / concurrency.subvec(intervals(0, phi[i]), intervals(1, phi[i]));
    }

    return phi;
  }

  template<
      bool UseWeights,
      typename MatType,
      typename LabelsType,
      typename WeightsType>
  void Bootstrap(
      const MatType&     dataset,
      const LabelsType&  labels,
      const WeightsType& weights,
      MatType&           bootstrapDataset,
      LabelsType&        bootstrapLabels,
      WeightsType&       bootstrapWeights)
  {
    if (labels.n_cols != intervals.n_cols)
      throw std::invalid_argument("SequentialBootstrap::Bootstrap(): "
          "labels n_cols and intervals n_cols differ!");

    // observations are stored as columns and dimensions
    // (number of features) as rows.
    const arma::uvec phi = ComputeSamples(dataset.n_cols);

    bootstrapDataset = dataset.cols(phi);
    bootstrapLabels = labels.cols(phi);
    if (UseWeights)
      bootstrapWeights = weights.cols(phi);
  }

 private:
  const IndMatType intervals;
};

} // namespace mlpack

#endif
