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
    if (intervals.n_cols != 2)
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
    if (labels.n_cols != intervals.n_rows)
      throw std::invalid_argument("SequentialBootstrap::Bootstrap(): "
        "labels n_cols and intervals n_rows differ!");

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
  std::vector<arma::uword> ComputeSamples(arma::uword colCount) const
  {
    DiscreteDistribution     d;
    std::vector<arma::uword> phi;

    phi.reserve(colCount);
    for (arma::uword i(0); i < colCount; ++i) {
      d.Probabilities() = ComputeNextDrawProbabilities(phi, intervals);
      phi.push_back(d.Random()[0]);
    }

    return phi;
  }

  /**
   * Compute the average uniqueness of each event at any sampling time point.
   * The average uniqueness is a measure for how isolated an event is during
   * its lifetime from other events.
   *
   * @param[in] intervals Is a `2 x m` matrix, where each column has the start
   *                      sample and the end sample of an interval.
   * @param[in] indices Indices of the average uniqueness that is returned.
   * @return The average uniqueness of the events in @p intervals.
   */
  static arma::vec ComputeAverageUniqueness(
    const IndMatType& intervals,
    arma::uvec        indices)
  {
    arma::vec avg(intervals.n_rows, arma::fill::zeros);
    arma::uvec concurrency;

    for (arma::uword i(0); i < intervals.n_rows; ++i) {
      if (i + 1 < intervals.n_rows && intervals(i, 0) > intervals(i + 1, 0)) {
        throw std::invalid_argument(
          "intervals must be sorted by starting sample");
      }

      if (intervals(i, 1) >= concurrency.size()) {
        concurrency.insert_rows(concurrency.n_rows,
          1 + intervals(i, 1) - concurrency.n_rows);
      }

      concurrency.subvec(intervals(i, 0), intervals(i, 1)) += 1;
    }

    for (arma::uword i(0); i < intervals.n_rows; ++i) {
      for (arma::uword s(intervals(i, 0)); s <= intervals(i, 1); ++s) {
        // online averaging algorithm
        const arma::uword n(s - intervals(i, 0) + 1);

        avg(i) = ((n - 1) * avg(i) + (1.0 / concurrency(s))) / n;
      }
    }

    return avg.rows(indices);
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
    const std::vector<arma::uword>& phi,
    const IndMatType& intervals)
  {
    auto sorter = [&intervals](arma::uword lhs, arma::uword rhs)
      {
        return intervals(lhs, 0) < intervals(rhs, 0);
      };
    arma::vec avg(intervals.n_rows);

    std::vector<arma::uword> rows(phi);

    std::sort(rows.begin(), rows.end(), sorter);

    rows.reserve(rows.size() + 1);
    for (arma::uword i(0); i < avg.size(); ++i) {
      const std::vector<arma::uword>::const_iterator iter(
        rows.insert(std::upper_bound(rows.begin(), rows.end(), i, sorter),
          i));

      avg[i] = ComputeAverageUniqueness(
        intervals.rows(arma::conv_to<arma::uvec>::from(rows)),
        arma::uvec(1u,
          arma::fill::value(std::distance(rows.cbegin(), iter)))).back();

      rows.erase(iter);
    }

    return avg / arma::sum(avg);
  }

 private:
  const IndMatType intervals;
};

} // namespace mlpack

#endif
