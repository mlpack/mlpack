/**
 * @file methods/kmeans/e_means_impl.hpp
 *
 * Implementation of e-means.  Algorithm 1, equations and constants refer to
 * "e-means: outperforming k-means (in one-tenth the epochs)", Bessa et al.,
 * arXiv:2611.09501, 2026.  The fitted batch and step laws are the
 * coefficients shipped with the reference implementation in Faiss.
 *
 * Parts of this file are ported from Faiss (faiss/contrib/e_means), which is
 * distributed under the following license:
 *
 *   MIT License
 *
 *   Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a
 *   copy of this software and associated documentation files (the "Software"),
 *   to deal in the Software without restriction, including without limitation
 *   the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *   and/or sell copies of the Software, and to permit persons to whom the
 *   Software is furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *   DEALINGS IN THE SOFTWARE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_KMEANS_E_MEANS_IMPL_HPP
#define MLPACK_METHODS_KMEANS_E_MEANS_IMPL_HPP

// In case it hasn't been included yet.
#include "e_means.hpp"

namespace mlpack {

namespace emeans_constants {

// Epsilon in Equation 5: the final step is this fraction of the initial step.
constexpr double alphaMinFactor = 0.1;
// Number of batch updates between revival checks (Algorithm 1, line 9).
constexpr size_t checkUnusedEvery = 5;
// Initial mass N_j of each cluster (Algorithm 1, line 1).
constexpr double initUsage = 1.0 / 65536.0;
// Relative perturbation of split centroids (Appendix J, Equation 8).
constexpr double splitEpsilon = 1e-4;
// Default revival threshold, and the dimensionality at or below which
// revival is disabled by default.
constexpr double unusedThreshold = 0.1;
constexpr size_t noRevivalMaxDim = 2;
// The AUTO batch law selects QUALITY at or below this dimensionality.
constexpr size_t autoQualityMaxDim = 16;
// Batch law B = c * k^b_k * n^b_n * T^b_T (Equation 7).
constexpr double throughputBatchCoef = 0.035134;
constexpr double qualityBatchCoef = 0.00439175;
constexpr double batchExpK = 0.134749;
constexpr double batchExpN = 0.876473;
constexpr double batchExpT = 0.813344;
// Step law alpha0 = c * k^a_k * B^a_B * n^a_n * T^a_T (Equation 6).
constexpr double stepCoef = 0.8244;
constexpr double stepExpK = -0.4617;
constexpr double stepExpB = 0.9003;
constexpr double stepExpN = -0.4498;
constexpr double stepExpT = -0.1057;

// Round a positive value to the nearest power of two.
inline size_t RoundPowerOfTwo(const double value)
{
  if (value <= 1.0)
    return 1;
  const size_t low = size_t(1) << size_t(std::floor(std::log2(value)));
  const size_t high = 2 * low;
  return (value / low < high / value) ? low : high;
}

} // namespace emeans_constants

template<typename InitialPartitionPolicy>
EMeans<InitialPartitionPolicy>::EMeans(
    const double epochs,
    const size_t batchSize,
    const double alpha0,
    const BatchLaw batchLaw,
    const double unusedThreshold,
    const InitialPartitionPolicy& partitioner) :
    epochs(epochs),
    batchSize(batchSize),
    alpha0(alpha0),
    batchLaw(batchLaw),
    unusedThreshold(unusedThreshold),
    partitioner(partitioner),
    usedBatchSize(0),
    usedAlpha0(0.0),
    steps(0),
    splits(0)
{
  // Nothing to do.
}

template<typename InitialPartitionPolicy>
size_t EMeans<InitialPartitionPolicy>::BatchSizeLaw(const size_t n,
                                                    const size_t k,
                                                    const double epochs,
                                                    const size_t d,
                                                    const BatchLaw law)
{
  using namespace emeans_constants;
  const bool quality = (law == QUALITY) ||
      (law == AUTO && d <= autoQualityMaxDim);
  const double coef = quality ? qualityBatchCoef : throughputBatchCoef;
  size_t batch = RoundPowerOfTwo(coef * std::pow((double) k, batchExpK) *
      std::pow((double) n, batchExpN) * std::pow(epochs, batchExpT));
  if (quality)
    batch = std::min(batch, n / 2);
  else
    batch = std::min(batch, RoundPowerOfTwo(n / 2.0));
  return std::max<size_t>(1, std::min(batch, n));
}

template<typename InitialPartitionPolicy>
double EMeans<InitialPartitionPolicy>::StepLaw(const size_t n,
                                               const size_t k,
                                               const double epochs,
                                               const size_t batchSize)
{
  using namespace emeans_constants;
  const double a = stepCoef * std::pow((double) k, stepExpK) *
      std::pow((double) batchSize, stepExpB) *
      std::pow((double) n, stepExpN) * std::pow(epochs, stepExpT);
  return std::min(std::max(a, 1e-4), 1.0);
}

template<typename InitialPartitionPolicy>
double EMeans<InitialPartitionPolicy>::ScheduledAlpha(const double alpha0,
                                                      const size_t step,
                                                      const size_t totalSteps)
{
  using namespace emeans_constants;
  const double fraction = double(std::min(step, totalSteps)) /
      double(std::max<size_t>(1, totalSteps));
  const double cosine = 0.5 * (1.0 + std::cos(M_PI * fraction));
  return alpha0 * (alphaMinFactor + (1.0 - alphaMinFactor) * cosine);
}

template<typename InitialPartitionPolicy>
std::vector<size_t> EMeans<InitialPartitionPolicy>::EpochRows(
    const size_t n, const double epochs)
{
  // Whole passes, then a fractional pass, then a full last pass whose
  // assignments give the final centroids.  Ties round to even.
  const size_t whole = (size_t) std::floor(epochs);
  const size_t remainder = (size_t) std::nearbyint((epochs - whole) * n);
  std::vector<size_t> rows;
  if (whole == 0)
  {
    rows.push_back(remainder);
    return rows;
  }
  for (size_t i = 0; i + 1 < whole; ++i)
    rows.push_back(n);
  if (remainder > 0)
    rows.push_back(remainder);
  rows.push_back(n);
  return rows;
}

template<typename InitialPartitionPolicy>
template<typename MatType>
void EMeans<InitialPartitionPolicy>::Assign(
    const MatType& data,
    const arma::Mat<typename MatType::elem_type>& centroids,
    arma::Row<size_t>& assignments,
    arma::Col<typename MatType::elem_type>* distances)
{
  using eT = typename MatType::elem_type;
  const size_t n = data.n_cols;
  const size_t k = centroids.n_cols;
  if (k == 0)
  {
    throw std::invalid_argument("EMeans::Assign(): there must be at least "
        "one centroid!");
  }
  util::CheckSameDimensionality(data, centroids, "EMeans::Assign()");
  assignments.set_size(n);
  if (distances)
    distances->set_size(n);

  const arma::Row<eT> cNorms = arma::sum(arma::square(centroids), 0);
  // Keep each k x chunk block of distances near 32 MiB.
  const size_t chunk = std::max<size_t>(1, std::min<size_t>(65536,
      (size_t(32) << 20) / (k * sizeof(eT))));
  arma::Mat<eT> partial;
  for (size_t begin = 0; begin < n; begin += chunk)
  {
    const size_t end = std::min(begin + chunk, n);
    // ||x||^2 is the same for every centroid, so the closest centroid
    // minimizes ||c||^2 - 2 c^T x.
    partial = centroids.t() * data.cols(begin, end - 1);
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < end - begin; ++j)
    {
      size_t best = 0;
      eT bestValue = eT(-2) * partial(0, j) + cNorms[0];
      for (size_t c = 1; c < k; ++c)
      {
        const eT value = eT(-2) * partial(c, j) + cNorms[c];
        if (value < bestValue)
        {
          bestValue = value;
          best = c;
        }
      }
      assignments[begin + j] = best;
      if (distances)
      {
        const eT dist = bestValue +
            arma::dot(data.col(begin + j), data.col(begin + j));
        (*distances)[begin + j] = std::max(dist, eT(0));
      }
    }
  }
}

template<typename InitialPartitionPolicy>
template<typename MatType, typename OrderFunction, typename UniformFunction>
void EMeans<InitialPartitionPolicy>::Train(
    const MatType& data,
    arma::Mat<typename MatType::elem_type>& centroids,
    const size_t batchSize,
    const double alpha0,
    const double epochs,
    const double unusedThreshold,
    OrderFunction&& epochOrder,
    UniformFunction&& uniform,
    size_t& steps,
    size_t& splits)
{
  using namespace emeans_constants;
  using eT = typename MatType::elem_type;

  const size_t d = data.n_rows;
  const size_t k = centroids.n_cols;
  if (batchSize == 0)
  {
    throw std::invalid_argument("EMeans::Train(): the batch size must be "
        "positive!");
  }
  if (!(epochs > 0.0) || !std::isfinite(epochs))
  {
    throw std::invalid_argument("EMeans::Train(): the number of epochs must "
        "be positive and finite!");
  }
  const std::vector<size_t> epochRows = EpochRows(data.n_cols, epochs);
  size_t totalSteps = 0;
  for (const size_t rows : epochRows)
    totalSteps += (rows + batchSize - 1) / batchSize;

  // The EMA mass N_j of each cluster; the centroid c_j stores S_j / N_j.
  arma::Col<eT> usage(k);
  usage.fill(eT(initUsage));
  arma::Mat<eT> batchSums(d, k);
  arma::Col<eT> batchCounts(k);
  // Counts and sums of the current epoch; the sums are kept for the last
  // epoch only (Algorithm 1, line 12).
  arma::vec passCounts(k);
  arma::Mat<eT> passSums;
  arma::Mat<eT> batch;
  arma::Row<size_t> labels;

  // Direction of the split perturbation: +1, -1, +1, ... per dimension.
  arma::Col<eT> offset(d);
  for (size_t i = 0; i < d; ++i)
    offset[i] = eT(splitEpsilon) * ((i % 2 == 0) ? eT(1) : eT(-1));

  steps = 0;
  splits = 0;
  for (size_t epoch = 0; epoch < epochRows.size(); ++epoch)
  {
    const bool lastEpoch = (epoch + 1 == epochRows.size());
    const arma::uvec order = epochOrder(epoch);
    const size_t rows = epochRows[epoch];
    passCounts.zeros();
    if (lastEpoch)
      passSums.zeros(d, k);

    for (size_t begin = 0; begin < rows; begin += batchSize)
    {
      const size_t end = std::min(begin + batchSize, rows);
      // Lines 6-7: assign the batch against one fixed set of centroids.
      batch = data.cols(order.subvec(begin, end - 1));
      Assign(batch, centroids, labels);
      batchSums.zeros();
      batchCounts.zeros();
      for (size_t j = 0; j < batch.n_cols; ++j)
      {
        batchSums.col(labels[j]) += batch.col(j);
        batchCounts[labels[j]] += eT(1);
      }

      // Line 8: exponential moving averages of the counts and sums,
      // applied as an update of the mass and position of each cluster.
      const double alpha = ScheduledAlpha(alpha0, steps, totalSteps);
      const eT a = eT(alpha);
      const eT decay = eT(1.0 - alpha);
      passCounts += arma::conv_to<arma::vec>::from(batchCounts);
      if (lastEpoch)
        passSums += batchSums;
      if (alpha != 1.0)
        usage *= decay;
      for (size_t c = 0; c < k; ++c)
      {
        if (batchCounts[c] == eT(0))
          continue;
        const eT retained = (alpha == 1.0) ? eT(0) : usage[c];
        const eT mass = retained + a * batchCounts[c];
        const eT weight = a * batchCounts[c] / mass;
        if (weight == eT(1))
        {
          centroids.col(c) = batchSums.col(c) / batchCounts[c];
        }
        else
        {
          centroids.col(c) += weight *
              (batchSums.col(c) / batchCounts[c] - centroids.col(c));
        }
        usage[c] = mass;
      }

      // Line 9: revive clusters with little mass by splitting clusters
      // sampled in proportion to their mass, without replacement.
      if (unusedThreshold > 0.0 && (steps + 1) % checkUnusedEvery == 0)
      {
        const eT threshold = (arma::accu(usage) * eT(unusedThreshold)) /
            eT(k);
        std::vector<size_t> unused, donors;
        for (size_t c = 0; c < k; ++c)
        {
          if (usage[c] < threshold)
            unused.push_back(c);
          else if (usage[c] > eT(0))
            donors.push_back(c);
        }
        const size_t nsplit = std::min(unused.size(), donors.size());
        if (nsplit > 0)
        {
          // Gumbel-top-k sampling of the donors.
          const arma::vec u = uniform(k);
          std::vector<std::pair<double, size_t>> keys;
          keys.reserve(donors.size());
          for (const size_t c : donors)
          {
            const double uc = std::max(u[c],
                std::numeric_limits<double>::min());
            keys.emplace_back(std::log((double) usage[c]) -
                std::log(-std::log(uc)), c);
          }
          std::sort(keys.begin(), keys.end(),
              [](const std::pair<double, size_t>& x,
                 const std::pair<double, size_t>& y)
              { return x.first > y.first; });

          for (size_t i = 0; i < nsplit; ++i)
          {
            const size_t source = keys[i].second;
            const size_t target = unused[i];
            const arma::Col<eT> c = centroids.col(source);
            arma::Col<eT> donor = c % (eT(1) - offset);
            arma::Col<eT> child = c % (eT(1) + offset);
            // A zero centroid gives identical halves: separate them.
            if (arma::all(donor == child))
            {
              eT next = std::nextafter(child[0],
                  std::numeric_limits<eT>::infinity());
              if (!std::isfinite(next))
              {
                next = std::nextafter(child[0],
                    -std::numeric_limits<eT>::infinity());
              }
              child[0] = next;
            }
            const eT half = usage[source] / eT(2);
            usage[source] = half;
            centroids.col(source) = donor;
            usage[target] = half;
            centroids.col(target) = child;
          }
          splits += nsplit;
        }
      }
      ++steps;
    }
  }

  // Line 12: final centroids are the means of the last epoch's assignments.
  for (size_t c = 0; c < k; ++c)
  {
    if (passCounts[c] > 0.0)
      centroids.col(c) = passSums.col(c) / eT(passCounts[c]);
  }
}

template<typename InitialPartitionPolicy>
template<typename MatType>
void EMeans<InitialPartitionPolicy>::Cluster(
    const MatType& data,
    const size_t clusters,
    arma::Mat<typename MatType::elem_type>& centroids,
    const bool initialGuess)
{
  const size_t n = data.n_cols;
  const size_t d = data.n_rows;
  if (clusters == 0 || clusters > n)
  {
    std::ostringstream oss;
    oss << "EMeans::Cluster(): the number of clusters (" << clusters
        << ") must be positive and at most the number of points (" << n
        << ")!";
    throw std::invalid_argument(oss.str());
  }
  if (!(epochs > 0.0) || !std::isfinite(epochs))
  {
    throw std::invalid_argument("EMeans::Cluster(): the number of epochs "
        "must be positive and finite!");
  }
  if (alpha0 < 0.0 || alpha0 > 1.0)
  {
    throw std::invalid_argument("EMeans::Cluster(): alpha0 must be in [0, 1] "
        "(0 selects it with the step law)!");
  }

  if (initialGuess)
  {
    util::CheckSameSizes(centroids, clusters, "EMeans::Cluster()",
        "clusters");
    util::CheckSameDimensionality(data, centroids, "EMeans::Cluster()");
  }
  else
  {
    partitioner.Cluster(data, clusters, centroids);
  }

  usedBatchSize = (batchSize == 0) ?
      BatchSizeLaw(n, clusters, epochs, d, batchLaw) :
      std::min(batchSize, n);
  usedAlpha0 = (alpha0 == 0.0) ?
      StepLaw(n, clusters, epochs, usedBatchSize) : alpha0;
  const double threshold = (unusedThreshold < 0.0) ?
      ((d <= emeans_constants::noRevivalMaxDim) ? 0.0 :
          emeans_constants::unusedThreshold) :
      unusedThreshold;

  Train(data, centroids, usedBatchSize, usedAlpha0, epochs, threshold,
      [n](const size_t /* epoch */)
      { return arma::randperm<arma::uvec>(n); },
      [](const size_t k) { return arma::randu<arma::vec>(k); },
      steps, splits);

  Log::Info << "EMeans::Cluster(): " << steps << " batch updates of "
      << usedBatchSize << " points, alpha0 " << usedAlpha0 << ", " << splits
      << " revived clusters." << std::endl;
}

template<typename InitialPartitionPolicy>
template<typename MatType>
void EMeans<InitialPartitionPolicy>::Cluster(
    const MatType& data,
    const size_t clusters,
    arma::Row<size_t>& assignments,
    arma::Mat<typename MatType::elem_type>& centroids,
    const bool initialGuess)
{
  Cluster(data, clusters, centroids, initialGuess);
  Assign(data, centroids, assignments);
}

template<typename InitialPartitionPolicy>
template<typename Archive>
void EMeans<InitialPartitionPolicy>::serialize(Archive& ar,
                                               const uint32_t /* version */)
{
  size_t law = (size_t) batchLaw;
  ar(CEREAL_NVP(epochs));
  ar(CEREAL_NVP(batchSize));
  ar(CEREAL_NVP(alpha0));
  ar(CEREAL_NVP(law));
  ar(CEREAL_NVP(unusedThreshold));
  batchLaw = (BatchLaw) law;
}

} // namespace mlpack

#endif
