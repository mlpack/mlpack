/**
 * @file similarity_interpolation.hpp
 * @author Wenhao Huang
 *
 * Definition of SimilarityInterpolation class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_SIMILARITY_INTERPOLATION_HPP
#define MLPACK_METHODS_CF_SIMILARITY_INTERPOLATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace cf {

/**
 * With SimilarityInterpolation, interpolation weights are based on
 * similarities between query user and its neighbors. All interpolation
 * weights sum up to one.
 */
class SimilarityInterpolation
{
 public:
  // Empty onstructor.
  SimilarityInterpolation() { }

  /**
   * This constructor is needed for interface consistency.
   */
  SimilarityInterpolation(const arma::sp_mat& /* cleanedData */) { }

  /**
   * Interpolation weights are computed as normalized similarities.
   *
   * @param weights Resulting interpolation weights.
   * @param w Matrix W from decomposition.
   * @param h Matrix H from decomposition.
   * @param queryUser Queried user.
   * @param neighbors Neighbors of queried user.
   * @param similarities Similarites between query user and neighbors.
   * @param cleanedData Sparse rating matrix.
   */
  void GetWeights(arma::vec& weights,
                  const arma::mat& /* w */,
                  const arma::mat& /* h */,
                  const size_t /* queryUser */,
                  const arma::Col<size_t>& /* neighbors */,
                  const arma::vec& similarities,
                  const arma::sp_mat& /* cleanedData */)
  {
    if (similarities.n_elem == 0)
    {
      Log::Fatal << "Require: similarities.n_elem > 0. There should be at "
          << "least one neighbor!" << std::endl;
    }

    double similaritiesSum = arma::sum(similarities);
    if (std::fabs(similaritiesSum) < 1e-14)
    {
      weights.set_size(similarities.n_elem);
      weights.fill(1.0 / similarities.n_elem);
    }
    else
    {
      weights = similarities / similaritiesSum;
    }
  }
};

} // namespace cf
} // namespace mlpack

#endif
