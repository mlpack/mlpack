/**
 * @file similarity_interpolation.hpp
 * @author Wenhao Huang
 *
 * Interpolation weights are based on similarities and weights sum up to one.
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
  // Em[ty onstructor.
  SimilarityInterpolation() { }

 /**
  * Interpolation weights are computed as normalized similarities.
  *
  * @param weights resulting interpolation weights.
  * @param similarities similarites between query user and neighbors.
  */
  void GetWeights(arma::vec& weights, const arma::vec& similarities) const
  {
    if (similarities.n_elem == 0)
    {
      Log::Fatal << "Require: similarities.n_elem > 0. There should be at least "
          << "one neighbor!" << std::endl;
    }

    double similaritiesSum = arma::sum(similarites);
    if (std::fabs(similaritiesSum) < 1e-14)
      weights = arma::vec(similarities.n_elem, arma::fill::zeros);
    else
      weights = similarities / similaritiesSum;
  }
};

} // namespace cf
} // namespace mlpack

#endif
