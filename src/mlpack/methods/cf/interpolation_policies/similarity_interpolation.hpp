/**
 * @file methods/cf/interpolation_policies/similarity_interpolation.hpp
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

/**
 * With SimilarityInterpolation, interpolation weights are based on
 * similarities between query user and its neighbors. All interpolation
 * weights sum up to one.
 *
 * An example of how to use SimilarityInterpolation in CF is shown below:
 *
 * @code
 * extern arma::mat data; // data is a (user, item, rating) table.
 * // Users for whom recommendations are generated.
 * extern arma::Col<size_t> users;
 * arma::Mat<size_t> recommendations; // Resulting recommendations.
 *
 * CFType<> cf(data);
 *
 * // Generate 10 recommendations for all users.
 * cf.template GetRecommendations<
 *     EuclideanSearch,
 *     SimilarityInterpolation>(10, recommendations);
 * @endcode
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
   * After getting the weights, CF algorithm multiplies each neighbor's rating
   * by its corresponding weight and sums them to get predicted rating.
   *
   * @param weights Resulting interpolation weights. The size of weights should
   *     be set to the number of neighbors before calling GetWeights().
   * @param * (decomposition) Decomposition object.
   * @param * (queryUser) Queried user.
   * @param neighbors Neighbors of queried user.
   * @param similarities Similarities between query user and neighbors.
   * @param * (cleanedData) Sparse rating matrix.
   */
  template <typename VectorType,
            typename DecompositionPolicy>
  void GetWeights(VectorType&& weights,
                  const DecompositionPolicy& /* decomposition */,
                  const size_t /* queryUser */,
                  const arma::Col<size_t>& neighbors,
                  const arma::vec& similarities,
                  const arma::sp_mat& /* cleanedData */)
  {
    if (similarities.n_elem == 0)
    {
      Log::Fatal << "Require: similarities.n_elem > 0. There should be at "
          << "least one neighbor!" << std::endl;
    }

    if (weights.n_elem != neighbors.n_elem)
    {
      Log::Fatal << "The size of the first parameter (weights) should "
          << "be set to the number of neighbors before calling GetWeights()."
          << std::endl;
    }

    double similaritiesSum = sum(similarities);
    if (std::fabs(similaritiesSum) < 1e-14)
    {
      weights.fill(1.0 / similarities.n_elem);
    }
    else
    {
      weights = similarities / similaritiesSum;
    }
  }
};

} // namespace mlpack

#endif
