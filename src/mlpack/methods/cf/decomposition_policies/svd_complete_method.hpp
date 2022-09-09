/**
 * @file methods/cf/decomposition_policies/svd_complete_method.hpp
 * @author Haritha Nair
 *
 * Implementation of the SVD complete incremental method for use in the
 * Collaborative Filtering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_SVD_COMPLETE_METHOD_HPP
#define MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_SVD_COMPLETE_METHOD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/amf/update_rules/nmf_als.hpp>
#include <mlpack/methods/amf/termination_policies/max_iteration_termination.hpp>
#include <mlpack/methods/amf/termination_policies/simple_residue_termination.hpp>

namespace mlpack {

/**
 * Implementation of the SVD complete incremental policy to act as a wrapper
 * when accessing SVD complete decomposition from within CFType.
 *
 * An example of how to use SVDCompletePolicy in CF is shown below:
 *
 * @code
 * extern arma::mat data; // data is a (user, item, rating) table.
 * // Users for whom recommendations are generated.
 * extern arma::Col<size_t> users;
 * arma::Mat<size_t> recommendations; // Resulting recommendations.
 *
 * CFType<SVDCompletePolicy> cf(data);
 *
 * // Generate 10 recommendations for all users.
 * cf.GetRecommendations(10, recommendations);
 * @endcode
 */
class SVDCompletePolicy
{
 public:
  /**
   * Apply Collaborative Filtering to the provided data set using the
   * SVD complete incremental policy.
   *
   * @param * (data) Data matrix: dense matrix (coordinate lists)
   *    or sparse matrix(cleaned).
   * @param cleanedData item user table in form of sparse matrix.
   * @param rank Rank parameter for matrix factorization.
   * @param maxIterations Maximum number of iterations.
   * @param minResidue Residue required to terminate.
   * @param mit Whether to terminate only when maxIterations is reached.
   */
  template<typename MatType>
  void Apply(const MatType& /* data */,
             const arma::sp_mat& cleanedData,
             const size_t rank,
             const size_t maxIterations,
             const double minResidue,
             const bool mit)
  {
    if (mit)
    {
      MaxIterationTermination iter(maxIterations);

      // Do singular value decomposition using complete incremental method
      // using cleaned data in form of sparse matrix.
      AMF<MaxIterationTermination, RandomAMFInitialization,
          SVDCompleteIncrementalLearning<arma::sp_mat>> svdci(iter);

      svdci.Apply(cleanedData, rank, w, h);
    }
    else
    {
      SimpleResidueTermination srt(minResidue, maxIterations);

      // Do singular value decomposition using complete incremental method
      // using cleaned data in form of sparse matrix.
      SVDCompleteIncrementalFactorizer<arma::sp_mat> svdci(srt);

      svdci.Apply(cleanedData, rank, w, h);
    }
  }

  /**
   * Return predicted rating given user ID and item ID.
   *
   * @param user User ID.
   * @param item Item ID.
   */
  double GetRating(const size_t user, const size_t item) const
  {
    double rating = arma::as_scalar(w.row(item) * h.col(user));
    return rating;
  }

  /**
   * Get predicted ratings for a user.
   *
   * @param user User ID.
   * @param rating Resulting rating vector.
   */
  void GetRatingOfUser(const size_t user, arma::vec& rating) const
  {
    rating = w * h.col(user);
  }

  /**
   * Get the neighborhood and corresponding similarities for a set of users.
   *
   * @tparam NeighborSearchPolicy The policy to perform neighbor search.
   *
   * @param users Users whose neighborhood is to be computed.
   * @param numUsersForSimilarity The number of neighbors returned for
   *     each user.
   * @param neighborhood Neighbors represented by user IDs.
   * @param similarities Similarity between each user and each of its
   *     neighbors.
   */
  template<typename NeighborSearchPolicy>
  void GetNeighborhood(const arma::Col<size_t>& users,
                       const size_t numUsersForSimilarity,
                       arma::Mat<size_t>& neighborhood,
                       arma::mat& similarities) const
  {
    // We want to avoid calculating the full rating matrix, so we will do
    // nearest neighbor search only on the H matrix, using the observation that
    // if the rating matrix X = W*H, then d(X.col(i), X.col(j)) = d(W H.col(i),
    // W H.col(j)).  This can be seen as nearest neighbor search on the H
    // matrix with the Mahalanobis distance where M^{-1} = W^T W.  So, we'll
    // decompose M^{-1} = L L^T (the Cholesky decomposition), and then multiply
    // H by L^T. Then we can perform nearest neighbor search.
    arma::mat l = arma::chol(w.t() * w);
    arma::mat stretchedH = l * h; // Due to the Armadillo API, l is L^T.

    // Temporarily store feature vector of queried users.
    arma::mat query(stretchedH.n_rows, users.n_elem);
    // Select feature vectors of queried users.
    for (size_t i = 0; i < users.n_elem; ++i)
      query.col(i) = stretchedH.col(users(i));

    NeighborSearchPolicy neighborSearch(stretchedH);
    neighborSearch.Search(
        query, numUsersForSimilarity, neighborhood, similarities);
  }

  //! Get the Item Matrix.
  const arma::mat& W() const { return w; }
  //! Get the User Matrix.
  const arma::mat& H() const { return h; }

  /**
   * Serialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(w));
    ar(CEREAL_NVP(h));
  }

 private:
  //! Item matrix.
  arma::mat w;
  //! User matrix.
  arma::mat h;
};

} // namespace mlpack

#endif
