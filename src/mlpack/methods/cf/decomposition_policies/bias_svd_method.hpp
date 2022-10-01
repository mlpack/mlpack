/**
 * @file methods/cf/decomposition_policies/bias_svd_method.hpp
 * @author Wenhao Huang
 *
 * Implementation of the bias svd method for use in the Collaborative
 * Filtering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_BIAS_SVD_METHOD_HPP
#define MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_BIAS_SVD_METHOD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/bias_svd/bias_svd.hpp>

namespace mlpack {

/**
 * Implementation of the Bias SVD policy to act as a wrapper when
 * accessing Bias SVD from within CFType.
 *
 * An example of how to use BiasSVDPolicy in CF is shown below:
 *
 * @code
 * extern arma::mat data; // data is a (user, item, rating) table.
 * // Users for whom recommendations are generated.
 * extern arma::Col<size_t> users;
 * arma::Mat<size_t> recommendations; // Resulting recommendations.
 *
 * CFType<BiasSVDPolicy> cf(data);
 *
 * // Generate 10 recommendations for all users.
 * cf.GetRecommendations(10, recommendations);
 * @endcode
 */
class BiasSVDPolicy
{
 public:
  /**
   * Use Bias SVD method to perform collaborative filtering.
   *
   * @param maxIterations Number of iterations.
   * @param alpha Learning rate for optimization.
   * @param lambda Regularization parameter for optimization.
   */
  BiasSVDPolicy(const size_t maxIterations = 10,
                const double alpha = 0.02,
                const double lambda = 0.05) :
      maxIterations(maxIterations),
      alpha(alpha),
      lambda(lambda)
  {
    /* Nothing to do here */
  }

  /**
   * Apply Collaborative Filtering to the provided data set using the
   * bias SVD.
   *
   * @param data Data matrix: dense matrix (coordinate lists)
   *    or sparse matrix(cleaned).
   * @param * (cleanedData) item user table in form of sparse matrix.
   * @param rank Rank parameter for matrix factorization.
   * @param maxIterations Maximum number of iterations.
   * @param * (minResidue) Residue required to terminate.
   * @param * (mit) Whether to terminate only when maxIterations is reached.
   */
  void Apply(const arma::mat& data,
             const arma::sp_mat& /* cleanedData */,
             const size_t rank,
             const size_t maxIterations,
             const double /* minResidue */,
             const bool /* mit */)
  {
    // Perform decomposition using the bias SVD algorithm.
    BiasSVD<> biassvd(maxIterations, alpha, lambda);
    biassvd.Apply(data, rank, w, h, p, q);
  }

  /**
   * Return predicted rating given user ID and item ID.
   *
   * @param user User ID.
   * @param item Item ID.
   */
  double GetRating(const size_t user, const size_t item) const
  {
    double rating =
        arma::as_scalar(w.row(item) * h.col(user)) + p(item) + q(user);
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
    rating = w * h.col(user) + p + q(user);
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
    // User latent vectors (matrix H) are used for neighbor search.
    // Temporarily store feature vector of queried users.
    arma::mat query(h.n_rows, users.n_elem);
    // Select feature vectors of queried users.
    for (size_t i = 0; i < users.n_elem; ++i)
      query.col(i) = h.col(users(i));

    NeighborSearchPolicy neighborSearch(h);
    neighborSearch.Search(
        query, numUsersForSimilarity, neighborhood, similarities);
  }

  //! Get the Item Matrix.
  const arma::mat& W() const { return w; }
  //! Get the User Matrix.
  const arma::mat& H() const { return h; }
  //! Get the User Bias Vector.
  const arma::vec& Q() const { return q; }
  //! Get the Item Bias Vector.
  const arma::vec& P() const { return p; }

  //! Get the number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the number of iterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Get learning rate.
  double Alpha() const { return alpha; }
  //! Modify learning rate.
  double& Alpha() { return alpha; }

  //! Get regularization parameter.
  double Lambda() const { return lambda; }
  //! Modify regularization parameter.
  double& Lambda() { return lambda; }

  /**
   * Serialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(maxIterations));
    ar(CEREAL_NVP(alpha));
    ar(CEREAL_NVP(lambda));
    ar(CEREAL_NVP(w));
    ar(CEREAL_NVP(h));
    ar(CEREAL_NVP(p));
    ar(CEREAL_NVP(q));
  }

 private:
  //! Locally stored number of iterations.
  size_t maxIterations;
  //! Learning rate for optimization.
  double alpha;
  //! Regularization parameter for optimization.
  double lambda;
  //! Item matrix.
  arma::mat w;
  //! User matrix.
  arma::mat h;
  //! Item bias.
  arma::vec p;
  //! User bias.
  arma::vec q;
};

} // namespace mlpack

#endif
