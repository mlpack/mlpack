/**
 * @file methods/cf/decomposition_policies/svdplusplus_method.hpp
 * @author Wenhao Huang
 *
 * Implementation of the svdplusplus method for use in the Collaborative
 * Filtering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_SVDPLUSPLUS_METHOD_HPP
#define MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_SVDPLUSPLUS_METHOD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/svdplusplus/svdplusplus.hpp>

namespace mlpack {

/**
 * Implementation of the SVDPlusPlus policy to act as a wrapper when
 * accessing SVDPlusPlus from within CFType.
 *
 * An example of how to use SVDPlusPlusPolicy in CF is shown below:
 *
 * @code
 * extern arma::mat data; // data is a (user, item, rating) table.
 * // Users for whom recommendations are generated.
 * extern arma::Col<size_t> users;
 * arma::Mat<size_t> recommendations; // Resulting recommendations.
 *
 * CFType<SVDPlusPlusPolicy> cf(data);
 *
 * // Generate 10 recommendations for all users.
 * cf.GetRecommendations(10, recommendations);
 * @endcode
 */
class SVDPlusPlusPolicy
{
 public:
  /**
   * Use SVDPlusPlus method to perform collaborative filtering.
   *
   * @param maxIterations Number of iterations.
   * @param alpha Learning rate for optimization.
   * @param lambda Regularization parameter for optimization.
   */
  SVDPlusPlusPolicy(const size_t maxIterations = 10,
                    const double alpha = 0.001,
                    const double lambda = 0.1) :
      maxIterations(maxIterations),
      alpha(alpha),
      lambda(lambda)
  {
    /* Nothing to do here */
  }

  /**
   * Apply Collaborative Filtering to the provided data set using the
   * svdplusplus.
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
    SVDPlusPlus<> svdpp(maxIterations, alpha, lambda);

    // Save implicit data in the form of sparse matrix.
    arma::mat implicitDenseData = data.submat(0, 0, 1, data.n_cols - 1);
    svdpp.CleanData(implicitDenseData, implicitData, data);

    // Perform decomposition using the svdplusplus algorithm.
    svdpp.Apply(data, implicitDenseData, rank, w, h, p, q, y);
  }

  /**
   * Return predicted rating given user ID and item ID.
   *
   * @param user User ID.
   * @param item Item ID.
   */
  double GetRating(const size_t user, const size_t item) const
  {
    // Iterate through each item which the user interacted with to calculate
    // user vector.
    arma::vec userVec(h.n_rows);
    arma::sp_mat::const_iterator it = implicitData.begin_col(user);
    arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
    size_t implicitCount = 0;
    for (; it != it_end; ++it)
    {
      userVec += y.col(it.row());
      implicitCount += 1;
    }
    if (implicitCount != 0)
      userVec /= std::sqrt(implicitCount);
    userVec += h.col(user);

    double rating =
        arma::as_scalar(w.row(item) * userVec) + p(item) + q(user);
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
    // Iterate through each item which the user interacted with to calculate
    // user vector.
    arma::vec userVec(h.n_rows);
    arma::sp_mat::const_iterator it = implicitData.begin_col(user);
    arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
    size_t implicitCount = 0;
    for (; it != it_end; ++it)
    {
      userVec += y.col(it.row());
      implicitCount += 1;
    }
    if (implicitCount != 0)
      userVec /= std::sqrt(implicitCount);
    userVec += h.col(user);

    rating = w * userVec + p + q(user);
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
  //! Get the Item Implicit Matrix.
  const arma::mat& Y() const { return y; }
  //! Get Implicit Feedback Data.
  const arma::sp_mat& ImplicitData() const { return implicitData; }

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
    ar(CEREAL_NVP(y));
    ar(CEREAL_NVP(implicitData));
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
  //! Item implicit matrix.
  arma::mat y;
  //! Implicit Data.
  arma::sp_mat implicitData;
};

} // namespace mlpack

#endif
