/**
 * @file cf.hpp
 * @author Mudit Raj Gupta
 * @author Sumedh Ghaisas
 *
 * Collaborative filtering.
 *
 * Defines the CFType class to perform collaborative filtering on the specified data
 * set using alternating least squares (ALS).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_CF_HPP
#define MLPACK_METHODS_CF_CF_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/amf/update_rules/nmf_als.hpp>
#include <mlpack/methods/amf/termination_policies/simple_residue_termination.hpp>
#include <mlpack/methods/cf/decomposition_policies/nmf_method.hpp>
#include <set>
#include <map>
#include <iostream>

namespace mlpack {
namespace cf /** Collaborative filtering. **/ {
/**
 * This class implements Collaborative Filtering (CF). This implementation
 * presently supports Alternating Least Squares (ALS) for collaborative
 * filtering.
 *
 * A simple example of how to run Collaborative Filtering is shown below.
 *
 * @code
 * extern arma::mat data; // (user, item, rating) table
 * extern arma::Col<size_t> users; // users seeking recommendations
 * arma::Mat<size_t> recommendations; // Recommendations
 *
 * CFType cf(data); // Default options.
 *
 * // Generate 10 recommendations for all users.
 * cf.GetRecommendations(10, recommendations);
 *
 * // Generate 10 recommendations for specified users.
 * cf.GetRecommendations(10, recommendations, users);
 *
 * @endcode
 *
 * The data matrix is a (user, item, rating) table.  Each column in the matrix
 * should have three rows.  The first represents the user; the second represents
 * the item; and the third represents the rating.  The user and item, while they
 * are in a matrix that holds doubles, should hold integer (or size_t) values.
 * The user and item indices are assumed to start at 0.
 *
 * @tparam DecompositionPolicy The algorithm to use to decompose
 *     the rating matrix (a W and H matrix).
 */
class CFType
{
 public:
  /**
   * Initialize the CFType object without performing any factorization.  Be sure to
   * call Train() before calling GetRecommendations() or any other functions!
   */
  CFType(const size_t numUsersForSimilarity = 5,
     const size_t rank = 0);

  /**
   * Initialize the CFType object using any decomposition method, immediately
   * factorizing the given data to create a model. There are parameters that can
   * be set; default values are provided for each of them. If the rank is left
   * unset (or is set to 0), a simple density-based heuristic will be used to
   * choose a rank.
   *
   * The provided dataset can be a coordinate list; that is, a 3-row matrix
   * where each column corresponds to a (user, item, rating) entry in the
   * matrix or a sparse matrix representing (user, item) table.
   *
   * @param data Data matrix: dense matrix (coordinate lists) 
   *    or sparse matrix(cleaned).
   * @param decomposition Instantiated DecompositionPolicy object.
   * @param numUsersForSimilarity Size of the neighborhood.
   * @param rank Rank parameter for matrix factorization.
   * @param maxIterations Maximum number of iterations.
   * @param minResidue Residue required to terminate.
   * @param mit Whether to terminate only when maxIterations is reached.
   */
  template<typename MatType, typename DecompositionPolicy = NMFPolicy>
  CFType(const MatType& data,
         DecompositionPolicy& decomposition = DecompositionPolicy(),
         const size_t numUsersForSimilarity = 5,
         const size_t rank = 0,
         const size_t maxIterations = 1000,
         const double minResidue = 1e-5,
         const bool mit = false);

  /**
   * Train the CFType model (i.e. factorize the input matrix) using the
   * parameters that have already been set for the model (specifically, the rank
   * parameter), and optionally, using the given DecompositionPolicy.
   *
   * @param data Input dataset; dense matrix (coordinate lists).
   * @param decomposition Instantiated DecompositionPolicy object.
   * @param maxIterations Maximum number of iterations.
   * @param minResidue Residue required to terminate.
   * @param mit Whether to terminate only when maxIterations is reached.
   */
  template<typename DecompositionPolicy>
  void Train(const arma::mat& data,
             DecompositionPolicy& decomposition,
             const size_t maxIterations = 1000,
             const double minResidue = 1e-5,
             const bool mit = false);

  /**
   * Train the CFType model (i.e. factorize the input matrix) using the
   * parameters that have already been set for the model (specifically, the
   * rank parameter), and optionally, using the given DecompositionPolicy.
   *
   * @param data Input dataset; sparse matrix (user item table).
   * @param decomposition Instantiated DecompositionPolicy object.
   * @param maxIterations Maximum number of iterations.
   * @param minResidue Residue required to terminate.
   * @param mit Whether to terminate only when maxIterations is reached.
   */
  template<typename DecompositionPolicy>
  void Train(const arma::sp_mat& data,
             DecompositionPolicy& decomposition,
             const size_t maxIterations = 1000,
             const double minResidue = 1e-5,
             const bool mit = false);

  //! Sets number of users for calculating similarity.
  void NumUsersForSimilarity(const size_t num)
  {
    if (num < 1)
    {
      Log::Warn << "CFType::NumUsersForSimilarity(): invalid value (< 1) "
          "ignored." << std::endl;
      return;
    }
    this->numUsersForSimilarity = num;
  }

  //! Gets number of users for calculating similarity.
  size_t NumUsersForSimilarity() const
  {
    return numUsersForSimilarity;
  }

  //! Sets rank parameter for matrix factorization.
  void Rank(const size_t rankValue)
  {
    this->rank = rankValue;
  }

  //! Gets rank parameter for matrix factorization.
  size_t Rank() const
  {
    return rank;
  }

  //! Get the User Matrix.
  const arma::mat& W() const { return w; }
  //! Get the Item Matrix.
  const arma::mat& H() const { return h; }
  //! Get the cleaned data matrix.
  const arma::sp_mat& CleanedData() const { return cleanedData; }

  /**
   * Generates the given number of recommendations for all users.
   *
   * @param numRecs Number of Recommendations.
   * @param recommendations Matrix to save recommendations into.
   */
  void GetRecommendations(const size_t numRecs,
                          arma::Mat<size_t>& recommendations);

  /**
   * Generates the given number of recommendations for the specified users.
   *
   * @param numRecs Number of Recommendations.
   * @param recommendations Matrix to save recommendations.
   * @param users Users for which recommendations are to be generated.
   */
  void GetRecommendations(const size_t numRecs,
                          arma::Mat<size_t>& recommendations,
                          const arma::Col<size_t>& users);

  //! Converts the User, Item, Value Matrix to User-Item Table.
  static void CleanData(const arma::mat& data, arma::sp_mat& cleanedData);

  /**
   * Predict the rating of an item by a particular user.
   *
   * @param user User to predict for.
   * @param item Item to predict for.
   */
  double Predict(const size_t user, const size_t item) const;

  /**
   * Predict ratings for each user-item combination in the given coordinate list
   * matrix.  The matrix 'combinations' should have two rows and number of
   * columns equal to the number of desired predictions.  The first element of
   * each column corresponds to the user index, and the second element of each
   * column corresponds to the item index.  The output vector 'predictions' will
   * have length equal to combinations.n_cols, and predictions[i] will be equal
   * to the prediction for the user/item combination in combinations.col(i).
   *
   * @param combinations User/item combinations to predict.
   * @param predictions Predicted ratings for each user/item combination.
   */
  void Predict(const arma::Mat<size_t>& combinations,
               arma::vec& predictions) const;

  /**
   * Serialize the CFType model to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Number of users for similarity.
  size_t numUsersForSimilarity;
  //! Rank used for matrix factorization.
  size_t rank;
  //! User matrix.
  arma::mat w;
  //! Item matrix.
  arma::mat h;
  //! Cleaned data matrix.
  arma::sp_mat cleanedData;

  //! Candidate represents a possible recommendation (value, item).
  typedef std::pair<double, size_t> Candidate;

  //! Compare two candidates based on the value.
  struct CandidateCmp {
    bool operator()(const Candidate& c1, const Candidate& c2)
    {
      return c1.first > c2.first;
    };
  };
}; // class CFType

} // namespace cf
} // namespace mlpack

// Include implementation of templated functions.
#include "cf_impl.hpp"

#endif
