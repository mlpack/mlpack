/**
 * @file cf.hpp
 * @author Mudit Raj Gupta
 * @author Sumedh Ghaisas
 *
 * Collaborative filtering.
 *
 * Defines the CF class to perform collaborative filtering on the specified data
 * set using alternating least squares (ALS).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_CF_HPP
#define MLPACK_METHODS_CF_CF_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/amf/update_rules/nmf_als.hpp>
#include <mlpack/methods/amf/termination_policies/simple_residue_termination.hpp>
#include <set>
#include <map>
#include <iostream>

namespace mlpack {
namespace cf /** Collaborative filtering. */ {

/**
 * Template class for factorizer traits. This stores the default values for the
 * variables to be assumed for a given factorizer. If any of the factorizers
 * needs to have a different value for the traits, a template specialization has
 * be wriiten for that factorizer. An example can be found in the module for
 * Regularized SVD.
 */
template<typename FactorizerType>
struct FactorizerTraits
{
  /**
   * If true, then the passed data matrix is used for factorizer.Apply().
   * Otherwise, it is modified into a form suitable for factorization.
   */
  static const bool UsesCoordinateList = false;
};

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
 * CF cf(data); // Default options.
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
 * @tparam FactorizerType The type of matrix factorization to use to decompose
 *     the rating matrix (a W and H matrix).  This must implement the method
 *     Apply(arma::sp_mat& data, size_t rank, arma::mat& W, arma::mat& H).
 */
class CF
{
 public:
  /**
   * Initialize the CF object without performing any factorization.  Be sure to
   * call Train() before calling GetRecommendations() or any other functions!
   */
  CF(const size_t numUsersForSimilarity = 5,
     const size_t rank = 0);

  /**
   * Initialize the CF object using an instantiated factorizer, immediately
   * factorizing the given data to create a model. There are parameters that can
   * be set; default values are provided for each of them. If the rank is left
   * unset (or is set to 0), a simple density-based heuristic will be used to
   * choose a rank.
   *
   * The provided dataset should be a coordinate list; that is, a 3-row matrix
   * where each column corresponds to a (user, item, rating) entry in the
   * matrix.
   *
   * @param data Data matrix: coordinate list or dense matrix.
   * @param factorizer Instantiated factorizer object.
   * @param numUsersForSimilarity Size of the neighborhood.
   * @param rank Rank parameter for matrix factorization.
   */
  template<typename FactorizerType = amf::NMFALSFactorizer>
  CF(const arma::mat& data,
     FactorizerType factorizer = FactorizerType(),
     const size_t numUsersForSimilarity = 5,
     const size_t rank = 0);

  /**
   * Initialize the CF object using an instantiated factorizer, immediately
   * factorizing the given data to create a model. There are parameters that can
   * be set; default values are provided for each of them. If the rank is left
   * unset (or is set to 0), a simple density-based heuristic will be used to
   * choose a rank. Data will be considered in the format of items vs. users and
   * will be passed directly to the factorizer without cleaning.  This overload
   * of the constructor will only be available if the factorizer does not use a
   * coordinate list (i.e. if UsesCoordinateList is false).
   *
   * The U and T template parameters are for SFINAE, so that this overload is
   * only available when the FactorizerType uses a coordinate list.
   *
   * @param data Sparse matrix data.
   * @param factorizer Instantiated factorizer object.
   * @param numUsersForSimilarity Size of the neighborhood.
   * @param rank Rank parameter for matrix factorization.
   */
  template<typename FactorizerType = amf::NMFALSFactorizer>
  CF(const arma::sp_mat& data,
     FactorizerType factorizer = FactorizerType(),
     const size_t numUsersForSimilarity = 5,
     const size_t rank = 0,
     const typename boost::disable_if_c<
         FactorizerTraits<FactorizerType>::UsesCoordinateList>::type* = 0);

  /**
   * Train the CF model (i.e. factorize the input matrix) using the parameters
   * that have already been set for the model (specifically, the rank
   * parameter), and optionally, using the given FactorizerType.
   *
   * @param data Input dataset; coordinate list or dense matrix.
   * @param factorizer Instantiated factorizer.
   */
  template<typename FactorizerType = amf::NMFALSFactorizer>
  void Train(const arma::mat& data,
             FactorizerType factorizer = FactorizerType());

  /**
   * Train the CF model (i.e. factorize the input matrix) using the parameters
   * that have already been set for the model (specifically, the rank
   * parameter), and optionally, using the given FactorizerType.
   *
   * @param data Sparse matrix data.
   * @param factorizer Instantiated factorizer.
   */
  template<typename FactorizerType = amf::NMFALSFactorizer>
  void Train(const arma::sp_mat& data,
             FactorizerType factorizer = FactorizerType(),
             const typename boost::disable_if_c<
                 FactorizerTraits<FactorizerType>::UsesCoordinateList>::type*
                 = 0);

  //! Sets number of users for calculating similarity.
  void NumUsersForSimilarity(const size_t num)
  {
    if (num < 1)
    {
      Log::Warn << "CF::NumUsersForSimilarity(): invalid value (< 1) "
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
   * @param numRecs Number of Recommendations
   * @param recommendations Matrix to save recommendations into.
   */
  void GetRecommendations(const size_t numRecs,
                          arma::Mat<size_t>& recommendations);

  /**
   * Generates the given number of recommendations for the specified users.
   *
   * @param numRecs Number of Recommendations
   * @param recommendations Matrix to save recommendations
   * @param users Users for which recommendations are to be generated
   */
  void GetRecommendations(const size_t numRecs,
                          arma::Mat<size_t>& recommendations,
                          const arma::Col<size_t>& users);

  //! Converts the User, Item, Value Matrix to User-Item Table
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
   * Serialize the CF model to the given archive.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

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
}; // class CF

} // namespace cf
} // namespace mlpack

// Include implementation of templated functions.
#include "cf_impl.hpp"

#endif
