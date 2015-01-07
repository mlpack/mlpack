/**
 * @file cf.hpp
 * @author Mudit Raj Gupta
 *
 * Collaborative filtering.
 *
 * Defines the CF class to perform collaborative filtering on the specified data
 * set using alternating least squares (ALS).
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_CF_CF_HPP
#define __MLPACK_METHODS_CF_CF_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/amf/update_rules/nmf_als.hpp>
#include <mlpack/methods/amf/termination_policies/simple_residue_termination.hpp>
#include <set>
#include <map>
#include <iostream>

namespace mlpack {
namespace cf /** Collaborative filtering. */{

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
 * CF<> cf(data); // Default options.
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
template<
    typename FactorizerType = amf::AMF<amf::SimpleResidueTermination,
                                       amf::RandomInitialization, 
                                       amf::NMFALSUpdate> >
class CF
{
 public:
  /**
   * Initialize the CF object. Store a reference to the data that we
   * will be using. There are parameters that can be set; default values
   * are provided for each of them.  If the rank is left unset (or is set to 0),
   * a simple density-based heuristic will be used to choose a rank.
   *
   * @param data Initial (user, item, rating) matrix.
   * @param numUsersForSimilarity Size of the neighborhood.
   * @param rank Rank parameter for matrix factorization.
   */
  CF(arma::mat& data,
     const size_t numUsersForSimilarity = 5,
     const size_t rank = 0);

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

  //! Sets factorizer for NMF
  void Factorizer(const FactorizerType& f)
  {
    this->factorizer = f;
  }

  //! Get the User Matrix.
  const arma::mat& W() const { return w; }
  //! Get the Item Matrix.
  const arma::mat& H() const { return h; }
  //! Get the Rating Matrix.
  const arma::mat& Rating() const { return rating; }
  //! Get the data matrix.
  const arma::mat& Data() const { return data; }
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
                          arma::Col<size_t>& users);

  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;

 private:
  //! Initial data matrix.
  arma::mat data;
  //! Number of users for similarity.
  size_t numUsersForSimilarity;
  //! Rank used for matrix factorization.
  size_t rank;
  //! Instantiated factorizer object.
  FactorizerType factorizer;
  //! User matrix.
  arma::mat w;
  //! Item matrix.
  arma::mat h;
  //! Rating matrix.
  arma::mat rating;
  //! Cleaned data matrix.
  arma::sp_mat cleanedData;
  //! Converts the User, Item, Value Matrix to User-Item Table
  void CleanData();

  /**
   * Helper function to insert a point into the recommendation matrices.
   *
   * @param queryIndex Index of point whose recommendations we are inserting
   *     into.
   * @param pos Position in list to insert into.
   * @param neighbor Index of item being inserted as a recommendation.
   * @param value Value of recommendation.
   */
  void InsertNeighbor(const size_t queryIndex,
                      const size_t pos,
                      const size_t neighbor,
                      const double value,
                      arma::Mat<size_t>& recommendations,
                      arma::mat& values) const;

}; // class CF

}; // namespace cf
}; // namespace mlpack

//Include implementation
#include "cf_impl.hpp"

#endif
