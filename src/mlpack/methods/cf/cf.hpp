/**
 * @file cf.hpp
 * @author Mudit Raj Gupta
 *
 * Collaborative filtering.
 *
 * Defines the CF class to perform collaborative filtering on the specified data
 * set using alternating least squares (ALS).
 *
 * This file is part of MLPACK 1.0.7.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_CF_CF_HPP
#define __MLPACK_METHODS_CF_CF_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
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
 * arma::mat recommendations; // Recommendations
 * size_t numRecommendations = 10;
 *
 * CF<> cf(data); // Default options.
 *
 * // Generate the default number of recommendations for all users.
 * cf.GenerateRecommendations(recommendations);
 *
 * // Generate the default number of recommendations for specified users.
 * cf.GenerateRecommendations(recommendations, users);
 *
 * // Generate 10 recommendations for specified users.
 * cf.GenerateRecommendations(recommendations, users, numRecommendations);
 *
 * @endcode
 *
 * The data matrix is a (user, item, rating) table.  Each column in the matrix
 * should have three rows.  The first represents the user; the second represents
 * the item; and the third represents the rating.  The user and item, while they
 * are in a matrix that holds doubles, should hold integer (or size_t) values.
 */
class CF
{
 public:
  /**
   * Create a CF object and (optionally) set the parameters with which
   * collaborative filtering will be run.
   *
   * @param data Initial (user,item,rating) matrix.
   * @param numRecs Desired number of recommendations for each user.
   * @param numUsersForSimilarity Size of the neighborhood.
   */
  CF(const size_t numRecs,const size_t numUsersForSimilarity,
     arma::mat& data);

  /**
   * Create a CF object and (optionally) set the parameters which CF
   * will be run with.
   *
   * @param data Initial User,Item,Rating Matrix
   * @param numRecs Number of Recommendations for each user.
   */
  CF(const size_t numRecs, arma::mat& data);

  /**
   * Create a CF object and (optionally) set the parameters which CF
   * will be run with.
   *
   * @param data Initial User,Item,Rating Matrix
   */
  CF(arma::mat& data);

  //! Sets number of Recommendations.
  void NumRecs(size_t recs)
  {
    if (recs < 1)
    {
      Log::Warn << "CF::NumRecs(): invalid value (< 1) "
          "ignored." << std::endl;
      return;
    }
    this->numRecs = recs;
  }

  //! Gets numRecs
  size_t NumRecs()
  {
    return numRecs;
  }

  //! Sets number of user for calculating similarity.
  void NumUsersForSimilarity(size_t num)
  {
    if (num < 1)
    {
      Log::Warn << "CF::NumUsersForSimilarity(): invalid value (< 1) "
          "ignored." << std::endl;
      return;
    }
    this->numUsersForSimilarity = num;
  }

  //! Gets number of users for calculating similarity/
  size_t NumUsersForSimilarity()
  {
    return numUsersForSimilarity;
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
   * Generates default number of recommendations for all users.
   *
   * @param recommendations Matrix to save recommendations into.
   */
  void GetRecommendations(arma::Mat<size_t>& recommendations);

  /**
   * Generates default number of recommendations for specified users.
   *
   * @param recommendations Matrix to save recommendations
   * @param users Users for which recommendations are to be generated
   */
  void GetRecommendations(arma::Mat<size_t>& recommendations,
                          arma::Col<size_t>& users);

  /**
   * Generates a fixed number of recommendations for specified users.
   *
   * @param recommendations Matrix to save recommendations
   * @param users Users for which recommendations are to be generated
   * @param num Number of Recommendations
   */
  void GetRecommendations(arma::Mat<size_t>& recommendations,
                          arma::Col<size_t>& users, size_t num);

  /**
   * Generates a fixed number of recommendations for specified users.
   *
   * @param recommendations Matrix to save recommendations
   * @param users Users for which recommendations are to be generated
   * @param num Number of Recommendations
   * @param neighbours Number of user to be considered while calculating
   *        the neighbourhood
   */
  void GetRecommendations(arma::Mat<size_t>& recommendations,
                          arma::Col<size_t>& users, size_t num,
                          size_t neighbours);

 private:
  //! Number of recommendations.
  size_t numRecs;
  //! Number of users for similarity.
  size_t numUsersForSimilarity;
  //! User matrix.
  arma::mat w;
  //! Item matrix.
  arma::mat h;
  //! Rating matrix.
  arma::mat rating;
  //! Initial data matrix.
  arma::mat data;
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

#endif
