/**
 * @file cf.cpp
 * @author Mudit Raj Gupta
 *
 * Collaborative Filtering.
 *
 * Implementation of CF class to perform Collaborative Filtering on the
 * specified data set.
 *
 * This file is part of MLPACK 1.0.6.
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
#include "cf.hpp"
#include <mlpack/methods/nmf/nmf.hpp>
#include <mlpack/methods/nmf/als_update_rules.hpp>

using namespace mlpack::nmf;
using namespace std;

namespace mlpack {
namespace cf {

/**
 * Construct the CF object.
 */
CF::CF(arma::mat& data) :
     data(data)
{
  Log::Info<<"Constructor (param: input data, default: numRecs;neighbourhood)"<<endl;
  this->numRecs = 5;
  this->numUsersForSimilarity = 5;

  CleanData();
}

CF::CF(const size_t numRecs,arma::mat& data) :
     data(data)
{
  // Validate number of recommendation factor.
  if (numRecs < 1)
  {
    Log::Warn << "CF::CF(): number of recommendations shoud be > 0("
        << numRecs << " given). Setting value to 5.\n";
    //Setting Default Value of 5
    this->numRecs = 5;
  }
  else
    this->numRecs = numRecs;
  this->numUsersForSimilarity = 5;

  CleanData();
}

CF::CF(const size_t numRecs, const size_t numUsersForSimilarity,
     arma::mat& data) :
     data(data)
{
  // Validate number of recommendation factor.
  if (numRecs < 1)
  {
    Log::Warn << "CF::CF(): number of recommendations shoud be > 0("
        << numRecs << " given). Setting value to 5.\n";
    //Setting Default Value of 5
    this->numRecs = 5;
  }
  else
    this->numRecs = numRecs;
  // Validate neighbourhood size.
  if (numUsersForSimilarity < 1)
  {
    Log::Warn << "CF::CF(): neighbourhood size shoud be > 0("
        << numUsersForSimilarity << " given). Setting value to 5.\n";
    //Setting Default Value of 5
    this->numUsersForSimilarity = 5;
  }
  else
    this->numUsersForSimilarity = numUsersForSimilarity;

  CleanData();
}

void CF::GetRecommendations(arma::Mat<size_t>& recommendations)
{
  // Used to save user IDs.
  arma::Col<size_t> users =
    arma::zeros<arma::Col<size_t> >(cleanedData.n_cols, 1);
  // Getting all user IDs.
  for (size_t i = 0; i < cleanedData.n_cols; i++)
    users(i) = i + 1;

  // Calling base function for recommendations.
  GetRecommendations(recommendations, users);
}

void CF::GetRecommendations(arma::Mat<size_t>& recommendations,
                            arma::Col<size_t>& users)
{
  // Base function for calculating recommendations.

  // Operations independent of the query:
  // Decompose the sparse data matrix to user and data matrices.
  // Should this rank be parameterizable?
  size_t rank = 2;

  // Presently only ALS (via NMF) is supported as an optimizer.  This should be
  // converted to a template when more optimizers are available.
  NMF<RandomInitialization, WAlternatingLeastSquaresRule,
      HAlternatingLeastSquaresRule> als(10000, 1e-5);
  als.Apply(cleanedData, rank, w, h);

  // Generate new table by multiplying approximate values.
  rating = w * h;

  // Now, we will use the decomposed w and h matrices to estimate what the user
  // would have rated items as, and then pick the best items.

  // Temporarily store feature vector of queried users.
  arma::mat query(rating.n_rows, users.n_elem);

  // Select feature vectors of queried users.
  for (size_t i = 0; i < users.n_elem; i++)
    query.col(i) = rating.col(users(i) - 1);

  // Temporary storage for neighborhood of the queried users.
  arma::Mat<size_t> neighborhood;

  // Calculate the neighborhood of the queried users.
  // This should be a templatized option.
  AllkNN a(rating, query);
  arma::mat resultingDistances; // Temporary storage.
  a.Search(numUsersForSimilarity, neighborhood, resultingDistances);

  // Temporary storage for storing the average rating for each user in their
  // neighborhood.
  arma::mat averages = arma::zeros<arma::mat>(rating.n_rows, query.n_cols);

  // Iterate over each query user.
  for (size_t i = 0; i < neighborhood.n_cols; ++i)
  {
    // Iterate over each neighbor of the query user.
    for (size_t j = 0; j < neighborhood.n_rows; ++j)
      averages.col(i) += rating.col(neighborhood(j, i));
    // Normalize average.
    averages.col(i) /= neighborhood.n_rows;
  }

  // Generate recommendations for each query user by finding the maximum numRecs
  // elements in the averages matrix.
  recommendations.set_size(numRecs, users.n_elem);
  recommendations.fill(cleanedData.n_rows + 1); // Invalid item number.
  arma::mat values(numRecs, users.n_elem);
  values.fill(-DBL_MAX); // The smallest possible value.
  for (size_t i = 0; i < users.n_elem; i++)
  {
    // Look through the averages column corresponding to the current user.
    for (size_t j = 0; j < averages.n_rows; ++j)
    {
      // Ensure that the user hasn't already rated the item.
      if (cleanedData(j, users(i) - 1) != 0.0)
        continue; // The user already rated the item.

      // Is the estimated value better than the worst candidate?
      const double value = averages(j, i);
      if (value > values(values.n_rows - 1, i))
      {
        // It should be inserted.  Which position?
        size_t insertPosition = values.n_rows - 1;
        while (insertPosition > 0)
        {
          if (value <= values(insertPosition - 1, i))
            break; // The current value is the right one.
          insertPosition--;
        }

        // Now insert it into the list, but insert item (j + 1), not item j,
        // because everything is offset.
        InsertNeighbor(i, insertPosition, j + 1, value, recommendations,
            values);
      }
    }

    // If we were not able to come up with enough recommendations, issue a
    // warning.
    if (recommendations(values.n_rows - 1, i) == cleanedData.n_rows + 1)
      Log::Warn << "Could not provide " << values.n_rows << " recommendations "
          << "for user " << users(i) << " (not enough un-rated items)!" << endl;
  }
}

void CF::GetRecommendations(arma::Mat<size_t>& recommendations,
                            arma::Col<size_t>& users,size_t num)
{
  //Setting Number of Recommendations
  NumRecs(num);
  //Calling Base Function for Recommendations
  GetRecommendations(recommendations,users);
}

void CF::GetRecommendations(arma::Mat<size_t>& recommendations,
                            arma::Col<size_t>& users,size_t num,size_t s)
{
  //Setting number of users that should be used for calculating
  //neighbours
  NumUsersForSimilarity(s);
  //Setting Number of Recommendations
  NumRecs(num);
  //Calling Base Function for Recommendations
  GetRecommendations(recommendations,users,num);
}

void CF::CleanData()
{
  // Generate list of locations for batch insert constructor for sparse
  // matrices.
  arma::umat locations(2, data.n_cols);
  arma::vec values(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // We have to transpose it because items are rows, and users are columns.
    locations(1, i) = ((arma::uword) data(0, i)) - 1;
    locations(0, i) = ((arma::uword) data(1, i)) - 1;
    values(i) = data(2, i);
  }

  // Find maximum user and item IDs.
  const size_t maxItemID = (size_t) max(locations.row(0)) + 1;
  const size_t maxUserID = (size_t) max(locations.row(1)) + 1;

  // Fill sparse matrix.
  cleanedData = arma::sp_mat(locations, values, maxItemID, maxUserID);
}

/**
 * Helper function to insert a point into the recommendation matrices.
 *
 * @param queryIndex Index of point whose recommendations we are inserting into.
 * @param pos Position in list to insert into.
 * @param neighbor Index of item being inserted as a recommendation.
 * @param value Value of recommendation.
 */
void CF::InsertNeighbor(const size_t queryIndex,
                        const size_t pos,
                        const size_t neighbor,
                        const double value,
                        arma::Mat<size_t>& recommendations,
                        arma::mat& values) const
{
  // We only memmove() if there is actually a need to shift something.
  if (pos < (recommendations.n_rows - 1))
  {
    const int len = (values.n_rows - 1) - pos;
    memmove(values.colptr(queryIndex) + (pos + 1),
        values.colptr(queryIndex) + pos,
        sizeof(double) * len);
    memmove(recommendations.colptr(queryIndex) + (pos + 1),
        recommendations.colptr(queryIndex) + pos,
        sizeof(size_t) * len);
  }

  // Now put the new information in the right index.
  values(pos, queryIndex) = value;
  recommendations(pos, queryIndex) = neighbor;
}


}; // namespace mlpack
}; // namespace cf
