/**
 * @file cf.cpp
 * @author Mudit Raj Gupta
 *
 * Collaborative Filtering.
 *
 * Implementation of CF class to perform Collaborative Filtering on the
 * specified data set.
 *
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

namespace mlpack {
namespace cf {

/**
 * Construct the CF object.
 */
template<typename FactorizerType>
CF<FactorizerType>::CF(arma::mat& data,
                       const size_t numUsersForSimilarity,
                       const size_t rank) :
    data(data),
    numUsersForSimilarity(numUsersForSimilarity),
    rank(rank),
    factorizer()
{
  // Validate neighbourhood size.
  if (numUsersForSimilarity < 1)
  {
    Log::Warn << "CF::CF(): neighbourhood size should be > 0("
        << numUsersForSimilarity << " given). Setting value to 5.\n";
    //Setting Default Value of 5
    this->numUsersForSimilarity = 5;
  }

  CleanData();
}

template<typename FactorizerType>
void CF<FactorizerType>::GetRecommendations(const size_t numRecs,
                                            arma::Mat<size_t>& recommendations)
{
  // Generate list of users.  Maybe it would be more efficient to pass an empty
  // users list, and then have the other overload of GetRecommendations() assume
  // that if users is empty, then recommendations should be generated for all
  // users?
  arma::Col<size_t> users = arma::linspace<arma::Col<size_t> >(0,
      cleanedData.n_cols - 1, cleanedData.n_cols);

  // Call the main overload for recommendations.
  GetRecommendations(numRecs, recommendations, users);
}

template<typename FactorizerType>
void CF<FactorizerType>::GetRecommendations(const size_t numRecs,
                                            arma::Mat<size_t>& recommendations,
                                            arma::Col<size_t>& users)
{
  // Base function for calculating recommendations.

  // Check if the user wanted us to choose a rank for them.
  if (rank == 0)
  {
    // This is a simple heuristic that picks a rank based on the density of the
    // dataset between 5 and 105.
    const double density = (cleanedData.n_nonzero * 100.0) / cleanedData.n_elem;
    const size_t rankEstimate = size_t(density) + 5;

    // Set to heuristic value.
    Log::Info << "No rank given for decomposition; using rank of "
        << rankEstimate << " calculated by density-based heuristic."
        << std::endl;
    rank = rankEstimate;
  }

  // Operations independent of the query:
  // Decompose the sparse data matrix to user and data matrices.
  factorizer.Apply(cleanedData, rank, w, h);

  // Generate new table by multiplying approximate values.
  rating = w * h;

  // Now, we will use the decomposed w and h matrices to estimate what the user
  // would have rated items as, and then pick the best items.

  // Temporarily store feature vector of queried users.
  arma::mat query(rating.n_rows, users.n_elem);

  // Select feature vectors of queried users.
  for (size_t i = 0; i < users.n_elem; i++)
    query.col(i) = rating.col(users(i));

  // Temporary storage for neighborhood of the queried users.
  arma::Mat<size_t> neighborhood;

  // Calculate the neighborhood of the queried users.
  // This should be a templatized option.
  neighbor::AllkNN a(rating, query);
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
  recommendations.fill(cleanedData.n_rows); // Invalid item number.
  arma::mat values(numRecs, users.n_elem);
  values.fill(-DBL_MAX); // The smallest possible value.
  for (size_t i = 0; i < users.n_elem; i++)
  {
    // Look through the averages column corresponding to the current user.
    for (size_t j = 0; j < averages.n_rows; ++j)
    {
      // Ensure that the user hasn't already rated the item.
      if (cleanedData(j, users(i)) != 0.0)
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

        // Now insert it into the list.
        InsertNeighbor(i, insertPosition, j, value, recommendations,
            values);
      }
    }

    // If we were not able to come up with enough recommendations, issue a
    // warning.
    if (recommendations(values.n_rows - 1, i) == cleanedData.n_rows + 1)
      Log::Warn << "Could not provide " << values.n_rows << " recommendations "
          << "for user " << users(i) << " (not enough un-rated items)!" << std::endl;
  }
}

template<typename FactorizerType>
void CF<FactorizerType>::CleanData()
{
  // Generate list of locations for batch insert constructor for sparse
  // matrices.
  arma::umat locations(2, data.n_cols);
  arma::vec values(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // We have to transpose it because items are rows, and users are columns.
    locations(1, i) = ((arma::uword) data(0, i));
    locations(0, i) = ((arma::uword) data(1, i));
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
template<typename FactorizerType>
void CF<FactorizerType>::InsertNeighbor(const size_t queryIndex,
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

// Return string of object.
template<typename FactorizerType>
std::string CF<FactorizerType>::ToString() const
{
  std::ostringstream convert;
  convert << "Collaborative Filtering [" << this << "]" << std::endl;
  //convert << "  Number of Recommendations: " << numRecs << std::endl;
  //convert << "  Number of Users for Similarity: " << numUsersForSimilarity;
  //convert << std::endl;
  //convert << "  Data: " << data.n_rows << "x" << data.n_cols << std::endl;
  return convert.str();
}

}; // namespace mlpack
}; // namespace cf
