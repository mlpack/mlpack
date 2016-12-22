/**
 * @file cf.cpp
 * @author Mudit Raj Gupta
 * @author Sumedh Ghaisas
 *
 * Collaborative Filtering.
 *
 * Implementation of CF class to perform Collaborative Filtering on the
 * specified data set.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "cf.hpp"

namespace mlpack {
namespace cf {

// Default CF constructor.
CF::CF(const size_t numUsersForSimilarity,
       const size_t rank) :
    numUsersForSimilarity(numUsersForSimilarity),
    rank(rank)
{
  // Validate neighbourhood size.
  if (numUsersForSimilarity < 1)
  {
    Log::Warn << "CF::CF(): neighbourhood size should be > 0 ("
        << numUsersForSimilarity << " given). Setting value to 5.\n";
    // Set default value of 5.
    this->numUsersForSimilarity = 5;
  }
}

void CF::GetRecommendations(const size_t numRecs,
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

void CF::GetRecommendations(const size_t numRecs,
                            arma::Mat<size_t>& recommendations,
                            const arma::Col<size_t>& users)
{
  // We want to avoid calculating the full rating matrix, so we will do nearest
  // neighbor search only on the H matrix, using the observation that if the
  // rating matrix X = W*H, then d(X.col(i), X.col(j)) = d(W H.col(i), W
  // H.col(j)).  This can be seen as nearest neighbor search on the H matrix
  // with the Mahalanobis distance where M^{-1} = W^T W.  So, we'll decompose
  // M^{-1} = L L^T (the Cholesky decomposition), and then multiply H by L^T.
  // Then we can perform nearest neighbor search.
  arma::mat l = arma::chol(w.t() * w);
  arma::mat stretchedH = l * h; // Due to the Armadillo API, l is L^T.

  // Now, we will use the decomposed w and h matrices to estimate what the user
  // would have rated items as, and then pick the best items.

  // Temporarily store feature vector of queried users.
  arma::mat query(stretchedH.n_rows, users.n_elem);

  // Select feature vectors of queried users.
  for (size_t i = 0; i < users.n_elem; i++)
    query.col(i) = stretchedH.col(users(i));

  // Temporary storage for neighborhood of the queried users.
  arma::Mat<size_t> neighborhood;

  // Calculate the neighborhood of the queried users.
  // This should be a templatized option.
  neighbor::KNN a(stretchedH);
  arma::mat resultingDistances; // Temporary storage.
  a.Search(query, numUsersForSimilarity, neighborhood, resultingDistances);

  // Generate recommendations for each query user by finding the maximum numRecs
  // elements in the averages matrix.
  recommendations.set_size(numRecs, users.n_elem);
  arma::mat values(numRecs, users.n_elem);

  for (size_t i = 0; i < users.n_elem; i++)
  {
    // First, calculate average of neighborhood values.
    arma::vec averages;
    averages.zeros(cleanedData.n_rows);

    for (size_t j = 0; j < neighborhood.n_rows; ++j)
      averages += w * h.col(neighborhood(j, i));
    averages /= neighborhood.n_rows;

    // Let's build the list of candidate recomendations for the given user.
    // Default candidate: the smallest possible value and invalid item number.
    const Candidate def = std::make_pair(-DBL_MAX, cleanedData.n_rows);
    std::vector<Candidate> vect(numRecs, def);
    typedef std::priority_queue<Candidate, std::vector<Candidate>, CandidateCmp>
        CandidateList;
    CandidateList pqueue(CandidateCmp(), std::move(vect));

    // Look through the averages column corresponding to the current user.
    for (size_t j = 0; j < averages.n_rows; ++j)
    {
      // Ensure that the user hasn't already rated the item.
      if (cleanedData(j, users(i)) != 0.0)
        continue; // The user already rated the item.


      // Is the estimated value better than the worst candidate?
      if (averages[i] > pqueue.top().first)
      {
        Candidate c = std::make_pair(averages[j], j);
        pqueue.pop();
        pqueue.push(c);
      }
    }

    for (size_t p = 1; p <= numRecs; p++)
    {
      recommendations(numRecs - p, i) = pqueue.top().second;
      values(numRecs - p, i) = pqueue.top().first;
      pqueue.pop();
    }

    // If we were not able to come up with enough recommendations, issue a
    // warning.
    if (recommendations(numRecs - 1, i) == def.second)
      Log::Warn << "Could not provide " << numRecs << " recommendations "
          << "for user " << users(i) << " (not enough un-rated items)!"
          << std::endl;
  }
}

// Predict the rating for a single user/item combination.
double CF::Predict(const size_t user, const size_t item) const
{
  // First, we need to find the nearest neighbors of the given user.
  // We'll use the same technique as for GetRecommendations().

  // We want to avoid calculating the full rating matrix, so we will do nearest
  // neighbor search only on the H matrix, using the observation that if the
  // rating matrix X = W*H, then d(X.col(i), X.col(j)) = d(W H.col(i), W
  // H.col(j)).  This can be seen as nearest neighbor search on the H matrix
  // with the Mahalanobis distance where M^{-1} = W^T W.  So, we'll decompose
  // M^{-1} = L L^T (the Cholesky decomposition), and then multiply H by L^T.
  // Then we can perform nearest neighbor search.
  arma::mat l = arma::chol(w.t() * w);
  arma::mat stretchedH = l * h; // Due to the Armadillo API, l is L^T.

  // Now, we will use the decomposed w and h matrices to estimate what the user
  // would have rated items as, and then pick the best items.

  // Temporarily store feature vector of queried users.
  arma::mat query = stretchedH.col(user);

  // Temporary storage for neighborhood of the queried users.
  arma::Mat<size_t> neighborhood;

  // Calculate the neighborhood of the queried users.
  // This should be a templatized option.
  neighbor::KNN a(stretchedH, neighbor::SINGLE_TREE_MODE);
  arma::mat resultingDistances; // Temporary storage.

  a.Search(query, numUsersForSimilarity, neighborhood, resultingDistances);

  double rating = 0; // We'll take the average of neighborhood values.

  for (size_t j = 0; j < neighborhood.n_rows; ++j)
    rating += arma::as_scalar(w.row(item) * h.col(neighborhood(j, 0)));
  rating /= neighborhood.n_rows;

  return rating;
}

// Predict the rating for a group of user/item combinations.
void CF::Predict(const arma::Mat<size_t>& combinations,
                 arma::vec& predictions) const
{
  // First, for nearest neighbor search, stretch the H matrix.
  arma::mat l = arma::chol(w.t() * w);
  arma::mat stretchedH = l * h; // Due to the Armadillo API, l is L^T.

  // Now, we must determine those query indices we need to find the nearest
  // neighbors for.  This is easiest if we just sort the combinations matrix.
  arma::Mat<size_t> sortedCombinations(combinations.n_rows,
                                       combinations.n_cols);
  arma::uvec ordering = arma::sort_index(combinations.row(0).t());
  for (size_t i = 0; i < ordering.n_elem; ++i)
    sortedCombinations.col(i) = combinations.col(ordering[i]);

  // Now, we have to get the list of unique users we will be searching for.
  arma::Col<size_t> users = arma::unique(combinations.row(0).t());

  // Assemble our query matrix from the stretchedH matrix.
  arma::mat queries(stretchedH.n_rows, users.n_elem);
  for (size_t i = 0; i < queries.n_cols; ++i)
    queries.col(i) = stretchedH.col(users[i]);

  // Now calculate the neighborhood of these users.
  neighbor::KNN a(stretchedH);
  arma::mat distances;
  arma::Mat<size_t> neighborhood;

  a.Search(queries, numUsersForSimilarity, neighborhood, distances);

  // Now that we have the neighborhoods we need, calculate the predictions.
  predictions.set_size(combinations.n_cols);

  size_t user = 0; // Cumulative user count, because we are doing it in order.
  for (size_t i = 0; i < sortedCombinations.n_cols; ++i)
  {
    // Could this be made faster by calculating dot products for multiple items
    // at once?
    double rating = 0.0;

    // Map the combination's user to the user ID used for kNN.
    while (users[user] < sortedCombinations(0, i))
      ++user;

    for (size_t j = 0; j < neighborhood.n_rows; ++j)
      rating += arma::as_scalar(w.row(sortedCombinations(1, i)) *
          h.col(neighborhood(j, user)));
    rating /= neighborhood.n_rows;

    predictions(ordering[i]) = rating;
  }
}

void CF::CleanData(const arma::mat& data, arma::sp_mat& cleanedData)
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
    if (values(i) == 0)
      Log::Warn << "User rating of 0 ignored for user " << locations(1, i)
          << ", item " << locations(0, i) << "." << std::endl;
  }

  // Find maximum user and item IDs.
  const size_t maxItemID = (size_t) max(locations.row(0)) + 1;
  const size_t maxUserID = (size_t) max(locations.row(1)) + 1;

  // Fill sparse matrix.
  cleanedData = arma::sp_mat(locations, values, maxItemID, maxUserID);
}

} // namespace mlpack
} // namespace cf
