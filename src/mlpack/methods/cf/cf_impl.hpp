/**
 * @file methods/cf/cf_impl.hpp
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
#ifndef MLPACK_METHODS_CF_CF_IMPL_HPP
#define MLPACK_METHODS_CF_CF_IMPL_HPP

// In case it hasn't been included yet.
#include "cf.hpp"

namespace mlpack {

// Default CF constructor.
template<typename DecompositionPolicy,
         typename NormalizationType>
CFType<DecompositionPolicy,
       NormalizationType>::
CFType(const size_t numUsersForSimilarity,
       const size_t rank) :
    numUsersForSimilarity(numUsersForSimilarity),
    rank(rank)
{
  // Validate neighbourhood size.
  if (numUsersForSimilarity < 1)
  {
    Log::Warn << "CFType::CFType(): neighbourhood size should be > 0 ("
        << numUsersForSimilarity << " given). Setting value to 5.\n";
    // Set default value of 5.
    this->numUsersForSimilarity = 5;
  }
}

/**
 * Construct the CF object using an instantiated decomposition policy.
 */
template<typename DecompositionPolicy,
         typename NormalizationType>
template<typename MatType>
CFType<DecompositionPolicy,
       NormalizationType>::
CFType(const MatType& data,
       const DecompositionPolicy& decomposition,
       const size_t numUsersForSimilarity,
       const size_t rank,
       const size_t maxIterations,
       const double minResidue,
       const bool mit) :
    numUsersForSimilarity(numUsersForSimilarity),
    rank(rank)
{
  // Validate neighbourhood size.
  if (numUsersForSimilarity < 1)
  {
    Log::Warn << "CFType::CFType(): neighbourhood size should be > 0 ("
        << numUsersForSimilarity << " given). Setting value to 5.\n";
    // Set default value of 5.
    this->numUsersForSimilarity = 5;
  }

  Train(data, decomposition, maxIterations, minResidue, mit);
}

// Train when data is given in dense matrix form.
template<typename DecompositionPolicy,
         typename NormalizationType>
void CFType<DecompositionPolicy,
            NormalizationType>::
Train(const arma::mat& data,
      const DecompositionPolicy& decomposition,
      const size_t maxIterations,
      const double minResidue,
      const bool mit)
{
  this->decomposition = decomposition;

  // Make a copy of data before performing normalization.
  arma::mat normalizedData(data);
  normalization.Normalize(normalizedData);
  CleanData(normalizedData, cleanedData);

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
    this->rank = rankEstimate;
  }

  // Decompose the data matrix (which is in coordinate list form) to user and
  // data matrices.
  this->decomposition.Apply(
      normalizedData, cleanedData, rank, maxIterations, minResidue, mit);
}

// Train when data is given as sparse matrix of user item table.
template<typename DecompositionPolicy,
         typename NormalizationType>
void CFType<DecompositionPolicy,
            NormalizationType>::
Train(const arma::sp_mat& data,
      const DecompositionPolicy& decomposition,
      const size_t maxIterations,
      const double minResidue,
      const bool mit)
{
  this->decomposition = decomposition;

  // data is not used in the following decomposition.Apply() method, so we only
  // need to Normalize cleanedData.
  cleanedData = data;
  normalization.Normalize(cleanedData);

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
    this->rank = rankEstimate;
  }

  // Decompose the data matrix (which is in coordinate list form) to user and
  // data matrices.
  this->decomposition.Apply(
      data, cleanedData, rank, maxIterations, minResidue, mit);
}

template<typename DecompositionPolicy,
         typename NormalizationType>
template<typename NeighborSearchPolicy,
         typename InterpolationPolicy>
void CFType<DecompositionPolicy,
            NormalizationType>::
GetRecommendations(const size_t numRecs,
                   arma::Mat<size_t>& recommendations)
{
  // Generate list of users.  Maybe it would be more efficient to pass an empty
  // users list, and then have the other overload of GetRecommendations() assume
  // that if users is empty, then recommendations should be generated for all
  // users?
  arma::Col<size_t> users = arma::linspace<arma::Col<size_t> >(0,
      cleanedData.n_cols - 1, cleanedData.n_cols);

  // Call the main overload for recommendations.
  GetRecommendations<NeighborSearchPolicy,
                     InterpolationPolicy>(numRecs, recommendations, users);
}

template<typename DecompositionPolicy,
         typename NormalizationType>
template<typename NeighborSearchPolicy,
         typename InterpolationPolicy>
void CFType<DecompositionPolicy,
            NormalizationType>::
GetRecommendations(const size_t numRecs,
                   arma::Mat<size_t>& recommendations,
                   const arma::Col<size_t>& users)
{
  // Temporary storage for neighborhood of the queried users.
  arma::Mat<size_t> neighborhood;
  // Resulting similarities.
  arma::mat similarities;

  // Calculate the neighborhood of the queried users.  Note that the query user
  // is part of the neighborhood---this is intentional.  We want to use the
  // weighted sum of both the query user and the local neighborhood of the
  // query user.
  // Calculate the neighborhood of the queried users.
  decomposition.template GetNeighborhood<NeighborSearchPolicy>(
      users, numUsersForSimilarity, neighborhood, similarities);

  // Generate recommendations for each query user by finding the maximum numRecs
  // elements in the ratings vector.
  recommendations.set_size(numRecs, users.n_elem);
  arma::mat values(numRecs, users.n_elem);
  recommendations.fill(SIZE_MAX);
  values.fill(DBL_MAX);

  // Initialization of an InterpolationPolicy object should be put ahead of the
  // following loop, because the initialization may takes a relatively long
  // time and we don't want to repeat the initialization process in each loop.
  InterpolationPolicy interpolation(cleanedData);

  for (size_t i = 0; i < users.n_elem; ++i)
  {
    // First, calculate the weighted sum of neighborhood values.
    arma::vec ratings;
    ratings.zeros(cleanedData.n_rows);

    // Calculate interpolation weights.
    arma::vec weights(numUsersForSimilarity);
    interpolation.GetWeights(weights, decomposition, users(i),
        neighborhood.col(i), similarities.col(i), cleanedData);

    for (size_t j = 0; j < neighborhood.n_rows; ++j)
    {
      arma::vec neighborRatings;
      decomposition.GetRatingOfUser(neighborhood(j, i), neighborRatings);
      ratings += weights(j) * neighborRatings;
    }

    // Let's build the list of candidate recomendations for the given user.
    // Default candidate: the smallest possible value and invalid item number.
    const Candidate def = std::make_pair(-DBL_MAX, cleanedData.n_rows);
    std::vector<Candidate> vect(numRecs, def);
    using CandidateList =
        std::priority_queue<Candidate, std::vector<Candidate>, CandidateCmp>;
    CandidateList pqueue(CandidateCmp(), std::move(vect));

    // Look through the ratings column corresponding to the current user.
    for (size_t j = 0; j < ratings.n_rows; ++j)
    {
      // Ensure that the user hasn't already rated the item.
      // The algorithm omits rating of zero. Thus, when normalizing original
      // ratings in Normalize(), if normalized rating equals zero, it is set
      // to the smallest positive double value.
      if (cleanedData(j, users(i)) != 0.0)
        continue; // The user already rated the item.

      // Is the estimated value better than the worst candidate?
      // Denormalize rating before comparison.
      double realRating = normalization.Denormalize(users(i), j, ratings[j]);
      if (realRating > pqueue.top().first)
      {
        Candidate c = std::make_pair(realRating, j);
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
template<typename DecompositionPolicy,
         typename NormalizationType>
template<typename NeighborSearchPolicy,
         typename InterpolationPolicy>
double CFType<DecompositionPolicy,
              NormalizationType>::
Predict(const size_t user, const size_t item) const
{
  // First, we need to find the nearest neighbors of the given user.
  // We'll use the same technique as for GetRecommendations().

  // Temporary storage for neighborhood of the queried users.
  arma::Mat<size_t> neighborhood;
  // Resulting similarities.
  arma::mat similarities;

  // Calculate the neighborhood of the queried users.  Note that the query user
  // is part of the neighborhood---this is intentional.  We want to use the
  // weighted sum of both the query user and the local neighborhood of the
  // query user.
  // Calculate the neighborhood of the queried users.
  arma::Col<size_t> users(1);
  users(0) = user;
  decomposition.template GetNeighborhood<NeighborSearchPolicy>(
      users, numUsersForSimilarity, neighborhood, similarities);

  arma::vec weights(numUsersForSimilarity);

  // Calculate interpolation weights.
  InterpolationPolicy interpolation(cleanedData);
  interpolation.GetWeights(weights, decomposition, user,
      neighborhood.col(0), similarities.col(0), cleanedData);

  double rating = 0; // We'll take the weighted sum of neighborhood values.

  for (size_t j = 0; j < neighborhood.n_rows; ++j)
    rating += weights(j) * decomposition.GetRating(neighborhood(j, 0), item);

  // Denormalize rating and return.
  double realRating = normalization.Denormalize(user, item, rating);
  return realRating;
}

// Predict the rating for a group of user/item combinations.
template<typename DecompositionPolicy,
         typename NormalizationType>
template<typename NeighborSearchPolicy,
         typename InterpolationPolicy>
void CFType<DecompositionPolicy,
            NormalizationType>::
Predict(const arma::Mat<size_t>& combinations,
        arma::vec& predictions) const
{
  // Now, we must determine those query indices we need to find the nearest
  // neighbors for.  This is easiest if we just sort the combinations matrix.
  arma::Mat<size_t> sortedCombinations(combinations.n_rows,
                                       combinations.n_cols);
  arma::uvec ordering = arma::sort_index(combinations.row(0).t());
  for (size_t i = 0; i < ordering.n_elem; ++i)
    sortedCombinations.col(i) = combinations.col(ordering[i]);

  // Now, we have to get the list of unique users we will be searching for.
  arma::Col<size_t> users = arma::unique(combinations.row(0).t());

  // Temporary storage for neighborhood of the queried users.
  arma::Mat<size_t> neighborhood;
  // Resulting similarities.
  arma::mat similarities;

  // Calculate the neighborhood of the queried users.  Note that the query user
  // is part of the neighborhood---this is intentional.  We want to use the
  // weighted sum of both the query user and the local neighborhood of the
  // query user.
  // Calculate the neighborhood of the queried users.
  decomposition.template GetNeighborhood<NeighborSearchPolicy>(
      users, numUsersForSimilarity, neighborhood, similarities);

  arma::mat weights(numUsersForSimilarity, users.n_elem);

  // Calculate interpolation weights.
  InterpolationPolicy interpolation(cleanedData);
  for (size_t i = 0; i < users.n_elem; ++i)
  {
    interpolation.GetWeights(weights.col(i), decomposition, users[i],
        neighborhood.col(i), similarities.col(i), cleanedData);
  }

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
    {
      rating += weights(j, user) * decomposition.GetRating(
          neighborhood(j, user), sortedCombinations(1, i));
    }

    predictions(ordering[i]) = rating;
  }

  // Denormalize ratings.
  normalization.Denormalize(combinations, predictions);
}

template<typename DecompositionPolicy,
         typename NormalizationType>
void CFType<DecompositionPolicy,
            NormalizationType>::
CleanData(const arma::mat& data, arma::sp_mat& cleanedData)
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

    // The algorithm omits rating of zero. Thus, when normalizing original
    // ratings in Normalize(), if normalized rating equals zero, it is set
    // to the smallest positive double value.
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

//! Serialize the model.
template<typename DecompositionPolicy,
         typename NormalizationType>
template<typename Archive>
void CFType<DecompositionPolicy,
            NormalizationType>::
serialize(Archive& ar, const uint32_t /* version */)
{
  // This model is simple; just serialize all the members. No special handling
  // required.
  ar(CEREAL_NVP(numUsersForSimilarity));
  ar(CEREAL_NVP(rank));
  ar(CEREAL_NVP(decomposition));
  ar(CEREAL_NVP(cleanedData));
  ar(CEREAL_NVP(normalization));
}

} // namespace mlpack

#endif
