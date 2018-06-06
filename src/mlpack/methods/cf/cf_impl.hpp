/**
 * @file cf_impl.hpp
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
namespace cf {

/**
 * Construct the CF object using an instantiated decomposition policy.
 */
template<typename MatType, typename DecompositionPolicy>
CFType::CFType(const MatType& data,
               DecompositionPolicy& decomposition,
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
template<typename DecompositionPolicy>
void CFType::Train(const arma::mat& data,
                   DecompositionPolicy& decomposition,
                   const size_t maxIterations,
                   const double minResidue,
                   const bool mit)
{
  CleanData(data, cleanedData);

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
  Timer::Start("cf_factorization");
  decomposition.Apply(data, cleanedData, rank, w,
      h, maxIterations, minResidue, mit);
  Timer::Stop("cf_factorization");
}

// Train when data is given as sparse matrix of user item table.
template<typename DecompositionPolicy>
void CFType::Train(const arma::sp_mat& data,
                   DecompositionPolicy& decomposition,
                   const size_t maxIterations,
                   const double minResidue,
                   const bool mit)
{
  cleanedData = data;

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
  Timer::Start("cf_factorization");
  decomposition.Apply(data, cleanedData, rank, w,
      h, maxIterations, minResidue, mit);
  Timer::Stop("cf_factorization");
}

//! Serialize the model.
template<typename Archive>
void CFType::serialize(Archive& ar, const unsigned int /* version */)
{
  // This model is simple; just serialize all the members. No special handling
  // required.
  ar & BOOST_SERIALIZATION_NVP(numUsersForSimilarity);
  ar & BOOST_SERIALIZATION_NVP(rank);
  ar & BOOST_SERIALIZATION_NVP(w);
  ar & BOOST_SERIALIZATION_NVP(h);
  ar & BOOST_SERIALIZATION_NVP(cleanedData);
}

} // namespace cf
} // namespace mlpack

#endif
