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

// Apply the factorizer when coordinate lists are used.
template<typename FactorizerType>
void ApplyFactorizer(FactorizerType& factorizer,
                     const arma::mat& data,
                     const arma::sp_mat& /* cleanedData */,
                     const size_t rank,
                     arma::mat& w,
                     arma::mat& h,
                     const typename std::enable_if_t<FactorizerTraits<
                         FactorizerType>::UsesCoordinateList
                         && !FactorizerTraits<
                         FactorizerType>::UsesQuicSVD>* = 0)
{
  factorizer.Apply(data, rank, w, h);
}

// Apply QuicSVD factorizer.
template<typename FactorizerType>
void ApplyFactorizer(FactorizerType& factorizer,
                     const arma::mat& data,
                     const arma::sp_mat& /* cleanedData */,
                     const size_t rank,
                     arma::mat& w,
                     arma::mat& h,
                     const typename std::enable_if_t<FactorizerTraits<
                         FactorizerType>::UsesCoordinateList
                         && FactorizerTraits<
                         FactorizerType>::UsesQuicSVD>* = 0)
{
  // Check if the given rank is valid.
  size_t r;
  if (rank > data.n_rows || rank > data.n_cols)
  {
    Log::Info << "Rank " << rank << ", given for decomposition is invalid."
        << std::endl;

    r = (data.n_rows > data.n_cols) ? data.n_cols : data.n_rows;
    Log::Info << "Setting decomposition rank to " << rank << std::endl;
  }
  else
  {
    r = rank;
  }

  // Get svd factorization.
  arma::vec sigma;
  factorizer.Apply(data, w, sigma, h);

  // Remove the part of w and h depending upon the value of rank.
  w = w.submat(0, 0, w.n_rows - 1, r - 1);
  h = h.submat(0, 0, h.n_cols - 1, r - 1);

  // Take only required eigenvalues.
  sigma = sigma.subvec(0, r - 1);

  // Eigenvalue matrix is multiplied to w.
  w = w * arma::diagmat(sigma);

  // Take transpose of the matrix h as required by CF module.
  h = arma::trans(h);
}

// Apply the factorizer when coordinate lists are not used.
template<typename FactorizerType>
void ApplyFactorizer(FactorizerType& factorizer,
                     const arma::mat& /* data */,
                     const arma::sp_mat& cleanedData,
                     const size_t rank,
                     arma::mat& w,
                     arma::mat& h,
                     const typename std::enable_if_t<!FactorizerTraits<
                         FactorizerType>::UsesCoordinateList>* = 0)
{
  factorizer.Apply(cleanedData, rank, w, h);
}

/**
 * Construct the CF object using an instantiated factorizer.
 */
template<typename FactorizerType>
CF::CF(const arma::mat& data,
       FactorizerType factorizer,
       const size_t numUsersForSimilarity,
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

  Train(data, factorizer);
}

/**
 * Construct the CF object using an instantiated factorizer.
 */
template<typename FactorizerType>
CF::CF(const arma::sp_mat& data,
       FactorizerType factorizer,
       const size_t numUsersForSimilarity,
       const size_t rank,
       const typename std::enable_if_t<
           !FactorizerTraits<FactorizerType>::UsesCoordinateList>*) :
    numUsersForSimilarity(numUsersForSimilarity),
    rank(rank)
{
  // Validate neighbourhood size.
  if (numUsersForSimilarity < 1)
  {
    Log::Warn << "CF::CF(): neighbourhood size should be > 0("
        << numUsersForSimilarity << " given). Setting value to 5.\n";
    // Setting Default Value of 5
    this->numUsersForSimilarity = 5;
  }

  Train(data, factorizer);
}

template<typename FactorizerType>
void CF::Train(const arma::mat& data, FactorizerType factorizer)
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
  ApplyFactorizer(factorizer, data, cleanedData, this->rank, w, h);
  Timer::Stop("cf_factorization");
}

template<typename FactorizerType>
void CF::Train(const arma::sp_mat& data,
               FactorizerType factorizer,
               const typename std::enable_if_t<!FactorizerTraits<
                   FactorizerType>::UsesCoordinateList>*)
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

  Timer::Start("cf_factorization");
  factorizer.Apply(cleanedData, this->rank, w, h);
  Timer::Stop("cf_factorization");
}

//! Serialize the model.
template<typename Archive>
void CF::serialize(Archive& ar, const unsigned int /* version */)
{
  // This model is simple; just serialize all the members.  No special handling
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
