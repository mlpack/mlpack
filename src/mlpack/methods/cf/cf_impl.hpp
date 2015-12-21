/**
 * @file cf_impl.hpp
 * @author Mudit Raj Gupta
 * @author Sumedh Ghaisas
 *
 * Collaborative Filtering.
 *
 * Implementation of CF class to perform Collaborative Filtering on the
 * specified data set.
 */
#ifndef __MLPACK_METHODS_CF_CF_IMPL_HPP
#define __MLPACK_METHODS_CF_CF_IMPL_HPP

// In case it hasn't been included yet.
#include "cf.hpp"

namespace mlpack {
namespace cf {

// Apply the factorizer when a coordinate list is used.
template<typename FactorizerType>
void ApplyFactorizer(FactorizerType& factorizer,
                     const arma::mat& data,
                     const arma::sp_mat& /* cleanedData */,
                     const size_t rank,
                     arma::mat& w,
                     arma::mat& h,
                     const typename boost::enable_if_c<FactorizerTraits<
                         FactorizerType>::UsesCoordinateList>::type* = 0)
{
  factorizer.Apply(data, rank, w, h);
}

// Apply the factorizer when coordinate lists are not used.
template<typename FactorizerType>
void ApplyFactorizer(FactorizerType& factorizer,
                     const arma::mat& /* data */,
                     const arma::sp_mat& cleanedData,
                     const size_t rank,
                     arma::mat& w,
                     arma::mat& h,
                     const typename boost::disable_if_c<FactorizerTraits<
                         FactorizerType>::UsesCoordinateList>::type* = 0)
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
       const typename boost::disable_if_c<FactorizerTraits<
           FactorizerType>::UsesCoordinateList>::type*) :
    numUsersForSimilarity(numUsersForSimilarity),
    rank(rank)
{
  // Validate neighbourhood size.
  if (numUsersForSimilarity < 1)
  {
    Log::Warn << "CF::CF(): neighbourhood size should be > 0("
        << numUsersForSimilarity << " given). Setting value to 5.\n";
    //Setting Default Value of 5
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
               const typename boost::disable_if_c<FactorizerTraits<
                   FactorizerType>::UsesCoordinateList>::type*)
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

} // namespace mlpack
} // namespace cf

#endif
