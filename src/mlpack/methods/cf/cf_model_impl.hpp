/**
 * @file methods/cf/cf_model_impl.hpp
 * @author Wenhao Huang
 *
 * A serializable CF model, used by the main program.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_CF_MODEL_IMPL_HPP
#define MLPACK_METHODS_CF_CF_MODEL_IMPL_HPP

#include "cf_model.hpp"

#include <mlpack/methods/cf/normalization/no_normalization.hpp>
#include <mlpack/methods/cf/normalization/overall_mean_normalization.hpp>
#include <mlpack/methods/cf/normalization/user_mean_normalization.hpp>
#include <mlpack/methods/cf/normalization/item_mean_normalization.hpp>
#include <mlpack/methods/cf/normalization/z_score_normalization.hpp>

using namespace mlpack::cf;

template <typename DecompositionPolicy,
          typename NormalizationType>
void DeleteVisitor::
operator()(CFType<DecompositionPolicy, NormalizationType>* c) const
{
  if (c)
    delete c;
}

template <typename DecompositionPolicy,
          typename NormalizationType>
void* GetValueVisitor::
operator()(CFType<DecompositionPolicy, NormalizationType>* c) const
{
  if (!c)
    throw std::runtime_error("no cf model initialized");

  return (void*) c;
}

template <typename NeighborSearchPolicy,
          typename InterpolationPolicy>
PredictVisitor<NeighborSearchPolicy, InterpolationPolicy>::PredictVisitor(
    const arma::Mat<size_t>& combinations,
    arma::vec& predictions) :
    combinations(combinations),
    predictions(predictions)
{ }

template <typename NeighborSearchPolicy,
          typename InterpolationPolicy>
template <typename DecompositionPolicy,
          typename NormalizationType>
void PredictVisitor<NeighborSearchPolicy, InterpolationPolicy>
        ::operator()(CFType<DecompositionPolicy, NormalizationType>* c) const
{
  if (!c)
  {
    throw std::runtime_error("no cf model initialized");
    return;
  }

  c->template Predict<NeighborSearchPolicy,
      InterpolationPolicy>(combinations, predictions);
}

template <typename NeighborSearchPolicy,
          typename InterpolationPolicy>
RecommendationVisitor<NeighborSearchPolicy, InterpolationPolicy>
        ::RecommendationVisitor(
    const size_t numRecs,
    arma::Mat<size_t>& recommendations,
    const arma::Col<size_t>& users,
    const bool usersGiven) :
    numRecs(numRecs),
    recommendations(recommendations),
    users(users),
    usersGiven(usersGiven)
{ }

template <typename NeighborSearchPolicy,
          typename InterpolationPolicy>
template <typename DecompositionPolicy,
          typename NormalizationType>
void RecommendationVisitor<NeighborSearchPolicy, InterpolationPolicy>
        ::operator()(CFType<DecompositionPolicy, NormalizationType>* c) const
{
  if (!c)
  {
    throw std::runtime_error("no cf model initialized");
    return;
  }

  if (usersGiven)
    c->template GetRecommendations<NeighborSearchPolicy, InterpolationPolicy>
        (numRecs, recommendations, users);
  else
    c->template GetRecommendations<NeighborSearchPolicy, InterpolationPolicy>
        (numRecs, recommendations);
}

CFModel::~CFModel()
{
  boost::apply_visitor(DeleteVisitor(), cf);
}

template<typename DecompositionPolicy,
         typename MatType>
void CFModel::Train(const MatType& data,
                    const size_t numUsersForSimilarity,
                    const size_t rank,
                    const size_t maxIterations,
                    const double minResidue,
                    const bool mit,
                    const std::string& normalization)
{
  // Delete the current CFType object, if there is one.
  boost::apply_visitor(DeleteVisitor(), cf);

  // Instantiate a new CFType object.
  DecompositionPolicy decomposition;
  if (normalization == "overall_mean")
  {
    cf = new CFType<DecompositionPolicy, OverallMeanNormalization>(data,
        decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
        mit);
  }
  else if (normalization == "item_mean")
  {
    cf = new CFType<DecompositionPolicy, ItemMeanNormalization>(data,
        decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
        mit);
  }
  else if (normalization == "user_mean")
  {
    cf = new CFType<DecompositionPolicy, UserMeanNormalization>(data,
        decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
        mit);
  }
  else if (normalization == "z_score")
  {
    cf = new CFType<DecompositionPolicy, ZScoreNormalization>(data,
        decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
        mit);
  }
  else if (normalization == "none")
  {
    cf = new CFType<DecompositionPolicy, NoNormalization>(data,
        decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
        mit);
  }
  else
  {
    throw std::runtime_error("Unsupported normalization algorithm."
                             " It should be one of none, overall_mean, "
                             "item_mean, user_mean or z_score");
  }
}

//! Make predictions.
template <typename NeighborSearchPolicy,
          typename InterpolationPolicy>
void CFModel::Predict(const arma::Mat<size_t>& combinations,
                      arma::vec& predictions)
{
  PredictVisitor<NeighborSearchPolicy, InterpolationPolicy>
      predict(combinations, predictions);
  boost::apply_visitor(predict, cf);
}

//! Compute recommendations for queried users.
template<typename NeighborSearchPolicy,
         typename InterpolationPolicy>
void CFModel::GetRecommendations(const size_t numRecs,
                                 arma::Mat<size_t>& recommendations,
                                 const arma::Col<size_t>& users)
{
  RecommendationVisitor<NeighborSearchPolicy, InterpolationPolicy>
      recommendation(numRecs, recommendations, users, true);
  boost::apply_visitor(recommendation, cf);
}

//! Compute recommendations for all users.
template<typename NeighborSearchPolicy,
         typename InterpolationPolicy>
void CFModel::GetRecommendations(const size_t numRecs,
                                 arma::Mat<size_t>& recommendations)
{
  arma::Col<size_t> users;
  RecommendationVisitor<NeighborSearchPolicy, InterpolationPolicy>
      recommendation(numRecs, recommendations, users, false);
  boost::apply_visitor(recommendation, cf);
}

template <typename DecompositionPolicy,
          typename NormalizationType>
const CFType<DecompositionPolicy, NormalizationType>* CFModel::CFPtr() const
{
  void* pointer = boost::apply_visitor(GetValueVisitor(), cf);
  return (CFType<DecompositionPolicy, NormalizationType>*) pointer;
}

template<typename Archive>
void CFModel::serialize(Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  // This should never happen, but just in case, be clean with memory.
  if (Archive::is_loading::value)
    boost::apply_visitor(DeleteVisitor(), cf);

  ar & CEREAL_VARIANT_POINTER(cf);
}

#endif
