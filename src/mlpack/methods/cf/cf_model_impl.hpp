/**
 * @file cf_model_impl.cpp
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

#include <boost/serialization/variant.hpp>

using namespace mlpack::cf;

template<typename DecompositionPolicy>
void DeleteVisitor::operator()(CFType<DecompositionPolicy>* c) const
{
  if (c)
    delete c;
}

template<typename DecompositionPolicy>
void* GetValueVisitor::operator()(CFType<DecompositionPolicy>* c) const
{
  if (!c)
    throw std::runtime_error("no cf model initialized");

  return (void*) c;
}

PredictVisitor::PredictVisitor(
    const arma::Mat<size_t>& combinations,
    arma::vec& predictions) :
    combinations(combinations),
    predictions(predictions)
{ }

template<typename DecompositionPolicy>
void PredictVisitor::operator()(CFType<DecompositionPolicy>* c) const
{
  if (!c)
  {
    throw std::runtime_error("no cf model initialized");
    return;
  }

  c->Predict(combinations, predictions);
}

RecommendationVisitor::RecommendationVisitor(
    const size_t numRecs,
    arma::Mat<size_t>& recommendations,
    const arma::Col<size_t>& users,
    const bool usersGiven) :
    numRecs(numRecs),
    recommendations(recommendations),
    users(users),
    usersGiven(usersGiven)
{ }

template<typename DecompositionPolicy>
void RecommendationVisitor::operator()(CFType<DecompositionPolicy>* c) const
{
  if (!c)
  {
    throw std::runtime_error("no cf model initialized");
    return;
  }

  if (usersGiven)
    c->GetRecommendations(numRecs, recommendations, users);
  else
    c->GetRecommendations(numRecs, recommendations);
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
                    const bool mit)
{
  // Delete the current CFType object, if there is one.
  boost::apply_visitor(DeleteVisitor(), cf);

  // Instantiate a new CFType object.
  DecompositionPolicy decomposition;
  cf = new CFType<DecompositionPolicy>(data, decomposition,
      numUsersForSimilarity, rank, maxIterations, minResidue, mit);
}

//! Make predictions.
void CFModel::Predict(const arma::Mat<size_t>& combinations,
                      arma::vec& predictions)
{
  PredictVisitor predict(combinations, predictions);
  boost::apply_visitor(predict, cf);
}

//! Compute recommendations for queried users.
void CFModel::GetRecommendations(const size_t numRecs,
                                 arma::Mat<size_t>& recommendations,
                                 const arma::Col<size_t>& users)
{
  RecommendationVisitor recommendation(numRecs, recommendations, users, true);
  boost::apply_visitor(recommendation, cf);
}

//! Compute recommendations for all users.
void CFModel::GetRecommendations(const size_t numRecs,
                                 arma::Mat<size_t>& recommendations)
{
  arma::Col<size_t> users;
  RecommendationVisitor recommendation(numRecs, recommendations, users, false);
  boost::apply_visitor(recommendation, cf);
}

template<typename DecompositionPolicy>
const CFType<DecompositionPolicy>* CFModel::CFPtr() const
{
  void* pointer = boost::apply_visitor(GetValueVisitor(), cf);
  return (CFType<DecompositionPolicy>*) pointer;
}

template<typename Archive>
void CFModel::serialize(Archive& ar, const unsigned int /* version */)
{
  // This should never happen, but just in case, be clean with memory.
  if (Archive::is_loading::value)
    boost::apply_visitor(DeleteVisitor(), cf);

  ar & BOOST_SERIALIZATION_NVP(cf);
}

#endif
