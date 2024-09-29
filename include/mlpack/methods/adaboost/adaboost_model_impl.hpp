/**
 * @file methods/adaboost/adaboost_model_impl.hpp
 * @author Ryan Curtin
 *
 * A serializable AdaBoost model, used by the main program.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ADABOOST_ADABOOST_MODEL_IMPL_HPP
#define MLPACK_METHODS_ADABOOST_ADABOOST_MODEL_IMPL_HPP

#include "adaboost.hpp"
#include "adaboost_model.hpp"

namespace mlpack {

//! Create an empty AdaBoost model.
inline AdaBoostModel::AdaBoostModel() :
    weakLearnerType(0),
    dsBoost(NULL),
    pBoost(NULL),
    dimensionality(0)
{
  // Nothing to do.
}

//! Create the AdaBoost model with the given mappings and type.
inline AdaBoostModel::AdaBoostModel(
    const arma::Col<size_t>& mappings,
    const size_t weakLearnerType) :
    mappings(mappings),
    weakLearnerType(weakLearnerType),
    dsBoost(NULL),
    pBoost(NULL),
    dimensionality(0)
{
  // Nothing to do.
}

//! Copy constructor.
inline AdaBoostModel::AdaBoostModel(const AdaBoostModel& other) :
    mappings(other.mappings),
    weakLearnerType(other.weakLearnerType),
    dsBoost(other.dsBoost == NULL ? NULL :
        new AdaBoost<ID3DecisionStump>(*other.dsBoost)),
    pBoost(other.pBoost == NULL ? NULL :
        new AdaBoost<Perceptron<>>(*other.pBoost)),
    dimensionality(other.dimensionality)
{
  // Nothing to do.
}

//! Move constructor.
inline AdaBoostModel::AdaBoostModel(AdaBoostModel&& other) :
    mappings(std::move(other.mappings)),
    weakLearnerType(other.weakLearnerType),
    dsBoost(other.dsBoost),
    pBoost(other.pBoost),
    dimensionality(other.dimensionality)
{
  other.weakLearnerType = 0;
  other.dsBoost = NULL;
  other.pBoost = NULL;
  other.dimensionality = 0;
}

//! Copy assignment operator.
inline AdaBoostModel& AdaBoostModel::operator=(const AdaBoostModel& other)
{
  if (this != &other)
  {
    mappings = other.mappings;
    weakLearnerType = other.weakLearnerType;

    delete dsBoost;
    dsBoost = (other.dsBoost == NULL) ? NULL :
        new AdaBoost<ID3DecisionStump>(*other.dsBoost);

    delete pBoost;
    pBoost = (other.pBoost == NULL) ? NULL :
        new AdaBoost<Perceptron<>>(*other.pBoost);

    dimensionality = other.dimensionality;
  }
  return *this;
}

//! Move assignment operator.
inline AdaBoostModel& AdaBoostModel::operator=(AdaBoostModel&& other)
{
  if (this != &other)
  {
    mappings = std::move(other.mappings);
    weakLearnerType = other.weakLearnerType;

    dsBoost = other.dsBoost;
    other.dsBoost = nullptr;

    pBoost = other.pBoost;
    other.pBoost = nullptr;

    dimensionality = other.dimensionality;
  }
  return *this;
}

inline AdaBoostModel::~AdaBoostModel()
{
  delete dsBoost;
  delete pBoost;
}

//! Train the model.
inline void AdaBoostModel::Train(const arma::mat& data,
                                 const arma::Row<size_t>& labels,
                                 const size_t numClasses,
                                 const size_t iterations,
                                 const double tolerance)
{
  dimensionality = data.n_rows;
  if (weakLearnerType == WeakLearnerTypes::DECISION_STUMP)
  {
    delete dsBoost;
    dsBoost = new AdaBoost<ID3DecisionStump>(data, labels, numClasses,
        iterations, tolerance);
  }
  else if (weakLearnerType == WeakLearnerTypes::PERCEPTRON)
  {
    delete pBoost;
    pBoost = new AdaBoost<Perceptron<>>(data, labels, numClasses, iterations,
        tolerance);
  }
}

//! Classify test points.
inline void AdaBoostModel::Classify(const arma::mat& testData,
                                    arma::Row<size_t>& predictions,
                                    arma::mat& probabilities)
{
  if (weakLearnerType == WeakLearnerTypes::DECISION_STUMP)
    dsBoost->Classify(testData, predictions, probabilities);
  else if (weakLearnerType == WeakLearnerTypes::PERCEPTRON)
    pBoost->Classify(testData, predictions, probabilities);
}

//! Classify test points.
inline void AdaBoostModel::Classify(const arma::mat& testData,
                                    arma::Row<size_t>& predictions)
{
  if (weakLearnerType == WeakLearnerTypes::DECISION_STUMP)
    dsBoost->Classify(testData, predictions);
  else if (weakLearnerType == WeakLearnerTypes::PERCEPTRON)
    pBoost->Classify(testData, predictions);
}

} // namespace mlpack

#endif
