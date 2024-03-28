/**
 * @file methods/grad_boosting/grad_boosting_model_impl.hpp
 * @author Abhimanyu Dayal
 *
 * A serializable Gradient Boosting model, used by the main program.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_MODEL_IMPL_HPP
#define MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_MODEL_IMPL_HPP

#include "grad_boosting.hpp"
#include "grad_boosting_model.hpp"

namespace mlpack {

//! Create an empty GradBoosting model.
inline GradBoostingModel::GradBoostingModel() :
    weakLearnerType(0),
    dsBoost(NULL),
    dimensionality(0)
{
  // Nothing to do.
}

//! Create the GradBoosting model with the given mappings and type.
inline GradBoostingModel::GradBoostingModel(
    const arma::Col<size_t>& mappings,
    const size_t weakLearnerType) :
    mappings(mappings),
    weakLearnerType(weakLearnerType),
    dsBoost(NULL),
    dimensionality(0)
{
  // Nothing to do.
}

//! Copy constructor.
inline GradBoostingModel::GradBoostingModel(const GradBoostingModel& other) :
    mappings(other.mappings),
    weakLearnerType(other.weakLearnerType),
    dsBoost(other.dsBoost == nullptr ? nullptr :
        new GradBoosting<ID3DecisionStump>(*other.dsBoost)),
    dimensionality(other.dimensionality)
{
  // Nothing to do.
}

//! Move constructor.
inline GradBoostingModel::GradBoostingModel(GradBoostingModel&& other) :
    mappings(std::move(other.mappings)),
    weakLearnerType(other.weakLearnerType),
    dsBoost(other.dsBoost),
    dimensionality(other.dimensionality)
{
  other.weakLearnerType = 0;
  other.dsBoost = NULL;
  other.dimensionality = 0;
}

//! Copy assignment operator.
inline GradBoostingModel& GradBoostingModel::operator=(const GradBoostingModel& other)
{
  if (this != &other)
  {
    mappings = other.mappings;
    weakLearnerType = other.weakLearnerType;

    delete dsBoost;
    dsBoost = (other.dsBoost == NULL) ? NULL :
        new GradBoosting<ID3DecisionStump>(*other.dsBoost);

    dimensionality = other.dimensionality;
  }
  return *this;
}

//! Move assignment operator.
inline GradBoostingModel& GradBoostingModel::operator=(GradBoostingModel&& other)
{
  if (this != &other)
  {
    mappings = std::move(other.mappings);
    weakLearnerType = other.weakLearnerType;

    dsBoost = other.dsBoost;
    other.dsBoost = nullptr;

    dimensionality = other.dimensionality;
  }
  return *this;
}

inline GradBoostingModel::~GradBoostingModel()
{
  delete dsBoost;
  delete pBoost;
}

//! Train the model.
inline void GradBoostingModel::Train(const arma::mat& data,
                                 const arma::Row<size_t>& labels,
                                 const size_t numClasses,
                                 const size_t num_models)
{
    dimensionality = data.n_rows;
    delete dsBoost;
    dsBoost = new GradBoosting<ID3DecisionStump>(data, labels, numClasses,
        num_models);
}

//! Classify test points.
inline void GradBoostingModel::Classify(const arma::mat& testData,
                                    arma::Row<size_t>& predictions,
                                    arma::mat& probabilities)
{
    dsBoost->Classify(testData, predictions, probabilities);
}

//! Classify test points.
inline void GradBoostingModel::Classify(const arma::mat& testData,
                                    arma::Row<size_t>& predictions)
{
    dsBoost->Classify(testData, predictions);
}

} // namespace mlpack

#endif
