/**
 * @file methods/grad_boosting/grad_boosting_model_impl.hpp
 * @author Abhimanyu Dayal
 *
 * A serializable Gradient Boosting model, used by the main program.
 * Implementation of GradBoostingModel class. 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Include guard - Used to prevent including double copies of the same code
#ifndef MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_MODEL_IMPL_HPP
#define MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_MODEL_IMPL_HPP

// Uses the grad_boosting class algorithms.
#include "grad_boosting.hpp"

// Base definition of the GradBoostingModel class.
#include "grad_boosting_model.hpp"

// Defined inside the mlpack namespace.
namespace mlpack {

//! Create an empty GradBoosting model.
// GradBoostingModel constructor. 
inline GradBoostingModel::GradBoostingModel() :
  // weakLearnerType initialised to 0, i.e. Decision Stump
  weakLearnerType(0),
  // gbModel value set to NULL
  gbModel(NULL),
  // Set dimensionality to 0
  dimensionality(0)
{
  // Nothing to do.
}

//! Create the GradBoosting model with the given mappings and type.
// For now, this constructor is unused.
inline GradBoostingModel::GradBoostingModel(const arma::Col<size_t>& mappings,
                                            const size_t weakLearnerType) :
  mappings(mappings),
  weakLearnerType(weakLearnerType),
  gbModel(NULL),
  dimensionality(0)
{
  // Nothing to do.
}

//! Copy constructor.
inline GradBoostingModel::GradBoostingModel(const GradBoostingModel& other) :
  mappings(other.mappings),
  weakLearnerType(other.weakLearnerType),
  gbModel(other.gbModel == nullptr ? nullptr :
    // Defaulted to ID3 Decision Stump with arma::mat MatType
    new GradBoosting(other.gbModel)),
  dimensionality(other.dimensionality)
{
  // Nothing to do.
}

//! Move constructor.
inline GradBoostingModel::GradBoostingModel(GradBoostingModel&& other) :
  mappings(std::move(other.mappings)),
  weakLearnerType(other.weakLearnerType),
  gbModel(other.gbModel),
  dimensionality(other.dimensionality)
{
  other.weakLearnerType = 0;
  other.gbModel = NULL;
  other.dimensionality = 0;
}

//! Copy assignment operator.
inline GradBoostingModel& GradBoostingModel::operator=(const GradBoostingModel& other)
{
  if (this != &other)
  {
    mappings = other.mappings;
    weakLearnerType = other.weakLearnerType;

    delete gbModel;
    gbModel = (other.gbModel == NULL) ? NULL :
      // Defaulted to ID3 Decision Stump with arma::mat MatType
      new GradBoosting(other.gbModel);

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

    gbModel = other.gbModel;
    other.gbModel = nullptr;

    dimensionality = other.dimensionality;
  }
  return *this;
}

//! Deconstructor
inline GradBoostingModel::~GradBoostingModel()
{
  delete gbModel;
}

//! Train the model.
inline void GradBoostingModel::Train(
  const arma::mat& data,
  const arma::Row<size_t>& labels,
  const size_t numClasses,
  const size_t numModels)
{
  dimensionality = data.n_rows;
  delete gbModel;
  
  // Defaulted to ID3 Decision Stump with arma::mat MatType
  gbModel = new GradBoosting(data, labels, numClasses,
    numModels);
}

//! Classify test points. Calculate the probabilities.
inline void GradBoostingModel::Classify(const arma::mat& testData,
                                        arma::Row<size_t>& predictions,
                                        arma::mat& probabilities)
{
  gbModel->Classify(testData, predictions, probabilities);
}

//! Classify test points. Not including probabilities.
inline void GradBoostingModel::Classify (const arma::mat& testData,
                                          arma::Row<size_t>& predictions)
{
  gbModel->Classify(testData, predictions);
}

} // namespace mlpack

#endif