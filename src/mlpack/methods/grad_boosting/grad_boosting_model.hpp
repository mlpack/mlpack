/**
 * @file methods/grad_boosting/grad_boosting_model.hpp
 * @author Abhimanyu Dayal
 *
 * A serializable Gradient Boosting model, used by the Gradient Boosting binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Include guard - Used to prevent including double copies of the same code
#ifndef MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_MODEL_HPP
#define MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_MODEL_HPP

// Importing base components required to write mlpack methods.
#include <mlpack/core.hpp>

// Use forward declaration instead of include to accelerate compilation.
class GradBoosting; 

// Defining the GradBoostingModel class within the mlpack namespace.
namespace mlpack {

/**
 * The model to save to disk.
 */
class GradBoostingModel {
 public:

  // List of weak learners that can be used in this. 
  // Only using Decision trees for now, but may be extended to other models too.
  enum WeakLearnerTypes 
  {
    DECISION_STUMP
  };

 private:

  //! The mappings for the labels.
  arma::Col<size_t> mappings;

  //! The type of weak learner.
  size_t weakLearnerType;

  //! Non-NULL if using decision stumps.
  // For now this is the only one we have as we're only using Decision stumps.
  // GradBoosting class contains the algorithms for the model implementation.
  GradBoosting<ID3DecisionStump*> dsBoost;

  //! Number of dimensions in training data.
  size_t dimensionality;

 public:

  // ### CONSTRUCTORS

  //! Create an empty GradBoosting model.
  GradBoostingModel();

  //! Create the GradBoosting model with the given mappings and type.
  // For now, we're not using this constructor anywhere.
  GradBoostingModel(const arma::Col<size_t>& mappings,
                    const size_t weakLearnerType);



  // ### OPERATION CONSTRUCTORS
  // All these operations currently unused.

  //! Copy constructor.
  GradBoostingModel(const GradBoostingModel& other);

  //! Move constructor.
  GradBoostingModel(GradBoostingModel&& other);

  //! Copy assignment operator.
  GradBoostingModel& operator=(const GradBoostingModel& other);

  //! Move assignment operator.
  GradBoostingModel& operator=(GradBoostingModel&& other);

  //! Clean up memory.
  ~GradBoostingModel();

  //! Get the mappings.
  const arma::Col<size_t>& Mappings() const { return mappings; }

  //! Modify the mappings.
  arma::Col<size_t>& Mappings() { return mappings; }

  //! Get the weak learner type.
  size_t WeakLearnerType() const { return weakLearnerType; }
  //! Modify the weak learner type.
  size_t& WeakLearnerType() { return weakLearnerType; }

  //! Get the dimensionality of the model.
  size_t Dimensionality() const { return dimensionality; }
  //! Modify the dimensionality of the model.
  size_t& Dimensionality() { return dimensionality; }

  // ### TRAINING

  //! Train the model, treat the data is all of the numeric type.
  void Train(const arma::mat& data,
              const arma::Row<size_t>& labels,
              const size_t numClasses,
              const size_t numModels);

  // ### CLASSIFY

  //! Classify test points. With probability.
  void Classify(const arma::mat& testData,
                arma::Row<size_t>& predictions,
                arma::mat& probabilities);

  //! Classify test points. Without probability.
  void Classify(
    const arma::mat& testData,
    arma::Row<size_t>& predictions
  );

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const uint32_t version) 
  {
    if (cereal::is_loading<Archive>()) 
    {
      dsBoost = NULL;
      delete dsBoost;
    }

    ar(CEREAL_NVP(mappings));
    ar(CEREAL_NVP(weakLearnerType));
    ar(CEREAL_POINTER(dsBoost));
    ar(CEREAL_NVP(dimensionality));
  }
};

}

// Include implementation.
#include <grad_boosting_model_impl.hpp>

#endif