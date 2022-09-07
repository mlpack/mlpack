/**
 * @file methods/adaboost/adaboost_model.hpp
 * @author Ryan Curtin
 *
 * A serializable AdaBoost model, used by the AdaBoost binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ADABOOST_ADABOOST_MODEL_HPP
#define MLPACK_METHODS_ADABOOST_ADABOOST_MODEL_HPP

#include <mlpack/core.hpp>

// Use forward declaration instead of include to accelerate compilation.
class AdaBoost;

namespace mlpack {

/**
 * The model to save to disk.
 */
class AdaBoostModel
{
 public:
  enum WeakLearnerTypes
  {
    DECISION_STUMP,
    PERCEPTRON
  };

 private:
  //! The mappings for the labels.
  arma::Col<size_t> mappings;
  //! The type of weak learner.
  size_t weakLearnerType;
  //! Non-NULL if using decision stumps.
  AdaBoost<ID3DecisionStump>* dsBoost;
  //! Non-NULL if using perceptrons.
  AdaBoost<Perceptron<>>* pBoost;
  //! Number of dimensions in training data.
  size_t dimensionality;

 public:
  //! Create an empty AdaBoost model.
  AdaBoostModel();

  //! Create the AdaBoost model with the given mappings and type.
  AdaBoostModel(const arma::Col<size_t>& mappings,
                const size_t weakLearnerType);

  //! Copy constructor.
  AdaBoostModel(const AdaBoostModel& other);

  //! Move constructor.
  AdaBoostModel(AdaBoostModel&& other);

  //! Copy assignment operator.
  AdaBoostModel& operator=(const AdaBoostModel& other);

  //! Move assignment operator.
  AdaBoostModel& operator=(AdaBoostModel&& other);

  //! Clean up memory.
  ~AdaBoostModel();

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

  //! Train the model, treat the data is all of the numeric type.
  void Train(const arma::mat& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const size_t iterations,
             const double tolerance);

  //! Classify test points.
  void Classify(const arma::mat& testData,
                arma::Row<size_t>& predictions);

  //! Classify test points.
  void Classify(const arma::mat& testData,
                arma::Row<size_t>& predictions,
                arma::mat& probabilities);

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    if (cereal::is_loading<Archive>())
    {
      if (dsBoost)
        delete dsBoost;
      if (pBoost)
        delete pBoost;

      dsBoost = NULL;
      pBoost = NULL;
    }

    ar(CEREAL_NVP(mappings));
    ar(CEREAL_NVP(weakLearnerType));
    if (weakLearnerType == WeakLearnerTypes::DECISION_STUMP)
      ar(CEREAL_POINTER(dsBoost));
    else if (weakLearnerType == WeakLearnerTypes::PERCEPTRON)
      ar(CEREAL_POINTER(pBoost));
    ar(CEREAL_NVP(dimensionality));
  }
};

} // namespace mlpack

// Include implementation.
#include "adaboost_model_impl.hpp"

#endif
