/**
 * @file methods/hmm/hmm_model.hpp
 * @author Ryan Curtin
 *
 * A serializable HMM model that also stores the type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_HMM_HMM_MODEL_HPP
#define MLPACK_METHODS_HMM_HMM_MODEL_HPP

#include "hmm.hpp"
#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/methods/gmm/diagonal_gmm.hpp>
#include <mlpack/core/util/params.hpp>

namespace mlpack {

enum HMMType : char
{
  DiscreteHMM = 0,
  GaussianHMM,
  GaussianMixtureModelHMM,
  DiagonalGaussianMixtureModelHMM
};

/**
 * A serializable HMM model that also stores the type.
 */
class HMMModel
{
 private:
  //! The type of the HMM.
  HMMType type;
  //! Not used if type is not DiscreteHMM.
  HMM<DiscreteDistribution<>>* discreteHMM;
  //! Not used if type is not GaussianHMM.
  HMM<GaussianDistribution<>>* gaussianHMM;
  //! Not used if type is not GaussianMixtureModelHMM.
  HMM<GMM>* gmmHMM;
  //! Not used if type is not DiagonalGaussianMixtureModelHMM.
  HMM<DiagonalGMM>* diagGMMHMM;

 public:
  //! Construct a model of the given type.
  HMMModel(const HMMType type = HMMType::DiscreteHMM) :
      type(type),
      discreteHMM(NULL),
      gaussianHMM(NULL),
      gmmHMM(NULL),
      diagGMMHMM(NULL)
  {
    if (type == HMMType::DiscreteHMM)
      discreteHMM = new HMM<DiscreteDistribution<>>();
    else if (type == HMMType::GaussianHMM)
      gaussianHMM = new HMM<GaussianDistribution<>>();
    else if (type == HMMType::GaussianMixtureModelHMM)
      gmmHMM = new HMM<GMM>();
    else if (type == HMMType::DiagonalGaussianMixtureModelHMM)
      diagGMMHMM = new HMM<DiagonalGMM>();
  }

  //! Copy another model.
  HMMModel(const HMMModel& other) :
      type(other.type),
      discreteHMM(NULL),
      gaussianHMM(NULL),
      gmmHMM(NULL),
      diagGMMHMM(NULL)
  {
    if (type == HMMType::DiscreteHMM)
      discreteHMM =
          new HMM<DiscreteDistribution<>>(*other.discreteHMM);
    else if (type == HMMType::GaussianHMM)
      gaussianHMM =
          new HMM<GaussianDistribution<>>(*other.gaussianHMM);
    else if (type == HMMType::GaussianMixtureModelHMM)
      gmmHMM = new HMM<GMM>(*other.gmmHMM);
    else if (type == HMMType::DiagonalGaussianMixtureModelHMM)
      diagGMMHMM = new HMM<DiagonalGMM>(*other.diagGMMHMM);
  }

  //! Take ownership of another model.
  HMMModel(HMMModel&& other) :
      type(other.type),
      discreteHMM(other.discreteHMM),
      gaussianHMM(other.gaussianHMM),
      gmmHMM(other.gmmHMM),
      diagGMMHMM(other.diagGMMHMM)
  {
    other.type = HMMType::DiscreteHMM;
    other.discreteHMM = new HMM<DiscreteDistribution<>>();
    other.gaussianHMM = NULL;
    other.gmmHMM = NULL;
    other.diagGMMHMM = NULL;
  }

  //! Copy assignment operator.
  HMMModel& operator=(const HMMModel& other)
  {
    if (this == &other)
      return *this;

    delete discreteHMM;
    delete gaussianHMM;
    delete gmmHMM;
    delete diagGMMHMM;

    discreteHMM = NULL;
    gaussianHMM = NULL;
    gmmHMM = NULL;
    diagGMMHMM = NULL;

    type = other.type;
    if (type == HMMType::DiscreteHMM)
      discreteHMM =
          new HMM<DiscreteDistribution<>>(*other.discreteHMM);
    else if (type == HMMType::GaussianHMM)
      gaussianHMM =
          new HMM<GaussianDistribution<>>(*other.gaussianHMM);
    else if (type == HMMType::GaussianMixtureModelHMM)
      gmmHMM = new HMM<GMM>(*other.gmmHMM);
    else if (type == HMMType::DiagonalGaussianMixtureModelHMM)
      diagGMMHMM = new HMM<DiagonalGMM>(*other.diagGMMHMM);

    return *this;
  }

  //! Move assignment operator.
  HMMModel& operator=(HMMModel&& other)
  {
    if (this != &other)
    {
      type = other.type;
      discreteHMM = other.discreteHMM;
      gaussianHMM = other.gaussianHMM;
      gmmHMM = other.gmmHMM;
      diagGMMHMM = other.diagGMMHMM;

      other.type = HMMType::DiscreteHMM;
      other.discreteHMM = new HMM<DiscreteDistribution<>>();
      other.gaussianHMM = nullptr;
      other.gmmHMM = nullptr;
      other.diagGMMHMM = nullptr;
    }
    return *this;
  }

  //! Clean memory.
  ~HMMModel()
  {
    delete discreteHMM;
    delete gaussianHMM;
    delete gmmHMM;
    delete diagGMMHMM;
  }

  /**
   * Given a functor type, perform that functor with the optional extra info on
   * the HMM.
   */
  template<typename ActionType,
           typename ExtraInfoType>
  void PerformAction(util::Params& params, ExtraInfoType* x)
  {
    if (type == HMMType::DiscreteHMM)
      ActionType::Apply(params, *discreteHMM, x);
    else if (type == HMMType::GaussianHMM)
      ActionType::Apply(params, *gaussianHMM, x);
    else if (type == HMMType::GaussianMixtureModelHMM)
      ActionType::Apply(params, *gmmHMM, x);
    else if (type == HMMType::DiagonalGaussianMixtureModelHMM)
      ActionType::Apply(params, *diagGMMHMM, x);
  }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(type));

    // If necessary, clean memory.
    if (cereal::is_loading<Archive>())
    {
      delete discreteHMM;
      delete gaussianHMM;
      delete gmmHMM;
      delete diagGMMHMM;

      discreteHMM = NULL;
      gaussianHMM = NULL;
      gmmHMM = NULL;
      diagGMMHMM = NULL;
    }

    if (type == HMMType::DiscreteHMM)
      ar(CEREAL_POINTER(discreteHMM));
    else if (type == HMMType::GaussianHMM)
      ar(CEREAL_POINTER(gaussianHMM));
    else if (type == HMMType::GaussianMixtureModelHMM)
      ar(CEREAL_POINTER(gmmHMM));
    else if (type == HMMType::DiagonalGaussianMixtureModelHMM)
      ar(CEREAL_POINTER(diagGMMHMM));
  }

  // Accessor method for type of HMM
  HMMType Type() { return type; }

  /**
   * Accessor methods for discreteHMM, gaussianHMM, gmmHMM, and diagGMMHMM.
   * Note that an instatiation of this class will only contain one type of HMM
   * (as indicated by the "type" instance variable) - the other two pointers
   * will be NULL.
   *
   * For instance, if the HMMModel object holds a discrete HMM, then:
   * type         --> DiscreteHMM
   * gaussianHMM  --> NULL
   * gmmHMM       --> NULL
   * diagGMMHMM   --> NULL
   * discreteHMM  --> HMM<DiscreteDistribution<>> object
   * and hence, calls to GMMHMM(), DiagGMMHMM() and GaussianHMM() will return
   * NULL. Only the call to DiscreteHMM() will return a non NULL pointer.
   *
   * Hence, in practice, a user should be careful to first check the type of HMM
   * (by calling the Type() accessor) and then perform subsequent actions, to
   * avoid null pointer dereferences.
   */
  HMM<DiscreteDistribution<>>* DiscreteHMM() { return discreteHMM; }
  HMM<GaussianDistribution<>>* GaussianHMM() { return gaussianHMM; }
  HMM<GMM>* GMMHMM() { return gmmHMM; }
  HMM<DiagonalGMM>* DiagGMMHMM() { return diagGMMHMM; }
};

} // namespace mlpack

#endif
