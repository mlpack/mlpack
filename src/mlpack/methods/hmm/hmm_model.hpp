/**
 * @file hmm_model.hpp
 * @author Ryan Curtin
 *
 * A serializable HMM model that also stores the type.
 */
#ifndef MLPACK_METHODS_HMM_HMM_MODEL_HPP
#define MLPACK_METHODS_HMM_HMM_MODEL_HPP

#include "hmm.hpp"
#include <mlpack/methods/gmm/gmm.hpp>

namespace mlpack {
namespace hmm {

enum HMMType : char
{
  DiscreteHMM = 0,
  GaussianHMM,
  GaussianMixtureModelHMM
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
  HMM<distribution::DiscreteDistribution>* discreteHMM;
  //! Not used if type is not GaussianHMM.
  HMM<distribution::GaussianDistribution>* gaussianHMM;
  //! Not used if type is not GaussianMixtureModelHMM.
  HMM<gmm::GMM>* gmmHMM;

 public:
  //! Construct an uninitialized model.
  HMMModel() :
      type(HMMType::DiscreteHMM),
      discreteHMM(new HMM<distribution::DiscreteDistribution>()),
      gaussianHMM(NULL),
      gmmHMM(NULL)
  {
    // Nothing to do.
  }

  //! Construct a model of the given type.
  HMMModel(const HMMType type) :
      type(type),
      discreteHMM(NULL),
      gaussianHMM(NULL),
      gmmHMM(NULL)
  {
    if (type == HMMType::DiscreteHMM)
      discreteHMM = new HMM<distribution::DiscreteDistribution>();
    else if (type == HMMType::GaussianHMM)
      gaussianHMM = new HMM<distribution::GaussianDistribution>();
    else if (type == HMMType::GaussianMixtureModelHMM)
      gmmHMM = new HMM<gmm::GMM>();
  }

  //! Copy another model.
  HMMModel(const HMMModel& other) :
      type(other.type),
      discreteHMM(NULL),
      gaussianHMM(NULL),
      gmmHMM(NULL)
  {
    if (type == HMMType::DiscreteHMM)
      discreteHMM =
          new HMM<distribution::DiscreteDistribution>(*other.discreteHMM);
    else if (type == HMMType::GaussianHMM)
      gaussianHMM =
          new HMM<distribution::GaussianDistribution>(*other.gaussianHMM);
    else if (type == HMMType::GaussianMixtureModelHMM)
      gmmHMM = new HMM<gmm::GMM>(*other.gmmHMM);
  }

  //! Take ownership of another model.
  HMMModel(HMMModel&& other) :
      type(other.type),
      discreteHMM(other.discreteHMM),
      gaussianHMM(other.gaussianHMM),
      gmmHMM(other.gmmHMM)
  {
    other.type = HMMType::DiscreteHMM;
    other.discreteHMM = new HMM<distribution::DiscreteDistribution>();
    other.gaussianHMM = NULL;
    other.gmmHMM = NULL;
  }

  //! Copy assignment operator.
  HMMModel& operator=(const HMMModel& other)
  {
    delete discreteHMM;
    delete gaussianHMM;
    delete gmmHMM;

    discreteHMM = NULL;
    gaussianHMM = NULL;
    gmmHMM = NULL;

    type = other.type;
    if (type == HMMType::DiscreteHMM)
      discreteHMM =
          new HMM<distribution::DiscreteDistribution>(*other.discreteHMM);
    else if (type == HMMType::GaussianHMM)
      gaussianHMM =
          new HMM<distribution::GaussianDistribution>(*other.gaussianHMM);
    else if (type == HMMType::GaussianMixtureModelHMM)
      gmmHMM = new HMM<gmm::GMM>(*other.gmmHMM);

    return *this;
  }

  //! Clean memory.
  ~HMMModel()
  {
    delete discreteHMM;
    delete gaussianHMM;
    delete gmmHMM;
  }

  /**
   * Given a functor type, perform that functor with the optional extra info on
   * the HMM.
   */
  template<typename ActionType,
           typename ExtraInfoType>
  void PerformAction(ExtraInfoType* x)
  {
    if (type == HMMType::DiscreteHMM)
      ActionType::Apply(*discreteHMM, x);
    else if (type == HMMType::GaussianHMM)
      ActionType::Apply(*gaussianHMM, x);
    else if (type == HMMType::GaussianMixtureModelHMM)
      ActionType::Apply(*gmmHMM, x);
  }

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(type, "type");

    // If necessary, clean memory.
    if (Archive::is_loading::value)
    {
      delete discreteHMM;
      delete gaussianHMM;
      delete gmmHMM;

      discreteHMM = NULL;
      gaussianHMM = NULL;
      gmmHMM = NULL;
    }

    if (type == HMMType::DiscreteHMM)
      ar & data::CreateNVP(discreteHMM, "discreteHMM");
    else if (type == HMMType::GaussianHMM)
      ar & data::CreateNVP(gaussianHMM, "gaussianHMM");
    else if (type == HMMType::GaussianMixtureModelHMM)
      ar & data::CreateNVP(gmmHMM, "gmmHMM");
  }
};

} // namespace hmm
} // namespace mlpack

#endif
