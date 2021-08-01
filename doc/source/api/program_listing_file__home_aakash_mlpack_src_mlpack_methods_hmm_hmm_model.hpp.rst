
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm_model.hpp:

Program Listing for File hmm_model.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_hmm_hmm_model.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/hmm/hmm_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_HMM_HMM_MODEL_HPP
   #define MLPACK_METHODS_HMM_HMM_MODEL_HPP
   
   #include "hmm.hpp"
   #include <mlpack/methods/gmm/gmm.hpp>
   #include <mlpack/methods/gmm/diagonal_gmm.hpp>
   
   namespace mlpack {
   namespace hmm {
   
   enum HMMType : char
   {
     DiscreteHMM = 0,
     GaussianHMM,
     GaussianMixtureModelHMM,
     DiagonalGaussianMixtureModelHMM
   };
   
   class HMMModel
   {
    private:
     HMMType type;
     HMM<distribution::DiscreteDistribution>* discreteHMM;
     HMM<distribution::GaussianDistribution>* gaussianHMM;
     HMM<gmm::GMM>* gmmHMM;
     HMM<gmm::DiagonalGMM>* diagGMMHMM;
   
    public:
     HMMModel(const HMMType type = HMMType::DiscreteHMM) :
         type(type),
         discreteHMM(NULL),
         gaussianHMM(NULL),
         gmmHMM(NULL),
         diagGMMHMM(NULL)
     {
       if (type == HMMType::DiscreteHMM)
         discreteHMM = new HMM<distribution::DiscreteDistribution>();
       else if (type == HMMType::GaussianHMM)
         gaussianHMM = new HMM<distribution::GaussianDistribution>();
       else if (type == HMMType::GaussianMixtureModelHMM)
         gmmHMM = new HMM<gmm::GMM>();
       else if (type == HMMType::DiagonalGaussianMixtureModelHMM)
         diagGMMHMM = new HMM<gmm::DiagonalGMM>();
     }
   
     HMMModel(const HMMModel& other) :
         type(other.type),
         discreteHMM(NULL),
         gaussianHMM(NULL),
         gmmHMM(NULL),
         diagGMMHMM(NULL)
     {
       if (type == HMMType::DiscreteHMM)
         discreteHMM =
             new HMM<distribution::DiscreteDistribution>(*other.discreteHMM);
       else if (type == HMMType::GaussianHMM)
         gaussianHMM =
             new HMM<distribution::GaussianDistribution>(*other.gaussianHMM);
       else if (type == HMMType::GaussianMixtureModelHMM)
         gmmHMM = new HMM<gmm::GMM>(*other.gmmHMM);
       else if (type == HMMType::DiagonalGaussianMixtureModelHMM)
         diagGMMHMM = new HMM<gmm::DiagonalGMM>(*other.diagGMMHMM);
     }
   
     HMMModel(HMMModel&& other) :
         type(other.type),
         discreteHMM(other.discreteHMM),
         gaussianHMM(other.gaussianHMM),
         gmmHMM(other.gmmHMM),
         diagGMMHMM(other.diagGMMHMM)
     {
       other.type = HMMType::DiscreteHMM;
       other.discreteHMM = new HMM<distribution::DiscreteDistribution>();
       other.gaussianHMM = NULL;
       other.gmmHMM = NULL;
       other.diagGMMHMM = NULL;
     }
   
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
             new HMM<distribution::DiscreteDistribution>(*other.discreteHMM);
       else if (type == HMMType::GaussianHMM)
         gaussianHMM =
             new HMM<distribution::GaussianDistribution>(*other.gaussianHMM);
       else if (type == HMMType::GaussianMixtureModelHMM)
         gmmHMM = new HMM<gmm::GMM>(*other.gmmHMM);
       else if (type == HMMType::DiagonalGaussianMixtureModelHMM)
         diagGMMHMM = new HMM<gmm::DiagonalGMM>(*other.diagGMMHMM);
   
       return *this;
     }
   
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
         other.discreteHMM = new HMM<distribution::DiscreteDistribution>();
         other.gaussianHMM = nullptr;
         other.gmmHMM = nullptr;
         other.diagGMMHMM = nullptr;
       }
       return *this;
     }
   
     ~HMMModel()
     {
       delete discreteHMM;
       delete gaussianHMM;
       delete gmmHMM;
       delete diagGMMHMM;
     }
   
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
       else if (type == HMMType::DiagonalGaussianMixtureModelHMM)
         ActionType::Apply(*diagGMMHMM, x);
     }
   
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
   
     HMM<distribution::DiscreteDistribution>* DiscreteHMM() { return discreteHMM; }
     HMM<distribution::GaussianDistribution>* GaussianHMM() { return gaussianHMM; }
     HMM<gmm::GMM>* GMMHMM() { return gmmHMM; }
     HMM<gmm::DiagonalGMM>* DiagGMMHMM() { return diagGMMHMM; }
   };
   
   } // namespace hmm
   } // namespace mlpack
   
   #endif
