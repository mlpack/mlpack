
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_adaboost_adaboost_model.hpp:

Program Listing for File adaboost_model.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_adaboost_adaboost_model.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/adaboost/adaboost_model.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ADABOOST_ADABOOST_MODEL_HPP
   #define MLPACK_METHODS_ADABOOST_ADABOOST_MODEL_HPP
   
   #include <mlpack/core.hpp>
   
   // Use forward declaration instead of include to accelerate compilation.
   class AdaBoost;
   
   namespace mlpack {
   namespace adaboost {
   
   class AdaBoostModel
   {
    public:
     enum WeakLearnerTypes
     {
       DECISION_STUMP,
       PERCEPTRON
     };
   
    private:
     arma::Col<size_t> mappings;
     size_t weakLearnerType;
     AdaBoost<tree::ID3DecisionStump>* dsBoost;
     AdaBoost<perceptron::Perceptron<>>* pBoost;
     size_t dimensionality;
   
    public:
     AdaBoostModel();
   
     AdaBoostModel(const arma::Col<size_t>& mappings,
                   const size_t weakLearnerType);
   
     AdaBoostModel(const AdaBoostModel& other);
   
     AdaBoostModel(AdaBoostModel&& other);
   
     AdaBoostModel& operator=(const AdaBoostModel& other);
   
     AdaBoostModel& operator=(AdaBoostModel&& other);
   
     ~AdaBoostModel();
   
     const arma::Col<size_t>& Mappings() const { return mappings; }
     arma::Col<size_t>& Mappings() { return mappings; }
   
     size_t WeakLearnerType() const { return weakLearnerType; }
     size_t& WeakLearnerType() { return weakLearnerType; }
   
     size_t Dimensionality() const { return dimensionality; }
     size_t& Dimensionality() { return dimensionality; }
   
     void Train(const arma::mat& data,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const size_t iterations,
                const double tolerance);
   
     void Classify(const arma::mat& testData,
                   arma::Row<size_t>& predictions);
   
     void Classify(const arma::mat& testData,
                   arma::Row<size_t>& predictions,
                   arma::mat& probabilities);
   
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
   
   } // namespace adaboost
   } // namespace mlpack
   
   #endif
