
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_adaboost_adaboost_model.cpp:

Program Listing for File adaboost_model.cpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_adaboost_adaboost_model.cpp>` (``/home/aakash/mlpack/src/mlpack/methods/adaboost/adaboost_model.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include "adaboost.hpp"
   #include "adaboost_model.hpp"
   
   using namespace mlpack;
   using namespace std;
   using namespace arma;
   using namespace mlpack::adaboost;
   using namespace mlpack::tree;
   using namespace mlpack::perceptron;
   
   AdaBoostModel::AdaBoostModel() :
       weakLearnerType(0),
       dsBoost(NULL),
       pBoost(NULL),
       dimensionality(0)
   {
     // Nothing to do.
   }
   
   AdaBoostModel::AdaBoostModel(
       const Col<size_t>& mappings,
       const size_t weakLearnerType) :
       mappings(mappings),
       weakLearnerType(weakLearnerType),
       dsBoost(NULL),
       pBoost(NULL),
       dimensionality(0)
   {
     // Nothing to do.
   }
   
   AdaBoostModel::AdaBoostModel(const AdaBoostModel& other) :
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
   
   AdaBoostModel::AdaBoostModel(AdaBoostModel&& other) :
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
   
   AdaBoostModel& AdaBoostModel::operator=(const AdaBoostModel& other)
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
   
   AdaBoostModel& AdaBoostModel::operator=(AdaBoostModel&& other)
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
   
   AdaBoostModel::~AdaBoostModel()
   {
     delete dsBoost;
     delete pBoost;
   }
   
   void AdaBoostModel::Train(const mat& data,
                             const Row<size_t>& labels,
                             const size_t numClasses,
                             const size_t iterations,
                             const double tolerance)
   {
     dimensionality = data.n_rows;
     if (weakLearnerType == WeakLearnerTypes::DECISION_STUMP)
     {
       delete dsBoost;
       ID3DecisionStump ds(data, labels, max(labels) + 1);
       dsBoost = new AdaBoost<ID3DecisionStump>(data, labels, numClasses, ds,
           iterations, tolerance);
     }
     else if (weakLearnerType == WeakLearnerTypes::PERCEPTRON)
     {
       delete pBoost;
       Perceptron<> p(data, labels, max(labels) + 1);
       pBoost = new AdaBoost<Perceptron<>>(data, labels, numClasses, p, iterations,
           tolerance);
     }
   }
   
   void AdaBoostModel::Classify(const mat& testData,
                                Row<size_t>& predictions,
                                mat& probabilities)
   {
     if (weakLearnerType == WeakLearnerTypes::DECISION_STUMP)
       dsBoost->Classify(testData, predictions, probabilities);
     else if (weakLearnerType == WeakLearnerTypes::PERCEPTRON)
       pBoost->Classify(testData, predictions, probabilities);
   }
   
   void AdaBoostModel::Classify(const mat& testData,
                                Row<size_t>& predictions)
   {
     if (weakLearnerType == WeakLearnerTypes::DECISION_STUMP)
       dsBoost->Classify(testData, predictions);
     else if (weakLearnerType == WeakLearnerTypes::PERCEPTRON)
       pBoost->Classify(testData, predictions);
   }
