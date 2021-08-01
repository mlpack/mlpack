
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_adaboost_adaboost.hpp:

Program Listing for File adaboost.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_adaboost_adaboost.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/adaboost/adaboost.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

     author = {Schapire, Robert E. and Singer, Yoram},
     title = {Improved Boosting Algorithms Using Confidence-rated Predictions},
     journal = {Machine Learning},
     volume = {37},
     number = {3},
     month = dec,
     year = {1999},
     issn = {0885-6125},
     pages = {297--336},
   }
   
   #ifndef MLPACK_METHODS_ADABOOST_ADABOOST_HPP
   #define MLPACK_METHODS_ADABOOST_ADABOOST_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/perceptron/perceptron.hpp>
   #include <mlpack/methods/decision_tree/decision_tree.hpp>
   
   namespace mlpack {
   namespace adaboost {
   
   template<typename WeakLearnerType = mlpack::perceptron::Perceptron<>,
            typename MatType = arma::mat>
   class AdaBoost
   {
    public:
     AdaBoost(const MatType& data,
              const arma::Row<size_t>& labels,
              const size_t numClasses,
              const WeakLearnerType& other,
              const size_t iterations = 100,
              const double tolerance = 1e-6);
   
     AdaBoost(const double tolerance = 1e-6);
   
     double Tolerance() const { return tolerance; }
     double& Tolerance() { return tolerance; }
   
     size_t NumClasses() const { return numClasses; }
   
     size_t WeakLearners() const { return alpha.size(); }
   
     double Alpha(const size_t i) const { return alpha[i]; }
     double& Alpha(const size_t i) { return alpha[i]; }
   
     const WeakLearnerType& WeakLearner(const size_t i) const { return wl[i]; }
     WeakLearnerType& WeakLearner(const size_t i) { return wl[i]; }
   
     double Train(const MatType& data,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const WeakLearnerType& learner,
                  const size_t iterations = 100,
                  const double tolerance = 1e-6);
   
     void Classify(const MatType& test,
                   arma::Row<size_t>& predictedLabels,
                   arma::mat& probabilities);
   
     void Classify(const MatType& test,
                   arma::Row<size_t>& predictedLabels);
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     size_t numClasses;
     // The tolerance for change in rt and when to stop.
     double tolerance;
   
     std::vector<WeakLearnerType> wl;
     std::vector<double> alpha;
   }; // class AdaBoost
   
   } // namespace adaboost
   } // namespace mlpack
   
   // Include implementation.
   #include "adaboost_impl.hpp"
   
   #endif
