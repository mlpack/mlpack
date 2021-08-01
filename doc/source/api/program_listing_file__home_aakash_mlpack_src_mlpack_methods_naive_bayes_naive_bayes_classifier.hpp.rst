
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_naive_bayes_naive_bayes_classifier.hpp:

Program Listing for File naive_bayes_classifier.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_naive_bayes_naive_bayes_classifier.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/naive_bayes/naive_bayes_classifier.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_HPP
   #define MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace naive_bayes  {
   
   template<typename ModelMatType = arma::mat>
   class NaiveBayesClassifier
   {
    public:
     // Convenience typedef.
     typedef typename ModelMatType::elem_type ElemType;
   
     template<typename MatType>
     NaiveBayesClassifier(const MatType& data,
                          const arma::Row<size_t>& labels,
                          const size_t numClasses,
                          const bool incrementalVariance = false,
                          const double epsilon = 1e-10);
   
     NaiveBayesClassifier(const size_t dimensionality = 0,
                          const size_t numClasses = 0,
                          const double epsilon = 1e-10);
   
     template<typename MatType>
     void Train(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const bool incremental = true);
   
     template<typename VecType>
     void Train(const VecType& point, const size_t label);
   
     template<typename VecType>
     size_t Classify(const VecType& point) const;
   
     template<typename VecType, typename ProbabilitiesVecType>
     void Classify(const VecType& point,
                   size_t& prediction,
                   ProbabilitiesVecType& probabilities) const;
   
     template<typename MatType>
     void Classify(const MatType& data,
                   arma::Row<size_t>& predictions) const;
   
     template<typename MatType, typename ProbabilitiesMatType>
     void Classify(const MatType& data,
                   arma::Row<size_t>& predictions,
                   ProbabilitiesMatType& probabilities) const;
   
     const ModelMatType& Means() const { return means; }
     ModelMatType& Means() { return means; }
   
     const ModelMatType& Variances() const { return variances; }
     ModelMatType& Variances() { return variances; }
   
     const ModelMatType& Probabilities() const { return probabilities; }
     ModelMatType& Probabilities() { return probabilities; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */);
   
    private:
     ModelMatType means;
     ModelMatType variances;
     ModelMatType probabilities;
     size_t trainingPoints;
     double epsilon;
   
     template<typename MatType>
     void LogLikelihood(const MatType& data,
                        ModelMatType& logLikelihoods) const;
   };
   
   } // namespace naive_bayes
   } // namespace mlpack
   
   // Include implementation.
   #include "naive_bayes_classifier_impl.hpp"
   
   #endif
