
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_dists_bernoulli_distribution_impl.hpp:

Program Listing for File bernoulli_distribution_impl.hpp
========================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_dists_bernoulli_distribution_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/dists/bernoulli_distribution_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_DISTRIBUTIONS_BERNOULLI_DISTRIBUTION_IMPL_HPP
   #define MLPACK_METHODS_ANN_DISTRIBUTIONS_BERNOULLI_DISTRIBUTION_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "bernoulli_distribution.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename DataType>
   BernoulliDistribution<DataType>::BernoulliDistribution() :
       applyLogistic(true),
       eps(1e-10)
   {
     // Nothing to do here.
   }
   
   template<typename DataType>
   BernoulliDistribution<DataType>::BernoulliDistribution(
       const DataType& param,
       const bool applyLogistic,
       const double eps) :
       logits(param),
       applyLogistic(applyLogistic),
       eps(eps)
   {
     if (applyLogistic)
     {
       LogisticFunction::Fn(logits, probability);
     }
     else
     {
       probability = arma::mat(logits.memptr(), logits.n_rows,
           logits.n_cols, false, false);
     }
   }
   
   template<typename DataType>
   DataType BernoulliDistribution<DataType>::Sample() const
   {
     DataType sample = arma::randu<DataType>
         (probability.n_rows, probability.n_cols);
   
     for (size_t i = 0; i < sample.n_elem; ++i)
         sample(i) = sample(i) < probability(i);
   
     return sample;
   }
   
   template<typename DataType>
   double BernoulliDistribution<DataType>::LogProbability(
       const DataType& observation) const
   {
     return arma::accu(arma::log(probability + eps) % observation +
         arma::log(1 - probability + eps) % (1 - observation)) /
         observation.n_cols;
   }
   
   template<typename DataType>
   void BernoulliDistribution<DataType>::LogProbBackward(
       const DataType& observation, DataType& output) const
   {
     if (!applyLogistic)
     {
       output = observation / (probability + eps) - (1 - observation) /
           (1 - probability + eps);
     }
     else
     {
       LogisticFunction::Deriv(probability, output);
       output = (observation / (probability + eps) - (1 - observation) /
           (1 - probability + eps)) % output;
     }
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
