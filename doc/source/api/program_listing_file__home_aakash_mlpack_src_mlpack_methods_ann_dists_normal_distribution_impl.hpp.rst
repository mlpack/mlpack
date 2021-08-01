
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_dists_normal_distribution_impl.hpp:

Program Listing for File normal_distribution_impl.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_dists_normal_distribution_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/dists/normal_distribution_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_IMPL_HPP
   #define MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "normal_distribution.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename DataType>
   NormalDistribution<DataType>::NormalDistribution()
   {
     // Nothing to do here.
   }
   
   template<typename DataType>
   NormalDistribution<DataType>::NormalDistribution(
       const DataType& mean,
       const DataType& sigma) :
       mean(mean),
       sigma(sigma)
   {
     // Nothing to do here.
   }
   
   template<typename DataType>
   DataType NormalDistribution<DataType>::Sample() const
   {
     return sigma * arma::randn<DataType>(mean.n_elem) + mean;
   }
   
   template<typename DataType>
   DataType NormalDistribution<DataType>::LogProbability(
       const DataType& observation) const
   {
     const DataType v1 = arma::log(sigma) + std::log(std::sqrt(2 * M_PI));
     const DataType v2 = arma::square(observation - mean) /
                         (2 * arma::square(sigma));
     return  (-v1 - v2);
   }
   
   template<typename DataType>
   void NormalDistribution<DataType>::ProbBackward(
       const DataType& observation,
       DataType& dmu,
       DataType& dsigma) const
   {
     dmu = (observation - mean) / (arma::square(sigma)) % Probability(observation);
     dsigma = (- 1.0 / sigma +
               (arma::square(observation - mean) / arma::pow(sigma, 3)))
               % Probability(observation);
   }
   
   template<typename DataType>
   template<typename Archive>
   void NormalDistribution<DataType>::serialize(Archive& ar,
                                                const uint32_t /* version */)
   {
     ar(CEREAL_NVP(mean));
     ar(CEREAL_NVP(sigma));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
