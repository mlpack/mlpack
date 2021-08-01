
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_metrics_mahalanobis_distance.hpp:

Program Listing for File mahalanobis_distance.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_metrics_mahalanobis_distance.hpp>` (``/home/aakash/mlpack/src/mlpack/core/metrics/mahalanobis_distance.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_METRICS_MAHALANOBIS_DISTANCE_HPP
   #define MLPACK_CORE_METRICS_MAHALANOBIS_DISTANCE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace metric {
   
   template<bool TakeRoot = true>
   class MahalanobisDistance
   {
    public:
     MahalanobisDistance() { }
   
     MahalanobisDistance(const size_t dimensionality) :
         covariance(arma::eye<arma::mat>(dimensionality, dimensionality)) { }
   
     MahalanobisDistance(arma::mat covariance) :
         covariance(std::move(covariance)) { }
   
     template<typename VecTypeA, typename VecTypeB>
     double Evaluate(const VecTypeA& a, const VecTypeB& b);
   
     const arma::mat& Covariance() const { return covariance; }
   
     arma::mat& Covariance() { return covariance; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t version);
   
    private:
     arma::mat covariance;
   };
   
   } // namespace metric
   } // namespace mlpack
   
   #include "mahalanobis_distance_impl.hpp"
   
   #endif
