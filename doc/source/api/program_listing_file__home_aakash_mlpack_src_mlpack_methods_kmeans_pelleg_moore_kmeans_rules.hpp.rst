
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_kmeans_pelleg_moore_kmeans_rules.hpp:

Program Listing for File pelleg_moore_kmeans_rules.hpp
======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_kmeans_pelleg_moore_kmeans_rules.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/kmeans/pelleg_moore_kmeans_rules.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_RULES_HPP
   #define MLPACK_METHODS_KMEANS_PELLEG_MOORE_KMEANS_RULES_HPP
   
   namespace mlpack {
   namespace kmeans {
   
   template<typename MetricType, typename TreeType>
   class PellegMooreKMeansRules
   {
    public:
     PellegMooreKMeansRules(const typename TreeType::Mat& dataset,
                            const arma::mat& centroids,
                            arma::mat& newCentroids,
                            arma::Col<size_t>& counts,
                            MetricType& metric);
   
     double BaseCase(const size_t queryIndex, const size_t referenceIndex);
   
     double Score(const size_t queryIndex, TreeType& referenceNode);
   
     double Rescore(const size_t queryIndex,
                    TreeType& referenceNode,
                    const double oldScore);
   
     size_t DistanceCalculations() const { return distanceCalculations; }
     size_t& DistanceCalculations() { return distanceCalculations; }
   
    private:
     const typename TreeType::Mat& dataset;
     const arma::mat& centroids;
     arma::mat& newCentroids;
     arma::Col<size_t>& counts;
     MetricType& metric;
   
     size_t distanceCalculations;
   };
   
   } // namespace kmeans
   } // namespace mlpack
   
   // Include implementation.
   #include "pelleg_moore_kmeans_rules_impl.hpp"
   
   #endif
