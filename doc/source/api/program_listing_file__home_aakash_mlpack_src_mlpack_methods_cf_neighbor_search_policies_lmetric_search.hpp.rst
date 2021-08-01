
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_neighbor_search_policies_lmetric_search.hpp:

Program Listing for File lmetric_search.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_neighbor_search_policies_lmetric_search.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/neighbor_search_policies/lmetric_search.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_LMETRIC_SEARCH_HPP
   #define MLPACK_METHODS_CF_LMETRIC_SEARCH_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/neighbor_search/neighbor_search.hpp>
   #include <mlpack/core/metrics/lmetric.hpp>
   
   namespace mlpack {
   namespace cf {
   
   template<int TPower>
   class LMetricSearch
   {
    public:
     using NeighborSearchType = neighbor::NeighborSearch<
         neighbor::NearestNeighborSort,
         metric::LMetric<TPower, true>>;
   
     LMetricSearch(const arma::mat& referenceSet) : neighborSearch(referenceSet)
     { }
   
     void Search(const arma::mat& query, const size_t k,
                 arma::Mat<size_t>& neighbors, arma::mat& similarities)
     {
       neighborSearch.Search(query, k, neighbors, similarities);
   
       // Calculate similarities from L_p distance. We restrict that similarities
       // are not larger than one.
       similarities = 1.0 / (1.0 + similarities);
     }
   
    private:
     NeighborSearchType neighborSearch;
   };
   
   using EuclideanSearch = LMetricSearch<2>;
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
